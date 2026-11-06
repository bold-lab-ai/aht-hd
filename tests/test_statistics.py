import numpy as np

from ah2ac2.tools.experiment3_stats import (
    average_ranks,
    bootstrap_counts,
    bootstrap_means,
    compatibility_effects,
    equal_mixture_expectation,
    game_rbf_kernel,
    holm_adjust,
    median_distinct_distance,
    unbiased_within,
)


def test_interaction_sign_separates_compatibility_from_easiness():
    compatible = compatibility_effects(
        bc_on_bc=np.array([4.0]),
        bc_on_latent=np.array([1.0]),
        latent_on_bc=np.array([1.0]),
        latent_on_latent=np.array([4.0]),
    )
    np.testing.assert_allclose(compatible["interaction"], 6.0)
    np.testing.assert_allclose(compatible["easiness"], 0.0)
    np.testing.assert_allclose(compatible["compatibility_magnitude"], 3.0)

    easy_bc_suite = compatibility_effects(
        bc_on_bc=np.array([4.0]),
        bc_on_latent=np.array([2.0]),
        latent_on_bc=np.array([4.0]),
        latent_on_latent=np.array([2.0]),
    )
    np.testing.assert_allclose(easy_bc_suite["interaction"], 0.0)
    np.testing.assert_allclose(easy_bc_suite["easiness"], 2.0)
    np.testing.assert_allclose(
        easy_bc_suite["old_appendix_arrangement"],
        2.0 * easy_bc_suite["easiness"],
    )


def test_game_bootstrap_counts_resample_whole_game_axis():
    counts = bootstrap_counts(7, 100, seed=3)
    assert counts.shape == (100, 7)
    np.testing.assert_allclose(counts.sum(axis=1), 7)
    assert np.all(counts >= 0)


def test_equal_suite_weighting_uses_components_not_component_row_counts():
    pair_values = {
        ("a", "a"): np.asarray(1.0),
        ("a", "b"): np.asarray(2.0),
        ("b", "b"): np.asarray(3.0),
    }
    # 1/4 * 1 + 1/2 * 2 + 1/4 * 3 = 2, independent of component sample sizes.
    value = equal_mixture_expectation(["a", "b"], pair_values, within=True)
    np.testing.assert_allclose(value, 2.0)


def test_unbiased_mmd_can_be_negative_and_is_not_truncated():
    x = np.array([[[-1.0], [-1.0]], [[1.0], [1.0]]])
    y = np.array([[[0.0], [0.0]], [[0.0], [0.0]]])
    k_xx = game_rbf_kernel(x, x, [1.0])[0]
    k_yy = game_rbf_kernel(y, y, [1.0])[0]
    k_xy = game_rbf_kernel(x, y, [1.0])[0]
    mmd2 = unbiased_within(k_xx) + unbiased_within(k_yy) - 2.0 * k_xy.mean()
    assert mmd2 < 0.0


def test_bandwidth_is_locked_from_development_vectors_only():
    development = np.array([[0.0], [2.0], [5.0]])
    locked = median_distinct_distance(development)
    audit_a = np.array([[100.0], [200.0]])
    audit_b = np.array([[-1_000.0], [1_000.0]])
    # Audit inputs do not enter the calibration function.
    assert locked == median_distinct_distance(development)
    assert not np.isclose(locked, median_distinct_distance(audit_a))
    assert not np.isclose(locked, median_distinct_distance(audit_b))


def test_holm_adjustment_is_monotone_in_sorted_order():
    raw = [0.04, 0.001, 0.03]
    adjusted = np.asarray(holm_adjust(raw))
    order = np.argsort(raw)
    assert np.all(adjusted[order][:-1] <= adjusted[order][1:])
    assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))


def test_average_rank_assigns_exact_ties_their_average_positions():
    np.testing.assert_allclose(
        average_ranks([10.0, 8.0, 8.0, 1.0]),
        [1.0, 2.5, 2.5, 4.0],
    )


def test_bootstrap_seed_is_deterministic():
    values = np.arange(20, dtype=np.float64)
    first = bootstrap_means(values, replicates=200, seed=11)
    second = bootstrap_means(values, replicates=200, seed=11)
    third = bootstrap_means(values, replicates=200, seed=12)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)
