"""Statistical primitives for rebuttal Experiment 3.

The independent sampling unit is always a complete Hanabi game.  Arrays that
represent games therefore have shape ``[game, seat, dimension]`` and all
bootstrap/permutation helpers resample the leading axis only.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_JAX_BILINEAR = None


def stable_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    raw = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(raw)
    adjusted = np.empty_like(raw)
    running = 0.0
    m = len(raw)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * raw[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def average_ranks(values: Sequence[float], *, descending: bool = True) -> np.ndarray:
    """Return one-based average ranks, including exact ties."""
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(-x if descending else x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[order[end]] == x[order[start]]:
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def compatibility_effects(
    bc_on_bc: np.ndarray,
    bc_on_latent: np.ndarray,
    latent_on_bc: np.ndarray,
    latent_on_latent: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the corrected 2x2 interaction and secondary effect sizes.

    ``d_bc`` and ``d_latent`` use the same suite contrast (BC minus latent).
    Thus compatibility is their difference, while their average is suite
    easiness.  The old appendix reverses the latent contrast and consequently
    computes ``d_bc + d_latent == 2 * easiness``.
    """
    d_bc = np.asarray(bc_on_bc) - np.asarray(bc_on_latent)
    d_latent = np.asarray(latent_on_bc) - np.asarray(latent_on_latent)
    interaction = d_bc - d_latent
    easiness = (d_bc + d_latent) / 2.0
    return {
        "d_bc": d_bc,
        "d_latent": d_latent,
        "interaction": interaction,
        "easiness": easiness,
        "compatibility_magnitude": interaction / 2.0,
        "old_appendix_arrangement": d_bc + d_latent,
    }


def bootstrap_means(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 1_000,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    rng = np.random.default_rng(seed)
    out = np.empty((replicates, x.shape[1]), dtype=np.float64)
    for start in range(0, replicates, chunk_size):
        end = min(start + chunk_size, replicates)
        indices = rng.integers(0, x.shape[0], size=(end - start, x.shape[0]))
        out[start:end] = x[indices].mean(axis=1)
    return out


def bootstrap_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    draws = bootstrap_means(values, replicates=replicates, seed=seed)[:, 0]
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975]))


def sign_flip_pvalue(
    values: np.ndarray,
    *,
    permutations: int,
    seed: int,
    two_sided: bool = True,
    chunk_size: int = 2_000,
) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    observed = float(x.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, permutations, chunk_size):
        count = min(chunk_size, permutations - start)
        signs = rng.integers(0, 2, size=(count, len(x)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        null = (signs @ x) / len(x)
        if two_sided:
            exceed += int(np.count_nonzero(np.abs(null) >= abs(observed)))
        else:
            exceed += int(np.count_nonzero(null >= observed))
    return (exceed + 1.0) / (permutations + 1.0)


def bootstrap_suite_means(
    suite_scores: Mapping[str, np.ndarray],
    *,
    replicates: int,
    seed: int,
    chunk_size: int = 500,
) -> dict[str, np.ndarray]:
    """Bootstrap several [game, method] score matrices with common game indices."""
    arrays = {name: np.asarray(value, dtype=np.float64) for name, value in suite_scores.items()}
    shapes = {value.shape for value in arrays.values()}
    if len(shapes) != 1:
        raise ValueError(f"Suite score shapes differ: {shapes}")
    n_games, n_methods = next(iter(shapes))
    rng = np.random.default_rng(seed)
    output = {name: np.empty((replicates, n_methods), dtype=np.float64) for name in arrays}
    for start in range(0, replicates, chunk_size):
        end = min(start + chunk_size, replicates)
        indices = rng.integers(0, n_games, size=(end - start, n_games))
        for name, value in arrays.items():
            output[name][start:end] = value[indices].mean(axis=1)
    return output


def rank_summary(
    point_scores: np.ndarray,
    bootstrap_scores: np.ndarray,
    method_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    point_ranks = average_ranks(point_scores)
    boot_ranks = np.stack([average_ranks(row) for row in bootstrap_scores])
    result: dict[str, dict[str, Any]] = {}
    for index, method_id in enumerate(method_ids):
        ranks = boot_ranks[:, index]
        unique, counts = np.unique(ranks, return_counts=True)
        result[method_id] = {
            "point_score": float(point_scores[index]),
            "point_rank": float(point_ranks[index]),
            "rank_probability_mass": {
                f"{rank:g}": float(count / len(ranks))
                for rank, count in zip(unique, counts, strict=True)
            },
            "median_rank": float(np.median(ranks)),
            "rank_interval95": [float(v) for v in np.quantile(ranks, [0.025, 0.975])],
            "top3_probability": float(np.mean(ranks <= 3.0)),
            "top5_probability": float(np.mean(ranks <= 5.0)),
        }
    return result


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    out = np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2.0 * (x @ y.T)
    return np.maximum(out, 0.0)


def game_rbf_kernel(
    x: np.ndarray,
    y: np.ndarray,
    bandwidths: Sequence[float],
) -> np.ndarray:
    """Seat-averaged game kernel, returned as [bandwidth, x_game, y_game]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 3 or y.ndim != 3 or x.shape[1] != 2 or y.shape[1] != 2:
        raise ValueError(f"Expected [game, 2, dim], got {x.shape} and {y.shape}")
    sigmas = np.asarray(bandwidths, dtype=np.float64)
    if np.any(sigmas <= 0):
        raise ValueError(f"Bandwidths must be positive: {sigmas}")
    result = np.zeros((len(sigmas), len(x), len(y)), dtype=np.float64)
    for seat_x in range(2):
        for seat_y in range(2):
            sq = _squared_distances(x[:, seat_x], y[:, seat_y])
            result += np.exp(-sq[None, :, :] / (2.0 * sigmas[:, None, None] ** 2))
    return (result / 4.0).astype(np.float32)


def game_energy_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Seat-averaged Euclidean distance between complete games."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    result = np.zeros((len(x), len(y)), dtype=np.float64)
    for seat_x in range(2):
        for seat_y in range(2):
            result += np.sqrt(_squared_distances(x[:, seat_x], y[:, seat_y]))
    return (result / 4.0).astype(np.float32)


def median_distinct_distance(seat_vectors: np.ndarray) -> float:
    x = np.asarray(seat_vectors, dtype=np.float64)
    distances = np.sqrt(_squared_distances(x, x))
    upper = distances[np.triu_indices(len(x), k=1)]
    median = float(np.median(upper))
    if not np.isfinite(median) or median <= 0:
        raise ValueError(f"Invalid development bandwidth median: {median}")
    return median


def unbiased_within(matrix: np.ndarray) -> float:
    k = np.asarray(matrix, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or len(k) < 2:
        raise ValueError(f"Expected square matrix with at least two games, got {k.shape}")
    return float((k.sum() - np.trace(k)) / (len(k) * (len(k) - 1)))


def cross_mean(matrix: np.ndarray) -> float:
    return float(np.asarray(matrix, dtype=np.float64).mean())


def equal_mixture_expectation(
    components: Sequence[str],
    pair_values: Mapping[tuple[str, str], np.ndarray | float],
    *,
    within: bool,
) -> np.ndarray:
    """Combine component expectations with exactly equal component weights."""
    names = list(components)
    weight = 1.0 / len(names)
    result: np.ndarray | float = 0.0
    if within:
        for left in names:
            result = result + weight * weight * pair_values[(left, left)]
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                result = result + 2.0 * weight * weight * pair_values[tuple(sorted((left, right)))]
    else:
        raise ValueError("Use cross_mixture_expectation for two different suites")
    return np.asarray(result)


def cross_mixture_expectation(
    left_components: Sequence[str],
    right_components: Sequence[str],
    pair_values: Mapping[tuple[str, str], np.ndarray | float],
) -> np.ndarray:
    result: np.ndarray | float = 0.0
    weight = 1.0 / (len(left_components) * len(right_components))
    for left in left_components:
        for right in right_components:
            result = result + weight * pair_values[tuple(sorted((left, right)))]
    return np.asarray(result)


def suite_mmd_from_expectations(
    left_components: Sequence[str],
    right_components: Sequence[str],
    pair_values: Mapping[tuple[str, str], np.ndarray | float],
) -> np.ndarray:
    left = equal_mixture_expectation(left_components, pair_values, within=True)
    right = equal_mixture_expectation(right_components, pair_values, within=True)
    cross = cross_mixture_expectation(left_components, right_components, pair_values)
    return left + right - 2.0 * cross


def suite_energy_from_expectations(
    left_components: Sequence[str],
    right_components: Sequence[str],
    pair_values: Mapping[tuple[str, str], np.ndarray | float],
) -> np.ndarray:
    left = equal_mixture_expectation(left_components, pair_values, within=True)
    right = equal_mixture_expectation(right_components, pair_values, within=True)
    cross = cross_mixture_expectation(left_components, right_components, pair_values)
    return 2.0 * cross - left - right


def bootstrap_counts(
    n_games: int,
    replicates: int,
    *,
    seed: int,
) -> np.ndarray:
    """Return multinomial game counts; seats never appear on this sampling axis."""
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n_games, size=(replicates, n_games), dtype=np.int32)
    counts = np.zeros((replicates, n_games), dtype=np.float32)
    rows = np.repeat(np.arange(replicates, dtype=np.int32), n_games)
    np.add.at(counts, (rows, indices.reshape(-1)), 1.0)
    return counts


def _bilinear_rows(
    left: np.ndarray,
    matrix: np.ndarray,
    right: np.ndarray,
    *,
    use_jax: bool,
    chunk_size: int = 1_000,
) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    matrix = np.asarray(matrix, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if use_jax:
        global _JAX_BILINEAR
        import jax
        import jax.numpy as jnp

        if _JAX_BILINEAR is None:
            _JAX_BILINEAR = jax.jit(lambda a, k, b: jnp.einsum("bi,ij,bj->b", a, k, b))

        k_device = jax.device_put(matrix)
        output = np.empty(len(left), dtype=np.float64)
        for start in range(0, len(left), chunk_size):
            end = min(start + chunk_size, len(left))
            output[start:end] = np.asarray(
                _JAX_BILINEAR(
                    jax.device_put(left[start:end]),
                    k_device,
                    jax.device_put(right[start:end]),
                )
            )
        return output
    return np.einsum("bi,ij,bj->b", left, matrix, right, optimize=True)


def resampled_pair_expectations(
    matrices: Mapping[tuple[str, str], np.ndarray],
    group_sizes: Mapping[str, int],
    *,
    replicates: int,
    seed: int,
    use_jax: bool,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], np.ndarray]]:
    """Compute point and stratified whole-game bootstrap pair expectations."""
    counts = {
        name: bootstrap_counts(
            size,
            replicates,
            seed=stable_seed(seed, f"bootstrap:{name}"),
        )
        for name, size in group_sizes.items()
    }
    point: dict[tuple[str, str], float] = {}
    draws: dict[tuple[str, str], np.ndarray] = {}
    for raw_key, raw_matrix in matrices.items():
        left, right = raw_key
        key = tuple(sorted(raw_key))
        matrix = np.asarray(raw_matrix)
        if raw_key != key:
            matrix = matrix.T
            left, right = key
        if left == right:
            n = group_sizes[left]
            point[key] = unbiased_within(matrix)
            quadratic = _bilinear_rows(counts[left], matrix, counts[left], use_jax=use_jax)
            diagonal = counts[left] @ np.diag(matrix).astype(np.float32)
            draws[key] = (quadratic - diagonal) / (n * (n - 1))
        else:
            point[key] = cross_mean(matrix)
            draws[key] = _bilinear_rows(counts[left], matrix, counts[right], use_jax=use_jax) / (
                group_sizes[left] * group_sizes[right]
            )
    return point, draws


def flattened_unbiased_mmd(kernel: np.ndarray, n_left: int) -> float:
    k = np.asarray(kernel, dtype=np.float64)
    a = k[:n_left, :n_left]
    b = k[n_left:, n_left:]
    ab = k[:n_left, n_left:]
    return unbiased_within(a) + unbiased_within(b) - 2.0 * ab.mean()


def permutation_mmd_pvalue(
    kernel: np.ndarray,
    n_left: int,
    *,
    permutations: int,
    seed: int,
    use_jax: bool,
) -> tuple[float, float]:
    """Upper-tail whole-game label permutation test using the pooled U statistic."""
    k = np.asarray(kernel, dtype=np.float32)
    n_total = len(k)
    n_right = n_total - n_left
    if min(n_left, n_right) < 2:
        raise ValueError("Permutation groups need at least two games")
    observed = flattened_unbiased_mmd(k, n_left)
    rng = np.random.default_rng(seed)
    indicator = np.zeros((permutations, n_total), dtype=np.float32)
    for row in range(permutations):
        indicator[row, rng.permutation(n_total)[:n_left]] = 1.0

    q_left = _bilinear_rows(indicator, k, indicator, use_jax=use_jax)
    row_sums = k.sum(axis=1, dtype=np.float64)
    total = float(row_sums.sum())
    diag = np.diag(k).astype(np.float64)
    left_to_all = indicator @ row_sums
    diag_left = indicator @ diag
    q_right = total - 2.0 * left_to_all + q_left
    diag_right = float(diag.sum()) - diag_left
    cross = left_to_all - q_left
    null = (
        (q_left - diag_left) / (n_left * (n_left - 1))
        + (q_right - diag_right) / (n_right * (n_right - 1))
        - 2.0 * cross / (n_left * n_right)
    )
    pvalue = (1.0 + np.count_nonzero(null >= observed)) / (permutations + 1.0)
    return float(observed), float(pvalue)
