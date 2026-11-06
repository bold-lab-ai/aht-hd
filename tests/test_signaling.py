import pytest

from ah2ac2.theory.signaling import effective_reference, enumerate_corruption_model


def test_effective_reference_has_the_registered_majority_margin():
    for p in (0.51, 0.7, 0.99):
        for eta in (0.0, 0.1, 0.25, 0.49, 0.5):
            q = effective_reference(p, eta)
            assert q == pytest.approx(p * (1.0 - eta) + (1.0 - p) * eta)
            assert q - 0.5 == pytest.approx((1.0 - 2.0 * eta) * (p - 0.5))
            if eta < 0.5:
                assert q > 0.5
            else:
                assert q == pytest.approx(0.5)


@pytest.mark.parametrize("model", ["coherent", "local"])
def test_exact_enumeration_recovers_pooled_sender_and_receiver_anchor(model):
    for p, eta in ((0.51, 0.0), (0.7, 0.2), (0.9, 0.49), (0.99, 0.5)):
        q = effective_reference(p, eta)
        result = enumerate_corruption_model(model, p, eta)

        assert result["total_probability"] == pytest.approx(1.0)
        assert list(result["sender_identity_conditionals"].values()) == pytest.approx([q, q])
        assert list(result["receiver_identity_conditionals"].values()) == pytest.approx([q, q])


def test_two_corruption_models_have_the_claimed_competence_and_branch_masses():
    p, eta = 0.8, 0.2
    coherent = enumerate_corruption_model("coherent", p, eta)
    local = enumerate_corruption_model("local", p, eta)

    assert coherent["reward"] == pytest.approx(1.0)
    assert local["reward"] == pytest.approx((1.0 - eta) ** 2 + eta**2)
    assert local["branch_masses"] == pytest.approx(
        {
            "identity_identity": p * (1.0 - eta) ** 2 + (1.0 - p) * eta**2,
            "identity_flip": eta * (1.0 - eta),
            "flip_identity": eta * (1.0 - eta),
            "flip_flip": (1.0 - p) * (1.0 - eta) ** 2 + p * eta**2,
        }
    )
