"""CPU checks for naming, checkpoint compatibility and numerical prior helpers."""

import ast
from pathlib import Path

import numpy as np
import pytest
import yaml

from ah2ac2.nn.checkpoint_utils import unwrap_vae_params
from ah2ac2.tools.prior_math import fit_best_gmm, fit_ppca, ppca_cov


@pytest.mark.parametrize("wrapper", ["VAE_0", "GAMMAVAE_0"])
def test_checkpoint_wrappers_preserve_the_parameter_tree(wrapper):
    tree = {"decoder": {"kernel": np.arange(6).reshape(2, 3)}}
    assert unwrap_vae_params({wrapper: tree}) is tree
    assert unwrap_vae_params(tree) is tree


def test_single_component_prior_and_ppca_remain_numerically_valid():
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(256, 4)) * np.array([3, 2, 1, 0.5])
    k, fitted, bic = fit_best_gmm(samples, kmax=1, seed=7)
    assert k == 1 and np.isfinite(bic)
    np.testing.assert_allclose(fitted.weights.sum(), 1)
    np.testing.assert_allclose(fitted.means[0], samples.mean(axis=0), atol=1e-9)
    assert np.linalg.eigvalsh(fitted.covs[0]).min() > 0
    basis, eigenvalues, noise = fit_ppca(samples, rank=2)
    covariance = ppca_cov(basis, eigenvalues, noise)
    assert covariance.shape == (4, 4)
    assert np.isfinite(covariance).all()
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_hydra_config_names_resolve_after_module_renaming():
    root = Path(__file__).resolve().parents[1] / "ah2ac2" / "training"
    for source in root.glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "main"
            ):
                keywords = {kw.arg: kw.value for kw in node.keywords}
                if "config_name" in keywords:
                    name = ast.literal_eval(keywords["config_name"])
                    assert (root / "config" / f"{name}.yaml").is_file()
    for source in (root / "config").glob("*.yaml"):
        config = yaml.safe_load(source.read_text())
        assert "gamma" not in config
        if "discount_factor" in config:
            assert 0 < config["discount_factor"] <= 1
