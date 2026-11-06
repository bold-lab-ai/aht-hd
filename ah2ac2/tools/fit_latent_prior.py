#!/usr/bin/env python3
"""
fit_latent_prior.py
-------------------
Encode latent posterior statistics (mu, logvar) from a trained VAE over
all trajectories, aggregate one vector per trajectory, then fit a *richer*
latent prior than N(\bar z, I).

What this script does
• Loads a saved VAE checkpoint (.safetensors) and its YAML config.
• Iterates the coordination dataset, reproducing the same deck→trajectory pipeline
  used in training to get (obs, act) sequences per player.
• Uses Flax `capture_intermediates` to extract the encoder Dense outputs that
  produce μ and log σ² even if the Module did not explicitly return them.
• Aggregates posterior means to one μ_i per trajectory (average over time, players).
• Fits a Gaussian Mixture Model (full covariance) with K ∈ [1..K_max], selects K* by BIC.
• Optionally fits a *structured* covariance: Σ = σ² I + U Λ Uᵀ (probabilistic PCA)
  either for the single‑Gaussian case or per‑component, providing a compact,
  regularized prior that captures anisotropy without overfitting.
• Saves the chosen prior to disk in a portable JSON (weights, means, Cholesky factors)
  plus a .npz of the raw Z and a CSV for easy sanity‑checking.

Notes on encoder capture
We rely on the fact that the two `nn.Dense(latent_dim)` layers in VAE's encoder
produce the pre‑aggregation μ_raw and logvar_raw of shape (T*B, latent_dim). We detect
these as the last two Dense outputs with the expected latent_dim and then reshape to
(T, B, latent_dim) and average over time as in training.
If you have modified VAE to expose `mu` and `lv` directly in the outputs, set
--expose-encoder True to use the fast path.

1) BIC‑selected full‑covariance GMM prior over z.
2) Low‑rank + diagonal covariance (PPCA prior) either globally or per mixture component.
3) (Scaffold) Energy‑Distance Regularized GMM: penalize component collapse by adding an
   E‑distance spread term; easy to add to the EM M‑step objective.
4) (Scaffold) Normalizing‑flow refinement over the chosen GMM (RealNVP in Flax). Provided
   a template to enable if desired.

Usage
-----
python fit_latent_prior.py \
  --ckpt path/to/vae_seed42.safetensors \
  --cfg  path/to/vae_seed42.yaml \
  --train-data data/train.parquet \
  --outdir outputs/latent_prior \
  --k-max 8 --ppca-rank 0

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from flax.core.frozen_dict import freeze
from flax.serialization import from_state_dict
from flax.traverse_util import flatten_dict, unflatten_dict

# Local deps (must be in PYTHONPATH)
from omegaconf import OmegaConf

from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset
from ah2ac2.tools.prior_math import fit_best_gmm, fit_ppca, log_gaussian

# Import the model definition from the training script
from ah2ac2.training.vae_2p import (
    VAE,
    make_trajectory_batcher,
    to_time_major,
    transform_trajectory_batch,
)

Array = jnp.ndarray

# -----------------------------
# Utilities
# -----------------------------


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p


def to_device_np(x: Array) -> np.ndarray:
    return np.asarray(jax.device_get(x))


def load_flax_safetensors(path: str):
    """Load a Flax params tree from a .safetensors file, handling both '/' and ',' key separators."""
    import re

    from safetensors.numpy import load_file as st_load

    flat = st_load(path)  # dict[str -> np.ndarray]

    # Keys might look like 'Module_0,Sub,Dense_0,kernel' or 'Module_0/Sub/Dense_0/kernel'
    def split_key(k: str):
        parts = [p for p in re.split(r"[\/,]", k) if p != ""]
        return tuple(parts)

    items = {split_key(k): jnp.asarray(v) for k, v in flat.items()}
    unflat = unflatten_dict(items)
    params = unflat.get("params", unflat)
    return freeze(params)


# ---- helpers to debug / align parameter trees ----


def _tree_shapes(d) -> Dict[str, tuple]:
    flat = flatten_dict(d, sep="/")
    return {k: tuple(v.shape) for k, v in flat.items()}


def _try_strip_prefix(loaded: dict, target_keys: set[tuple]) -> Optional[dict]:
    # Try dropping 1 leading segment from loaded keys if that yields exact key match.
    loaded_flat = flatten_dict(loaded)
    for drop in (1, 2):
        stripped = {}
        ok = True
        for k, v in loaded_flat.items():
            if len(k) <= drop:
                ok = False
                break
            nk = k[drop:]
            stripped[nk] = v
        if ok and set(stripped.keys()) == target_keys:
            return unflatten_dict(stripped)
    return None


def _map_by_suffix(loaded: dict, target: dict) -> dict:
    """Map loaded→target by unique suffix match of key tuples. Conservative: raises if ambiguous/missing."""
    lf = flatten_dict(loaded)
    tf = flatten_dict(target)
    # Build index of loaded key strings
    lkeys = {"/".join(k): k for k in lf.keys()}
    mapped = {}
    for tk in tf.keys():
        ts = "/".join(tk)
        candidates = [lk for lk in lkeys if lk.endswith(ts)]
        if len(candidates) == 1:
            mapped[tk] = lf[lkeys[candidates[0]]]
        else:
            # try dropping first segment of target
            tsub = "/".join(tk[1:]) if len(tk) > 1 else ts
            candidates = [lk for lk in lkeys if lk.endswith(tsub)]
            if len(candidates) == 1:
                mapped[tk] = lf[lkeys[candidates[0]]]
            else:
                raise ValueError(
                    f"Cannot uniquely map checkpoint param for target key '{ts}' (candidates={candidates[:5]})"
                )
    return unflatten_dict(mapped)


# -----------------------------
# Encoder capture
# -----------------------------


def extract_mu_lv_via_capture(
    model: VAE, params, obs_tm: Array, act_tm: Array, latent_dim: int
) -> Tuple[Array, Array]:
    """Run model.apply with capture_intermediates to recover μ, logvar tensors.
    Returns (mu_TB_D, lv_TB_D) aggregated to (T,B,D) shaped arrays before averaging.
    """

    def filter_dense(mod, *args, **kwargs):
        # Flax expects a predicate (module, *_) -> bool; accept any signature variants.
        return isinstance(mod, nn.Dense)

    variables = {"params": params}
    outs, mut = model.apply(
        variables,
        obs_seq=obs_tm,
        act_seq=act_tm,
        training=False,
        rngs={"latent": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        capture_intermediates=filter_dense,
        mutable=["intermediates"],  # capture intermediates collection explicitly
    )
    # mut is a dict with key 'intermediates' → {module_path: {'__call__': [outputs...]}}
    dense_outputs: List[Array] = []
    caps = mut.get("intermediates", {})
    for mod_path, calls in caps.items():
        outs_list = calls.get("__call__", [])
        for o in outs_list:
            dense_outputs.append(o)

    # Heuristic: the last two Dense outputs of size latent_dim are mu_raw and lv_raw
    candidates = [o for o in dense_outputs if o.ndim == 2 and o.shape[1] == latent_dim]
    if len(candidates) < 2:
        raise RuntimeError(
            f"Could not find two Dense outputs with latent_dim={latent_dim}. Found {len(candidates)} candidates."
        )
    mu_raw = candidates[-2]
    lv_raw = candidates[-1]

    # We need T and B to reshape. Infer from obs_tm: (T,B,feat)
    T, B, _ = obs_tm.shape
    # mu_raw is (T*B, D)
    mu_tb = mu_raw.reshape(T, B, latent_dim)
    lv_tb = lv_raw.reshape(T, B, latent_dim)
    return mu_tb, lv_tb


def aggregate_mu_over_time(mu_tb: Array, lv_tb: Array) -> Tuple[Array, Array]:
    """Aggregate over time like training code: mean over time to get per‑sequence μ and logvar.
    Returns (mu_B_D, lv_B_D).
    """
    mu_b = mu_tb.mean(axis=0)  # (B,D)
    lv_b = lv_tb.mean(axis=0)  # (B,D)
    return mu_b, lv_b


# -----------------------------
# Gaussian Mixture via EM (full covariance)
# -----------------------------


# -----------------------------
# Low-rank + diagonal covariance (PPCA)
# -----------------------------


# -----------------------------
# Saving / loading
# -----------------------------


def save_prior(outdir: str, name: str, prior: Dict):
    ensure_dir(outdir)
    with open(os.path.join(outdir, f"{name}.json"), "w") as f:
        json.dump(prior, f, indent=2)


# -----------------------------
# Main
# -----------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .safetensors params")
    parser.add_argument("--cfg", required=True, help="Path to YAML config saved with model")
    parser.add_argument(
        "--train-data", required=True, help="Path to training dataset (.safetensors)"
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--k-max", type=int, default=1)
    parser.add_argument("--ppca-rank", type=int, default=0, help="Low-rank rank; 0 disables PPCA")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--expose-encoder", action="store_true", help="If VAE returns mu/lv in outputs['mu','lv']"
    )
    parser.add_argument(
        "--fit-seat-prior",
        action="store_true",
        help="Fit independent priors per seat and save prior_gmm_seat{0,1}.json.",
    )
    parser.add_argument(
        "--fit-paired-prior",
        action="store_true",
        help="Fit joint prior over concatenated (z_seat0, z_seat1), saved as prior_gmm_paired.json.",
    )
    args = parser.parse_args()

    # Guard: obvious flag swap
    if args.ckpt.endswith((".yml", ".yaml")) and args.cfg.endswith(".safetensors"):
        print(
            "Note: --ckpt looks like YAML and --cfg looks like weights; swapping.", file=sys.stderr
        )
        args.ckpt, args.cfg = args.cfg, args.ckpt

    if not args.ckpt.endswith(".safetensors"):
        raise ValueError(f"--ckpt must be .safetensors, got {args.ckpt}")
    if not args.cfg.endswith((".yml", ".yaml")):
        raise ValueError(f"--cfg must be .yaml/.yml, got {args.cfg}")

    ensure_dir(args.outdir)

    # Load config
    cfg = ml_collections.ConfigDict(OmegaConf.to_container(OmegaConf.load(args.cfg), resolve=True))

    # Dataset & loader (as training)
    train_ds = HanabiLiveGamesDataset(file=args.train_data)
    rng = jax.random.PRNGKey(args.seed)
    rng, loader_rng = jax.random.split(rng)
    train_dl = HanabiLiveGamesDataloader(train_ds, args.batch_size, loader_rng)

    # Trajectory factory
    get_traj_vjit = jax.jit(jax.vmap(make_trajectory_batcher(train_ds.num_players), in_axes=0))

    # Build model
    model = VAE(
        preprocessing_features=tuple(cfg.preprocessing_features),
        lstm_features=tuple(cfg.lstm_features),
        postprocessing_features=tuple(cfg.postprocessing_features),
        action_dim=cfg.action_dim,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        act_fn=cfg.act_fn,
        beta=cfg.beta,
    )

    # Prepare a small batch for a sanity forward (shape check)
    batch0 = next(iter(train_dl))
    rng, key = jax.random.split(rng)
    keys0 = jax.random.split(key, batch0.actions.shape[0])
    traj0 = get_traj_vjit(keys0, batch0.decks, batch0.actions)
    obs_ba0, act_ba0, _, _ = transform_trajectory_batch(traj0, batch0.game_len_masks)
    obs_tm0, act_tm0 = to_time_major(obs_ba0, act_ba0)

    # Initialize a params skeleton to establish the exact tree/shape layout
    params0 = model.init(
        jax.random.PRNGKey(123), obs_seq=obs_tm0, act_seq=act_tm0, training=False
    )["params"]

    # Load params from .safetensors and map them into the skeleton tree
    loaded = load_flax_safetensors(args.ckpt)
    try:
        params = from_state_dict(params0, loaded)
    except Exception:
        # Attempt simple prefix stripping
        stripped = _try_strip_prefix(loaded, set(flatten_dict(params0).keys()))
        if stripped is not None:
            params = from_state_dict(params0, stripped)
        else:
            # Fall back to suffix-based key matching
            aligned = _map_by_suffix(loaded, params0)
            params = from_state_dict(params0, aligned)

    # Sanity forward to ensure checkpoint matches model and inputs
    _ = model.apply(
        {"params": params},
        obs_seq=obs_tm0,
        act_seq=act_tm0,
        training=False,
        rngs={"latent": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
    )

    # Encode all data
    all_mu = []
    all_lv = []
    for batch in train_dl:
        rng, key = jax.random.split(rng)
        keys = jax.random.split(key, batch.actions.shape[0])
        traj = get_traj_vjit(keys, batch.decks, batch.actions)
        obs_ba, act_ba, _, _ = transform_trajectory_batch(traj, batch.game_len_masks)
        obs_tm, act_tm = to_time_major(obs_ba, act_ba)

        if args.expose_encoder:
            # Fast path: assume model returns mu/lv in aux dict
            def forward(params, obs_tm, act_tm):
                _, outs = model.apply(
                    {"params": params},
                    obs_seq=obs_tm,
                    act_seq=act_tm,
                    training=False,
                    rngs={"latent": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
                )
                return outs["mu_tb"], outs["lv_tb"]  # Expected keys if you added them

            mu_tb, lv_tb = forward(params, obs_tm, act_tm)
        else:
            mu_tb, lv_tb = extract_mu_lv_via_capture(model, params, obs_tm, act_tm, cfg.latent_dim)
        mu_b, lv_b = aggregate_mu_over_time(mu_tb, lv_tb)  # (B*,D)
        all_mu.append(to_device_np(mu_b))
        all_lv.append(to_device_np(lv_b))

    Z = np.concatenate(all_mu, axis=0)  # (N,D) latent means as our point estimates
    LV = np.concatenate(all_lv, axis=0)

    np.savez(os.path.join(args.outdir, "latent_stats.npz"), Z=Z, LV=LV)
    # Also save CSV for quick visualization
    import csv

    with open(os.path.join(args.outdir, "Z.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"z{i}" for i in range(Z.shape[1])])
        w.writerows(Z.tolist())

    # Fit priors
    A = train_ds.num_players
    L = Z.shape[1]
    if Z.shape[0] % A != 0:
        raise RuntimeError(
            f"N={Z.shape[0]} not divisible by num_players={A}; cannot reshape to (N_games, A, L)."
        )

    Z_ba = Z.reshape(-1, A, L)  # (N_games, A, L)

    # Save per-seat arrays for convenience
    seat_arrays = {f"Z{a}": Z_ba[:, a, :] for a in range(A)}
    np.savez(os.path.join(args.outdir, "latent_stats_seats.npz"), **seat_arrays)

    if args.fit_seat_prior:
        for a in range(A):
            Za = Z_ba[:, a, :]  # (N_games, L)
            print(f"\n[Seat {a}] Fitting GMM prior over {Za.shape[0]} latents (dim={L})")
            Ka, gmma, bica = fit_best_gmm(Za, args.k_max, args.seed + a)
            prior_a = {
                "type": "gmm-fullcov",
                "K": int(Ka),
                "weights": gmma.weights.tolist(),
                "means": gmma.means.tolist(),
                "cov_cholesky": [np.linalg.cholesky(C).tolist() for C in gmma.covs],
            }
            save_prior(args.outdir, f"prior_gmm_seat{a}", prior_a)

    if args.fit_paired_prior:
        # Concatenate seats to form (N_games, A*L); for 2p: (N_games, 2L)
        Zpair = np.concatenate([Z_ba[:, a, :] for a in range(A)], axis=1)
        print(
            f"\n[Paired] Fitting GMM prior over paired latents: {Zpair.shape[0]} samples, dim={Zpair.shape[1]}"
        )
        Kp, gmmp, bicp = fit_best_gmm(Zpair, args.k_max, args.seed + 123)
        prior_pair = {
            "type": "gmm-fullcov",
            "K": int(Kp),
            "weights": gmmp.weights.tolist(),
            "means": gmmp.means.tolist(),  # each mean is length A*L
            "cov_cholesky": [np.linalg.cholesky(C).tolist() for C in gmmp.covs],
            "players": int(A),
            "latent_dim": int(L),
            "paired": True,
        }
        save_prior(args.outdir, "prior_gmm_paired", prior_pair)

        # Optional: also drop a .npz so you can quick-sample offline without loading JSON
        np.savez(os.path.join(args.outdir, "Zpair.npz"), Zpair=Zpair)

        # Also save a spherical Gaussian prior N(mean, I) for direct comparison
    # Matching the gmm-fullcov format so eval_latent_prior can load it seamlessly
    mean = Z.mean(axis=0)
    D = mean.shape[0]
    sph_chol = np.eye(D).tolist()
    sph_json = {
        "type": "gmm-fullcov",
        "K": 1,
        "weights": [1.0],
        "means": [mean.tolist()],
        "cov_cholesky": [sph_chol],
    }
    save_prior(args.outdir, "prior_spherical", sph_json)

    # Optional PPCA structure
    if args.ppca_rank > 0:
        K_star, gmm_star, _ = fit_best_gmm(Z, args.k_max, args.seed)
        if K_star == 1:
            U, Lam, sigma2 = fit_ppca(Z, args.ppca_rank)
            prior_ppca = {
                "type": "ppca",
                "rank": int(args.ppca_rank),
                "mean": Z.mean(axis=0).tolist(),
                "U": U.tolist(),
                "Lambda": Lam.tolist(),
                "sigma2": float(sigma2),
            }
            save_prior(args.outdir, "prior_ppca", prior_ppca)
        else:
            # Per-component PPCA
            comps = []
            # Compute responsibilities under best model
            N, D = Z.shape
            log_resp = []
            for k in range(K_star):
                log_resp.append(
                    np.log(gmm_star.weights[k] + 1e-12)
                    + log_gaussian(Z, gmm_star.means[k], gmm_star.covs[k])
                )
            log_resp = np.stack(log_resp, axis=1)
            m = np.max(log_resp, axis=1, keepdims=True)
            resp = np.exp(log_resp - m)
            resp /= resp.sum(axis=1, keepdims=True)

            for k in range(K_star):
                rk = resp[:, k]
                Xk = Z[rk > 1e-4]
                if len(Xk) < args.ppca_rank + 1:
                    # fallback: use global
                    U = np.zeros((D, 0))
                    Lam = np.zeros((0,))
                    sigma2 = float(np.trace(gmm_star.covs[k]) / D)
                else:
                    U, Lam, sigma2 = fit_ppca(Xk, args.ppca_rank)
                comps.append(
                    {
                        "mean": gmm_star.means[k].tolist(),
                        "weight": float(gmm_star.weights[k]),
                        "U": U.tolist(),
                        "Lambda": Lam.tolist(),
                        "sigma2": float(sigma2),
                    }
                )
            save_prior(args.outdir, "prior_gmm_ppca", {"type": "gmm-ppca", "components": comps})

    print("Saved priors and latent arrays to:", args.outdir)


if __name__ == "__main__":
    main()
