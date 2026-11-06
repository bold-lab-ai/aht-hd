# ah2ac2/tools/latent_diagnostics.py

import logging
import os
from types import SimpleNamespace
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset

# Reuse actor core from crossplay_eval
from ah2ac2.tools.crossplay_eval import (
    ActorLstm,
    load_actor_vars_for_vae,
)
from ah2ac2.training.vae_2p import (
    VAE,
    make_trajectory_batcher,
    to_time_major,
    transform_trajectory_batch,
)

# -------------------------------------------------
# VAE loading (same style as latent_search)
# -------------------------------------------------


def load_vae_model_and_params(rng, cfg) -> Tuple[VAE, dict]:
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

    T, B, D = cfg.dummy_TBD
    dummy_o = jnp.ones((T, B, D), jnp.float32)
    dummy_a = jnp.zeros((T, B, 1), jnp.int32)
    params = model.init(rng, obs_seq=dummy_o, act_seq=dummy_a, training=False)["params"]

    from flax.core import freeze
    from flax.traverse_util import unflatten_dict
    from safetensors.numpy import load_file as st_load

    flat = st_load(cfg.vae_ckpt)
    flat = {k: jnp.array(v) for k, v in flat.items()}

    def split_key(k: str):
        if "/" in k:
            return tuple(k.split("/"))
        if "," in k:
            return tuple(k.split(","))
        return (k,)

    pairs = {split_key(k): v for k, v in flat.items()}
    tree = unflatten_dict(pairs)

    if "params" in tree:
        params = freeze(tree["params"])
    else:
        params = freeze(tree)

    return model, params


# -------------------------------------------------
# Build a pool of realistic observation states
# -------------------------------------------------


def sample_obs_states_from_dataset(
    ds_path: str,
    num_players: int,
    batch_size: int,
    rng: jax.Array,
    num_states: int,
) -> jnp.ndarray:
    """
    Returns:
        obs_states: (S, D) JAX array
    """
    ds = HanabiLiveGamesDataset(file=ds_path)
    rng, loader_rng = jax.random.split(rng)
    dl = HanabiLiveGamesDataloader(ds, batch_size, loader_rng)

    # One batch is plenty for diagnostics
    batch = next(iter(dl))

    get_traj_batch = jax.jit(jax.vmap(make_trajectory_batcher(num_players), in_axes=0))
    dummy_keys = jax.random.split(rng, batch.actions.shape[0])
    traj_b = get_traj_batch(dummy_keys, batch.decks, batch.actions)

    obs_ba, act_ba, _, _ = transform_trajectory_batch(traj_b, batch.game_len_masks)
    # obs_tm: (T, B*, D)
    obs_tm, _ = to_time_major(obs_ba, act_ba)
    obs_np = np.array(obs_tm, dtype=np.float32)

    T, B_flat, D = obs_np.shape
    total = T * B_flat
    num_states = min(num_states, total)

    # Sample random (t, b) pairs
    idxs = np.random.choice(total, size=num_states, replace=False)
    obs_states = obs_np.reshape(total, D)[idxs]  # (S, D)
    return jnp.array(obs_states)


# -------------------------------------------------
# Forward pass: (obs, z) -> action probabilities
# -------------------------------------------------


def build_actor(
    vae_ckpt: str,
    action_dim: int,
    preprocessing_features,
    lstm_features,
    postprocessing_features,
    dropout: float,
    act_fn: str,
):
    actor_mod = ActorLstm(
        preprocessing_features=preprocessing_features,
        lstm_features=lstm_features,
        postprocessing_features=postprocessing_features,
        action_dim=action_dim,
        dropout_rate=dropout,
        activation_fn_name=act_fn,
    )
    actor_vars = load_actor_vars_for_vae(vae_ckpt)
    return actor_mod, actor_vars


def logits_for_state_and_z(
    actor_mod,
    actor_vars,
    obs: jnp.ndarray,  # (D,)
    z: jnp.ndarray,  # (L,)
) -> jnp.ndarray:
    """
    Single-step forward: treats `obs` as a one-timestep input for a single agent.
    Returns logits: (A,)
    """
    # Shapes: (T=1, B=1, D)
    obs_tb = obs[None, None, :]
    resets_tb = jnp.zeros((1, 1), dtype=bool)
    z_b = z[None, :]  # (B=1, L)

    carry0 = actor_mod.initialize_carry(batch_size=1)
    carry1, logits_tb, _ = actor_mod.apply(
        actor_vars,
        carry0,
        (obs_tb, resets_tb),
        z=z_b,
        training=False,
    )
    logits = logits_tb[0, 0, :]  # (A,)
    return logits


def probs_for_state_and_z(
    actor_mod,
    actor_vars,
    obs: jnp.ndarray,
    z: jnp.ndarray,
) -> jnp.ndarray:
    logits = logits_for_state_and_z(actor_mod, actor_vars, obs, z)
    return jax.nn.softmax(logits, axis=-1)


# -------------------------------------------------
# Diagnostic 1: Action-distribution sensitivity to z
# -------------------------------------------------


def diagnostic_action_sensitivity(
    logger,
    actor_mod,
    actor_vars,
    obs_states: jnp.ndarray,  # (S, D)
    latents_pool: jnp.ndarray,  # (K, L)
    num_states_diag: int,
    num_z_diag: int,
    seed: int,
):
    """
    For a subset of states and z's, compute pairwise KL between action distributions.
    Logs statistics of KL per-state and overall.
    """
    rng = np.random.RandomState(seed)

    S_total = obs_states.shape[0]
    K_total = latents_pool.shape[0]

    S = min(num_states_diag, S_total)
    K = min(num_z_diag, K_total)

    state_idxs = rng.choice(S_total, size=S, replace=False)
    z_idxs = rng.choice(K_total, size=K, replace=False)

    obs_sub = np.array(obs_states)[state_idxs]  # (S, D) numpy
    z_sub = np.array(latents_pool)[z_idxs]  # (K, L) numpy

    logger.info(f"[ActionSensitivity] Using S={S} states and K={K} latents for KL diagnostics.")

    per_state_kl_means = []
    per_state_kl_maxes = []

    for s in range(S):
        obs_s = jnp.array(obs_sub[s])
        # probs[k, a]
        probs_k = []
        for k in range(K):
            z_k = jnp.array(z_sub[k])
            p = np.array(probs_for_state_and_z(actor_mod, actor_vars, obs_s, z_k))
            probs_k.append(p)
        probs_k = np.stack(probs_k, axis=0)  # (K, A)

        # pairwise KL(p_i || p_j)
        # use numpy loops; K is small
        kl_vals = []
        for i in range(K):
            p_i = probs_k[i]
            log_p_i = np.log(p_i + 1e-8)
            for j in range(K):
                if i == j:
                    continue
                p_j = probs_k[j]
                log_p_j = np.log(p_j + 1e-8)
                kl_ij = np.sum(p_i * (log_p_i - log_p_j))
                kl_vals.append(kl_ij)
        kl_vals = np.array(kl_vals, dtype=np.float32)
        per_state_kl_means.append(kl_vals.mean())
        per_state_kl_maxes.append(kl_vals.max())

    per_state_kl_means = np.array(per_state_kl_means)
    per_state_kl_maxes = np.array(per_state_kl_maxes)

    logger.info(
        "[ActionSensitivity] Per-state mean KL: "
        f"mean={per_state_kl_means.mean():.4f}, "
        f"min={per_state_kl_means.min():.4f}, "
        f"max={per_state_kl_means.max():.4f}"
    )
    logger.info(
        "[ActionSensitivity] Per-state max KL: "
        f"mean={per_state_kl_maxes.mean():.4f}, "
        f"min={per_state_kl_maxes.min():.4f}, "
        f"max={per_state_kl_maxes.max():.4f}"
    )


# -------------------------------------------------
# Diagnostic 2: Gradient norm wrt z
# -------------------------------------------------


def diagnostic_grad_norm_z(
    logger,
    actor_mod,
    actor_vars,
    obs_states: jnp.ndarray,  # (S, D)
    latents_pool: jnp.ndarray,  # (K, L)
    num_states_diag: int,
    num_z_diag: int,
    seed: int,
):
    """
    For a subset of states and z's, compute ||∂ log π(a*|o, z) / ∂z||_2,
    where a* is the greedy action under π(·|o,z).
    """
    rng = np.random.RandomState(seed + 12345)

    S_total = obs_states.shape[0]
    K_total = latents_pool.shape[0]

    S = min(num_states_diag, S_total)
    K = min(num_z_diag, K_total)

    state_idxs = rng.choice(S_total, size=S, replace=False)
    z_idxs = rng.choice(K_total, size=K, replace=False)

    obs_sub = jnp.array(np.array(obs_states)[state_idxs])  # (S, D)
    z_sub = jnp.array(np.array(latents_pool)[z_idxs])  # (K, L)

    logger.info(f"[GradNormZ] Using S={S} states and K={K} latents for grad diagnostics.")

    def logprob_greedy_for_z(z, obs):
        # obs: (D,), z: (L,)
        logits = logits_for_state_and_z(actor_mod, actor_vars, obs, z)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        a_star = jnp.argmax(log_probs)
        return log_probs[a_star]

    # grad wrt first argument (z)
    grad_fn = jax.grad(logprob_greedy_for_z)

    norms = []
    for s in range(S):
        obs_s = obs_sub[s]
        for k in range(K):
            z_k = z_sub[k]
            g = grad_fn(z_k, obs_s)  # (L,)
            n = jnp.linalg.norm(g)
            norms.append(float(n))

    norms = np.array(norms, dtype=np.float32)
    logger.info(
        f"[GradNormZ] ||∂ log π / ∂z||₂ over {len(norms)} samples: "
        f"mean={norms.mean():.6f}, "
        f"std={norms.std():.6f}, "
        f"min={norms.min():.6f}, "
        f"max={norms.max():.6f}"
    )


# -------------------------------------------------
# CLI
# -------------------------------------------------


def main():
    import argparse

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Diagnostics for VAE latent usage in the decoder policy."
    )

    # Data / model
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument(
        "--action_dim",
        type=int,
        required=True,
        help="Action dim used for the VAE decoder policy (same as during training).",
    )

    parser.add_argument("--preprocessing_features", type=int, nargs="+", default=[1024])
    parser.add_argument("--lstm_features", type=int, nargs="+", default=[512])
    parser.add_argument("--postprocessing_features", type=int, nargs="+", default=[256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--act_fn", type=str, default="gelu")

    # Latent pool options
    parser.add_argument(
        "--selected_z_path",
        type=str,
        default=None,
        help="Path to .npy with selected latents (M,L). If not given, we sample random z.",
    )
    parser.add_argument(
        "--candidate_z_path",
        type=str,
        default=None,
        help="Optional fallback .npy with candidate latents (N,L).",
    )

    # Diagnostics hyperparams
    parser.add_argument(
        "--num_states_diag",
        type=int,
        default=16,
        help="Number of distinct states for diagnostics.",
    )
    parser.add_argument(
        "--num_z_diag",
        type=int,
        default=32,
        help="Number of distinct z values to use in diagnostics.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, required=False, default=None)

    args = parser.parse_args()

    rng = jax.random.PRNGKey(args.seed)

    # 1) Build dummy config to init VAE (only to infer shapes; we actually only need actor params)
    #    We still reuse the dataset transform to get obs states, so we need the vae_2p shapes.
    logger.info("Sampling observation states from dataset...")
    obs_states = sample_obs_states_from_dataset(
        ds_path=args.train_data_path,
        num_players=args.num_players,
        batch_size=args.batch_size,
        rng=rng,
        num_states=args.num_states_diag * 4,  # sample a bit more than we need
    )
    S_pool, D = obs_states.shape
    logger.info(f"Collected {S_pool} observation states of dim D={D}.")

    # Build config shim for VAE init (shapes only)
    # We'll pretend T=1, B=1 here because we don't need actual encoder forward.
    cfg = SimpleNamespace(
        preprocessing_features=tuple(args.preprocessing_features),
        lstm_features=tuple(args.lstm_features),
        postprocessing_features=tuple(args.postprocessing_features),
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        act_fn=args.act_fn,
        beta=1.0,
        dummy_TBD=(1, 1, D),
        vae_ckpt=args.vae_ckpt,
    )

    rng, vae_rng = jax.random.split(rng)
    # We don't actually use vae_model here, but this validates that the ckpt / config are sane
    _, _ = load_vae_model_and_params(vae_rng, cfg)
    logger.info("Loaded VAE params successfully.")

    # 2) Build actor decoder
    logger.info("Building actor decoder (LSTM) from VAE checkpoint...")
    actor_mod, actor_vars = build_actor(
        vae_ckpt=args.vae_ckpt,
        action_dim=args.action_dim,
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        dropout=args.dropout,
        act_fn=args.act_fn,
    )

    # 3) Build latent pool
    if args.selected_z_path and os.path.exists(args.selected_z_path):
        latents_pool = jnp.array(np.load(args.selected_z_path)).astype(jnp.float32)
        logger.info(
            f"Loaded latents from selected_z_path={args.selected_z_path} "
            f"with shape {latents_pool.shape}."
        )
    elif args.candidate_z_path and os.path.exists(args.candidate_z_path):
        latents_pool = jnp.array(np.load(args.candidate_z_path)).astype(jnp.float32)
        logger.info(
            f"Loaded latents from candidate_z_path={args.candidate_z_path} "
            f"with shape {latents_pool.shape}."
        )
    else:
        # Fall back: random Gaussian latents
        rng, z_rng = jax.random.split(rng)
        latents_pool = jax.random.normal(
            z_rng, shape=(args.num_z_diag * 4, args.latent_dim), dtype=jnp.float32
        )
        logger.info(
            f"No latent .npy provided; sampled random Gaussian latents with "
            f"shape {latents_pool.shape}."
        )

    # 4) Run diagnostics
    diagnostic_action_sensitivity(
        logger=logger,
        actor_mod=actor_mod,
        actor_vars=actor_vars,
        obs_states=obs_states,
        latents_pool=latents_pool,
        num_states_diag=args.num_states_diag,
        num_z_diag=args.num_z_diag,
        seed=args.seed,
    )

    diagnostic_grad_norm_z(
        logger=logger,
        actor_mod=actor_mod,
        actor_vars=actor_vars,
        obs_states=obs_states,
        latents_pool=latents_pool,
        num_states_diag=args.num_states_diag,
        num_z_diag=args.num_z_diag,
        seed=args.seed,
    )

    logger.info("Diagnostics complete.")


if __name__ == "__main__":
    main()
