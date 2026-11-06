# ah2ac2/tools/latent_es_refine.py
"""

Evolutionary refinement of a *set* of VAE latents Z ∈ R^{M×L}
for Hanabi. Objective: high self-play (SP) and low cross-play (XP)
between the induced policies.

Usage (starting from selected_z.npy):

python -m ah2ac2.tools.latent_es_refine \
  --num_players 2 \
  --vae_ckpt checkpoints/vae/human_vae/latent_dim=16/5k/epoch52_seed0_valacc0.597.safetensors \
  --init_z_path checkpoints/vae/human_vae/latent_dim=16/5k/latent_search/selected_z.npy \
  --xp_weight 0.5 \
  --num_latents 8 \
  --num_iterations 200 \
  --population_size 128 \
  --num_rollouts_sp 64 \
  --num_rollouts_xp 64 \
  --num_pairs_xp 8 \
  --sigma 0.1 \
  --lr 0.05 \
  --seed 0 \
  --out_dir checkpoints/vae/human_vae/latent_dim=16/5k/es_refine/


python -m ah2ac2.tools.latent_es_refine \
  --num_players 2 \
  --vae_ckpt checkpoints/vae/human_vae/latent_dim=32/100k/vae_on_all_human.safetensors \
  --init_z_path checkpoints/vae/human_vae/latent_dim=32/100k/latent_search/selected_z.npy \
  --xp_weight 0.5 \
  --num_latents 8 \
  --num_iterations 50 \
  --population_size 128 \
  --num_rollouts_sp 64 \
  --num_rollouts_xp 64 \
  --num_pairs_xp 8 \
  --sigma 0.1 \
  --lr 0.05 \
  --seed 0 \
  --out_dir checkpoints/vae/human_vae/latent_dim=32/100k/es_refine/

python -m ah2ac2.tools.latent_es_refine \
  --num_players 2 \
  --vae_ckpt checkpoints/vae_lowdim_decoder/human_vae/latent_dim=32/3k/epoch60_seed0_valacc0.638.safetensors \
  --init_z_path checkpoints/vae_lowdim_decoder/human_vae/latent_dim=32/3k/latent_search/selected_z.npy \
  --xp_weight 0.75 \
  --num_latents 3 \
  --num_iterations 2_000 \
  --population_size 256 \
  --num_rollouts_sp 64 \
  --num_rollouts_xp 64 \
  --num_pairs_xp 8 \
  --sigma 0.1 \
  --lr 0.05 \
  --seed 0 \
  --out_dir checkpoints/vae/human_vae/latent_dim=32/3k/es_refine/ \
  --lstm_features 256 \
  --postprocessing_features 128

python -m ah2ac2.tools.latent_es_refine \
  --num_players 2 \
  --vae_ckpt checkpoints/vae_lowdim_masked_mixture/human_plus_obl_vae/latent_dim32/human_frac=0.05/epoch52_seed0_valacc0.635.safetensors \
  --init_z_path checkpoints/vae_lowdim_masked_mixture/human_plus_obl_vae/latent_dim32/human_frac=0.05/latent_search/selected_z.npy \
  --xp_weight 1.25 \
  --num_latents 8 \
  --num_iterations 2_000 \
  --population_size 256 \
  --num_rollouts_sp 32 \
  --num_rollouts_xp 32 \
  --num_pairs_xp 8 \
  --sigma 0.1 \
  --lr 0.05 \
  --seed 0 \
  --out_dir checkpoints/vae_lowdim_masked_mixture/human_plus_obl_vae/latent_dim32/human_frac=0.05/es_refine/ \
  --lstm_features 256 \
  --postprocessing_features 128

python -m ah2ac2.tools.latent_es_refine \
  --num_players 2 \
  --vae_ckpt checkpoints/vae_masked/human_vae/latent_dim32/100k/epoch98_seed0_valacc0.813.safetensors \
  --init_z_path checkpoints/vae_masked/human_vae/latent_dim32/100k/latent_search/selected_z.npy \
  --xp_weight 0.75 \
  --num_latents 8 \
  --num_iterations 1_000 \
  --population_size 256 \
  --num_rollouts_sp 32 \
  --num_rollouts_xp 32 \
  --num_pairs_xp 8 \
  --sigma 0.1 \
  --lr 0.05 \
  --seed 0 \
  --out_dir checkpoints/vae_masked/human_vae/latent_dim32/100k/es_refine/ \
  --lstm_features 512 \
  --postprocessing_features 256

python -m ah2ac2.tools.latent_es_refine \
  --num_players 2 \
  --vae_ckpt checkpoints/vae_lowdim_masked/human_vae/latent_dim32/100k/epoch99_seed0_valacc0.803.safetensors \
  --init_z_path checkpoints/vae_lowdim_masked/human_vae/latent_dim32/100k/latent_search/selected_z.npy \
  --xp_weight 0.75 \
  --num_latents 8 \
  --num_iterations 1_000 \
  --population_size 256 \
  --num_rollouts_sp 32 \
  --num_rollouts_xp 32 \
  --num_pairs_xp 3 \
  --sigma 0.05 \
  --lr 0.05 \
  --seed 5 \
  --out_dir checkpoints/vae_lowdim_masked/human_vae/latent_dim32/100k/es_refine/ \
  --lstm_features 256 \
  --postprocessing_features 128


Usage (random init on VAE prior manifold):

python -m ah2ac2.tools.latent_es_refine \
  --num_players 2 \
  --vae_ckpt checkpoints/vae_masked_3/human_vae/latent_dim32/100k/epoch99_seed2_valacc0.832.safetensors \
  --prior_json outputs/latent_prior_human_dataset_100k/prior_spherical.json \
  --xp_weight 0.75 \
  --num_latents 3 \
  --num_iterations 3_000 \
  --population_size 256 \
  --num_rollouts_sp 32 \
  --num_rollouts_xp 32 \
  --num_pairs_xp 3 \
  --sigma 0.05 \
  --lr 0.05 \
  --seed 5 \
  --out_dir outputs/es_refine_test_3/ \
  --lstm_features 512 \
  --postprocessing_features 256


python -m ah2ac2.tools.latent_es_refine \
  --num_players 2 \
  --vae_ckpt checkpoints/vae_masked/human_vae/latent_dim32/100k/epoch98_seed0_valacc0.813.safetensors \
  --prior_json outputs/latent_prior_human_dataset_100k/prior_spherical.json \
  --xp_weight 0 \
  --num_latents 1 \
  --num_iterations 3_000 \
  --population_size 256 \
  --num_rollouts_sp 32 \
  --num_rollouts_xp 32 \
  --num_pairs_xp 3 \
  --sigma 0.05 \
  --lr 0.01 \
  --seed 5 \
  --out_dir outputs/latent_search_no_xp/ \
  --lstm_features 512 \
  --postprocessing_features 256


w/ xp ramping:

python -m ah2ac2.tools.latent_es_refine \
  --num_players 2 \
  --vae_ckpt checkpoints/epoch96_seed0_valacc0.826.safetensors \
  --prior_json outputs/latent_prior_human_dataset_100k/prior_spherical.json \
  --xp_weight 0.75 \
  --ramp_xp_weight \
  --xp_weight_start 0.4 \
  --xp_warmup_iters 50 \
  --xp_ramp_iters 250 \
  --sp_gate 8.0 \
  --num_latents 8 \
  --num_iterations 1_000 \
  --population_size 256 \
  --num_rollouts_sp 32 \
  --num_rollouts_xp 32 \
  --num_pairs_xp 8 \
  --sigma 0.1 \
  --lr 0.1 \
  --seed 0 \
  --out_dir checkpoints/es_refine_latentdim32 \
  --lstm_features 512 \
  --postprocessing_features 256
"""

import json
import logging
import os
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl import make

# Reuse Hanabi wrapper + actor core + prior helpers from crossplay_eval
from ah2ac2.tools.crossplay_eval import (
    ActorLstm,
    LogWrapper,
    load_actor_vars_for_vae,
    sample_z_from_prior,
    unwrap_state,
)

# -------------------------
# Env + single-game harness
# -------------------------


def build_env(num_players: int):
    hand_size = 5 if num_players <= 3 else 4
    base_env = make("hanabi", hand_size=hand_size, num_agents=num_players)
    env = LogWrapper(base_env)
    return env


def infer_action_dim(env):
    rng = jax.random.PRNGKey(0)
    obs0, env_state0 = env.reset(rng)
    legal0 = env.get_legal_moves(unwrap_state(env_state0))
    first_agent = list(env.agents)[0]
    return int(legal0[first_agent].shape[-1])


def make_vae_single_game_fn(env, actor_mod, actor_vars, num_players: int):
    """
    Returns a jitted function single_game(rng, z1, z2) -> scalar return,
    where z1 controls player 0 and z2 controls player 1 (via curr_player
    switching logic exactly as in crossplay_eval).
    """
    agents = list(env.agents)
    B = len(agents)
    assert B == num_players, "num_players must match env.agents"

    @jax.jit
    def _step_env(rng, env_state, actions):
        rng, _rng = jax.random.split(rng)
        obs, new_env_state, reward, dones, infos = env.step(_rng, env_state, actions)
        legal_moves = env.get_legal_moves(unwrap_state(new_env_state))
        return rng, new_env_state, obs, legal_moves, reward, dones

    def single_game(rng, z1, z2):
        # Reset env
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)
        legal_moves = env.get_legal_moves(unwrap_state(env_state))

        # LSTM state for two policies
        carry1 = actor_mod.initialize_carry(B)
        carry2 = actor_mod.initialize_carry(B)

        done = False
        cum_rew = 0.0

        def cond_fn(state):
            cum_rew, done, *_ = state
            return jnp.logical_not(done)

        def body_fn(state):
            (
                cum_rew,
                done,
                rng,
                env_state,
                obs,
                legal_moves,
                carry1,
                carry2,
            ) = state

            rng, step_rng = jax.random.split(rng)
            curr_player = jnp.argmax(unwrap_state(env_state).cur_player_idx)

            # Build (T=1,B,*) batches
            obs_batch = jnp.stack([obs[a] for a in agents], axis=0)[None, ...]  # (1,B,D)
            resets = jnp.full((1, B), done, dtype=bool)
            avail = jnp.stack([legal_moves[a] for a in agents], axis=0)[None, ...]  # (1,B,A)

            # Latent per policy
            z1_b = jnp.repeat(z1[None, :], B, axis=0)  # (B,L)
            z2_b = jnp.repeat(z2[None, :], B, axis=0)  # (B,L)

            # Forward actor for both policies
            carry1, logits1, _ = actor_mod.apply(
                actor_vars,
                carry1,
                (obs_batch, resets),
                z=z1_b,
                training=False,
            )
            carry2, logits2, _ = actor_mod.apply(
                actor_vars,
                carry2,
                (obs_batch, resets),
                z=z2_b,
                training=False,
            )

            # Mask illegal actions and pick greedy
            masked1 = logits1 - (1.0 - avail) * 1e10
            masked2 = logits2 - (1.0 - avail) * 1e10

            actions1 = jnp.argmax(masked1, axis=-1).squeeze(0)  # (B,)
            actions2 = jnp.argmax(masked2, axis=-1).squeeze(0)  # (B,)

            def build_actions(vec):
                return {agent_name: vec[i] for i, agent_name in enumerate(agents)}

            # Policy index = curr_player
            actions = jax.lax.cond(
                curr_player == 1,
                lambda _: build_actions(actions2),
                lambda _: build_actions(actions1),
                operand=None,
            )

            rng_next, env_state, obs, legal_moves, reward, dones = _step_env(
                step_rng, env_state, actions
            )

            done_next = dones["__all__"]
            cum_rew_next = cum_rew + reward["__all__"]
            return (
                cum_rew_next,
                done_next,
                rng_next,
                env_state,
                obs,
                legal_moves,
                carry1,
                carry2,
            )

        init_state = (
            cum_rew,
            done,
            rng,
            env_state,
            obs,
            legal_moves,
            carry1,
            carry2,
        )
        cum_rew_final, *_ = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return cum_rew_final

    return jax.jit(single_game)


# -------------------------
# SP / XP estimators
# -------------------------


def per_latent_xp_avg(xp_mat: np.ndarray) -> np.ndarray:
    """
    xp_mat: (M,M) symmetric XP matrix (including diagonal).
    Returns xp_avg: (M,) where xp_avg[i] = mean_j!=i xp_mat[i,j].
    If M==1, returns zeros.
    """
    M = xp_mat.shape[0]
    if M <= 1:
        return np.zeros((M,), dtype=np.float32)
    mask = ~np.eye(M, dtype=bool)
    # row-wise mean excluding diagonal
    xp_avg = (xp_mat * mask).sum(axis=1) / mask.sum(axis=1)
    return xp_avg.astype(np.float32)


def save_latents_with_scores(
    out_dir: str,
    prefix: str,
    Z: np.ndarray,
    sp_vec: np.ndarray,
    xp_avg: np.ndarray,
    decimals: int = 3,
):
    """
    Save each latent z_i with SP and avg-XP in filename for easy filtering.

    Example filename:
      {prefix}_i000_SP=9.125_XPavg=4.238.npy
    """
    os.makedirs(out_dir, exist_ok=True)
    M = Z.shape[0]
    assert sp_vec.shape[0] == M
    assert xp_avg.shape[0] == M

    fmt = f"{{:.{decimals}f}}"
    for i in range(M):
        sp_s = fmt.format(float(sp_vec[i]))
        xp_s = fmt.format(float(xp_avg[i]))
        fname = f"{prefix}_i{i:03d}_SP={sp_s}_XPavg={xp_s}.npy"
        np.save(os.path.join(out_dir, fname), Z[i])


def build_sp_estimator(single_game, num_latents: int, num_rollouts_sp: int):
    """
    Returns a function sp_scores(rng, Z) -> (M,) of SP scores.
    Z: (M, L)
    """

    def sp_scores(rng, Z):
        M = Z.shape[0]
        assert M == num_latents
        rng_latents = jax.random.split(rng, M)

        def per_latent(r, z):
            rollout_keys = jax.random.split(r, num_rollouts_sp)
            returns = jax.vmap(lambda key: single_game(key, z, z))(rollout_keys)
            return jnp.mean(returns)

        return jax.vmap(per_latent, in_axes=(0, 0))(rng_latents, Z)  # (M,)

    return sp_scores


def build_xp_estimator(
    single_game, all_pairs: jnp.ndarray, num_pairs_xp: int, num_rollouts_xp: int
):
    """
    Returns a function xp_stats(rng, Z) -> (xp_median, xp_mean)
    over a random subset of pairs from all_pairs.

    If num_pairs_xp == 0 or |all_pairs| == 0, both stats are 0.
    all_pairs: (P_total, 2) jnp array of (i,j), i < j.
    """
    P_total = all_pairs.shape[0]

    if P_total == 0 or num_pairs_xp == 0 or num_rollouts_xp == 0:

        def xp_stats(rng, Z):
            _ = (rng, Z)
            zero = jnp.array(0.0, dtype=jnp.float32)
            return zero, zero  # (xp_median, xp_mean)

        return xp_stats

    P = min(num_pairs_xp, P_total)

    def xp_stats(rng, Z):
        # Sample P pairs without replacement
        rng_perm, rng_pairs = jax.random.split(rng)
        perm = jax.random.permutation(rng_perm, P_total)
        chosen = perm[:P]  # (P,)
        pairs = all_pairs[chosen]  # (P, 2)

        rngs = jax.random.split(rng_pairs, P)

        def per_pair(pair, r):
            i = pair[0]
            j = pair[1]
            zi = Z[i]
            zj = Z[j]
            keys = jax.random.split(r, num_rollouts_xp)
            returns = jax.vmap(lambda key: single_game(key, zi, zj))(keys)
            return jnp.mean(returns)

        xp_vals = jax.vmap(per_pair, in_axes=(0, 0))(pairs, rngs)  # (P,)
        xp_median = jnp.median(xp_vals)
        xp_mean = jnp.mean(xp_vals)
        return xp_median, xp_mean

    return xp_stats


# -------------------------
# Objective builder: J(Z) = mean(SP) - xp_weight * mean(XP)
# -------------------------


def build_objective(
    single_game,
    num_latents: int,
    num_rollouts_sp: int,
    all_pairs: jnp.ndarray,
    num_pairs_xp: int,
    num_rollouts_xp: int,
    sp_alpha: float = 0.7,
    xp_median_weight: float = 0.7,
):
    sp_fn = build_sp_estimator(single_game, num_latents, num_rollouts_sp)
    xp_fn = build_xp_estimator(single_game, all_pairs, num_pairs_xp, num_rollouts_xp)

    def objective(theta, rng, xp_weight: float):
        """
        theta: (D,) flattened Z
        rng: PRNGKey
        xp_weight: scalar
        Returns: (J, min_sp, xp_agg, sp_scores)
        """
        M = num_latents
        D = theta.shape[0]
        L = D // M
        Z = theta.reshape((M, L))  # (M, L)

        rng_sp, rng_xp = jax.random.split(rng)

        sp_scores = sp_fn(rng_sp, Z)  # (M,)
        min_sp = jnp.min(sp_scores)
        mean_sp = jnp.mean(sp_scores)

        xp_median, xp_mean = xp_fn(rng_xp, Z)  # scalars

        sp_term = (1.0 - sp_alpha) * mean_sp + sp_alpha * min_sp
        xp_agg = xp_median_weight * xp_median + (1.0 - xp_median_weight) * xp_mean

        J = sp_term - xp_weight * xp_agg
        return J, min_sp, xp_agg, sp_scores

    return objective


# -------------------------
# ES step builder
# -------------------------


def build_es_step(objective, theta_dim: int, population_size: int, sigma: float, lr: float):
    """
    population-based ES:
      eps_k ~ N(0, I)
      theta_k = theta + sigma * eps_k
      J_k = J(theta_k)
      grad ≈ (1/(K*sigma)) sum_k J_k * eps_k
    """

    def es_step(theta, rng, xp_weight: float):
        rng, rng_eps, rng_eval = jax.random.split(rng, 3)

        eps = jax.random.normal(rng_eps, (population_size, theta_dim))  # (K,D)
        thetas = theta[None, :] + sigma * eps  # (K,D)

        rngs = jax.random.split(rng_eval, population_size)

        def eval_one(th, r):
            J, min_sp, xp_agg, _ = objective(th, r, xp_weight)
            return J, min_sp, xp_agg

        Js, min_sps, xp_aggs = jax.vmap(eval_one, in_axes=(0, 0))(thetas, rngs)

        Js_norm = (Js - jnp.mean(Js)) / (jnp.std(Js) + 1e-8)
        grad_hat = (Js_norm[:, None] * eps).mean(axis=0) / sigma
        theta_new = theta + lr * grad_hat

        return theta_new, rng, Js, min_sps, xp_aggs

    return jax.jit(es_step)


# -------------------------
# Full SP/XP diagnostics for a given Z
# -------------------------


def full_sp_xp_diagnostics(
    single_game, Z: jnp.ndarray, num_rollouts_sp_diag: int, num_rollouts_xp_diag: int
):
    """
    Compute full SP vector and XP matrix for diagnostics (expensive but occasional).

    Z: (M, L)
      - SP_diag[i]   = E[ return( z_i vs z_i ) ]
      - XP_diag[i,j] = E[ return( z_i vs z_j ) ]

    We keep this un-jitted at the outer level and just use JAX ops inside,
    because M is small (e.g. 8) and this is a diagnostic path.
    """
    M = int(Z.shape[0])

    # ---------------- SP diagnostics ----------------
    rng = jax.random.PRNGKey(12345)
    rng_sp, rng_xp = jax.random.split(rng)

    rngs_sp = jax.random.split(rng_sp, M)

    def per_latent(r, z):
        keys = jax.random.split(r, num_rollouts_sp_diag)
        returns = jax.vmap(lambda k: single_game(k, z, z))(keys)
        return jnp.mean(returns)

    sp_vec = jax.vmap(per_latent, in_axes=(0, 0))(rngs_sp, Z)  # (M,)

    # ---------------- XP diagnostics ----------------
    xp_np = np.zeros((M, M), dtype=np.float32)

    if M > 1 and num_rollouts_xp_diag > 0:
        rng_xp_loop = rng_xp
        for i in range(M):
            for j in range(i, M):
                rng_xp_loop, key_pair = jax.random.split(rng_xp_loop)
                keys = jax.random.split(key_pair, num_rollouts_xp_diag)
                # Evaluate z_i vs z_j
                returns = jax.vmap(lambda k: single_game(k, Z[i], Z[j]))(keys)
                val = float(jnp.mean(returns))
                xp_np[i, j] = val
                xp_np[j, i] = val

    xp_mat = jnp.array(xp_np)
    return sp_vec, xp_mat


# -------------------------
# Initialization helpers
# -------------------------


def init_latents_from_file(path: str) -> jnp.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected (M,L) array in {path}, got shape {arr.shape}")
    return jnp.array(arr, dtype=jnp.float32)


def init_latents_from_prior(
    prior_json: str, num_latents: int, latent_dim: Optional[int], seed: int
) -> jnp.ndarray:
    rng = jax.random.PRNGKey(seed)
    rngs = jax.random.split(rng, num_latents)
    zs = []
    for i in range(num_latents):
        z_i = sample_z_from_prior(rngs[i], prior_json)
        zs.append(z_i)
    Z = jnp.stack(zs, axis=0)  # (M, L)
    if latent_dim is not None and Z.shape[1] != latent_dim:
        raise ValueError(
            f"latent_dim mismatch: prior produced dim {Z.shape[1]}, but latent_dim={latent_dim}"
        )
    return Z


def save_latents_separately(out_dir: str, prefix: str, Z: np.ndarray):
    """
    Save each latent vector z_i (row of Z) into its own .npy file.

    Files:
      {out_dir}/{prefix}_z000.npy
      {out_dir}/{prefix}_z001.npy
      ...
    """
    os.makedirs(out_dir, exist_ok=True)
    M = Z.shape[0]
    for i in range(M):
        np.save(os.path.join(out_dir, f"{prefix}_z{i:03d}.npy"), Z[i])


# -------------------------
# Main CLI
# -------------------------


def main():
    import argparse

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="ES refinement of VAE latents for Hanabi SP/XP objective."
    )

    # Env
    parser.add_argument("--num_players", type=int, default=2)

    # VAE / network arch
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--preprocessing_features", type=int, nargs="+", default=[1024])
    parser.add_argument("--lstm_features", type=int, nargs="+", default=[512])
    parser.add_argument("--postprocessing_features", type=int, nargs="+", default=[256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--act_fn", type=str, default="gelu")

    # Latent init
    parser.add_argument(
        "--init_z_path",
        type=str,
        default=None,
        help="Path to (M,L) .npy of initial latents. "
        "If not provided, random init from prior_json.",
    )
    parser.add_argument(
        "--prior_json", type=str, default=None, help="Required for random init from prior."
    )
    parser.add_argument(
        "--num_latents",
        type=int,
        default=8,
        help="Number of latents (M) when random init is used.",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Optional latent dim for sanity check on random init.",
    )

    # Objective weights
    parser.add_argument(
        "--xp_weight",
        type=float,
        default=0.5,
        help="Weight λ on mean XP: J = mean_SP - λ * mean_XP.",
    )
    # XP ramping (competence-guarded)
    parser.add_argument(
        "--ramp_xp_weight",
        action="store_true",
        help="If set, use competence-guarded ramp of xp_weight.",
    )
    parser.add_argument(
        "--xp_weight_start",
        type=float,
        default=0.05,
        help="Starting xp_weight during warmup / before gate.",
    )
    parser.add_argument(
        "--xp_warmup_iters",
        type=int,
        default=50,
        help="Number of iterations to keep xp_weight at start value.",
    )
    parser.add_argument(
        "--xp_ramp_iters",
        type=int,
        default=200,
        help="Number of iterations over which to linearly ramp to --xp_weight.",
    )
    parser.add_argument(
        "--sp_gate",
        type=float,
        default=8.0,
        help="Begin ramp only once min_sp >= sp_gate (competence gate).",
    )

    # ES hyperparams
    parser.add_argument("--num_iterations", type=int, default=50)
    parser.add_argument("--population_size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.05)

    # Rollout budgets per ES eval
    parser.add_argument("--num_rollouts_sp", type=int, default=64)
    parser.add_argument("--num_rollouts_xp", type=int, default=32)
    parser.add_argument("--num_pairs_xp", type=int, default=8)

    # Diagnostic budgets (more expensive, run infrequently)
    parser.add_argument(
        "--diag_every",
        type=int,
        default=10,
        help="Every N ES iterations, run full SP/XP diagnostics.",
    )
    parser.add_argument("--num_rollouts_sp_diag", type=int, default=128)
    parser.add_argument("--num_rollouts_xp_diag", type=int, default=64)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    def xp_weight_schedule(it: int, min_sp_est: float) -> float:
        if not args.ramp_xp_weight:
            return float(args.xp_weight)

        w_max = float(args.xp_weight)
        w0 = float(args.xp_weight_start)

        if it <= args.xp_warmup_iters:
            return min(w0, w_max)

        if min_sp_est < args.sp_gate:
            return min(w0, w_max)

        t = it - args.xp_warmup_iters
        frac = min(1.0, max(0.0, t / float(args.xp_ramp_iters)))
        return w0 + (w_max - w0) * frac

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Build env + actor
    logger.info("Building environment and actor...")
    env = build_env(args.num_players)
    action_dim = infer_action_dim(env)

    actor_mod = ActorLstm(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=args.act_fn,
    )
    actor_vars = load_actor_vars_for_vae(args.vae_ckpt)
    single_game = make_vae_single_game_fn(env, actor_mod, actor_vars, args.num_players)

    # 2) Initialize Z
    logger.info("Initializing latents...")
    if args.init_z_path is not None:
        Z0 = init_latents_from_file(args.init_z_path)
        logger.info(f"Loaded initial latents from {args.init_z_path} with shape {Z0.shape}.")
    else:
        if args.prior_json is None:
            raise ValueError("Random init requested but --prior_json is not provided.")
        Z0 = init_latents_from_prior(
            prior_json=args.prior_json,
            num_latents=args.num_latents,
            latent_dim=args.latent_dim,
            seed=args.seed,
        )
        logger.info(
            f"Randomly initialized latents from prior {args.prior_json} with shape {Z0.shape}."
        )

    M, L = int(Z0.shape[0]), int(Z0.shape[1])
    logger.info(f"ES will optimize M={M} latents of dim L={L} (total D={M * L}).")

    # 3) Precompute pair indices for XP
    if M > 1:
        pair_list = [(i, j) for i in range(M) for j in range(i + 1, M)]
        all_pairs = jnp.array(pair_list, dtype=jnp.int32)
        logger.info(f"Total unique pairs for XP: {all_pairs.shape[0]}.")
    else:
        all_pairs = jnp.zeros((0, 2), dtype=jnp.int32)
        logger.info("Only one latent: XP term will be zero.")

    # 4) Build objective & ES step
    objective = build_objective(
        single_game=single_game,
        num_latents=M,
        num_rollouts_sp=args.num_rollouts_sp,
        all_pairs=all_pairs,
        num_pairs_xp=args.num_pairs_xp,
        num_rollouts_xp=args.num_rollouts_xp,
    )

    theta0 = Z0.reshape(-1)  # (D,)
    theta_dim = int(theta0.shape[0])

    es_step = build_es_step(
        objective=objective,
        theta_dim=theta_dim,
        population_size=args.population_size,
        sigma=args.sigma,
        lr=args.lr,
    )

    # 5) ES loop
    rng = jax.random.PRNGKey(args.seed)
    theta = theta0

    rng, rng_probe = jax.random.split(rng)
    probe_w = float(args.xp_weight_start) if args.ramp_xp_weight else float(args.xp_weight)
    _, min_sp_est_jax, _, _ = objective(theta, rng_probe, probe_w)
    min_sp_est = float(min_sp_est_jax)

    log_history = []

    logger.info("Starting ES optimization...")
    for it in range(1, args.num_iterations + 1):
        xp_w = xp_weight_schedule(it, min_sp_est)

        theta, rng, Js, min_sps, xp_aggs = es_step(theta, rng, xp_w)

        Js_np = np.array(Js)
        min_sps_np = np.array(min_sps)
        xp_aggs_np = np.array(xp_aggs)

        log_entry = {
            "iter": it,
            "xp_weight_used": float(xp_w),
            "J_mean": float(Js_np.mean()),
            "J_max": float(Js_np.max()),
            "minSP_mean_pop": float(min_sps_np.mean()),
            "minSP_max_pop": float(min_sps_np.max()),
            "XPagg_mean_pop": float(xp_aggs_np.mean()),
            "XPagg_min_pop": float(xp_aggs_np.min()),
        }
        log_history.append(log_entry)

        logger.info(
            f"[Iter {it:03d}] xp_w={xp_w:.3f} "
            f"J_mean={log_entry['J_mean']:.3f}, J_max={log_entry['J_max']:.3f}, "
            f"minSP_mean={log_entry['minSP_mean_pop']:.3f}, "
            f"XPagg_mean={log_entry['XPagg_mean_pop']:.3f}"
        )

        # Update competence estimate for next iteration's gate
        rng, rng_probe = jax.random.split(rng)
        _, min_sp_est_jax, _, _ = objective(theta, rng_probe, xp_w)
        min_sp_est = float(min_sp_est_jax)

        # Diagnostics: full SP/XP on current theta
        if args.diag_every > 0 and (it % args.diag_every == 0 or it == args.num_iterations):
            Z_curr = theta.reshape(M, L)
            sp_vec, xp_mat = full_sp_xp_diagnostics(
                single_game=single_game,
                Z=Z_curr,
                num_rollouts_sp_diag=args.num_rollouts_sp_diag,
                num_rollouts_xp_diag=args.num_rollouts_xp_diag,
            )
            sp_vec_np = np.array(sp_vec)
            xp_mat_np = np.array(xp_mat)

            diag = {
                "iter": it,
                "SP_diag_mean": float(sp_vec_np.mean()),
                "SP_diag_min": float(sp_vec_np.min()),
                "SP_diag_max": float(sp_vec_np.max()),
            }
            if xp_mat_np.shape[0] > 1:
                off_diag = xp_mat_np[~np.eye(M, dtype=bool)]
                diag.update(
                    {
                        "XP_diag_mean": float(off_diag.mean()),
                        "XP_diag_min": float(off_diag.min()),
                        "XP_diag_max": float(off_diag.max()),
                    }
                )
            logger.info(
                f"[Diag {it:03d}] "
                f"SP_diag_mean={diag['SP_diag_mean']:.3f}, "
                f"SP_diag_min={diag['SP_diag_min']:.3f}, "
                f"SP_diag_max={diag['SP_diag_max']:.3f}"
            )
            if xp_mat_np.shape[0] > 1:
                logger.info(
                    f"[Diag {it:03d}] "
                    f"XP_diag_mean={diag['XP_diag_mean']:.3f}, "
                    f"XP_diag_min={diag['XP_diag_min']:.3f}, "
                    f"XP_diag_max={diag['XP_diag_max']:.3f}"
                )

            # Save snapshots
            # np.save(os.path.join(args.out_dir, f"es_Z_iter{it}.npy"), Z_curr)
            # save_latents_separately(args.out_dir, f"es_Z_iter{it}", np.array(Z_curr))
            # np.save(os.path.join(args.out_dir, f"es_SP_iter{it}.npy"), sp_vec_np)
            # np.save(os.path.join(args.out_dir, f"es_XP_iter{it}.npy"), xp_mat_np)

    # 6) Final save
    Z_final = theta.reshape(M, L)
    sp_final, xp_final = full_sp_xp_diagnostics(
        single_game=single_game,
        Z=Z_final,
        num_rollouts_sp_diag=args.num_rollouts_sp_diag,
        num_rollouts_xp_diag=args.num_rollouts_xp_diag,
    )

    Z_final_np = np.array(Z_final)
    sp_final_np = np.array(sp_final)
    xp_final_np = np.array(xp_final)

    np.save(os.path.join(args.out_dir, "es_final_Z.npy"), Z_final_np)
    np.save(os.path.join(args.out_dir, "es_final_SP.npy"), sp_final_np)
    np.save(os.path.join(args.out_dir, "es_final_XP.npy"), xp_final_np)

    xp_avg_final_np = per_latent_xp_avg(xp_final_np)
    save_latents_with_scores(
        args.out_dir,
        prefix="es_final_Z",
        Z=Z_final_np,
        sp_vec=sp_final_np,
        xp_avg=xp_avg_final_np,
        decimals=3,
    )

    with open(os.path.join(args.out_dir, "es_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    logger.info("ES optimization complete.")
    logger.info(f"Final Z saved to {os.path.join(args.out_dir, 'es_final_Z.npy')}")


if __name__ == "__main__":
    main()
