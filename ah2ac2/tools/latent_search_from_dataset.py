# ah2ac2/tools/latent_search_from_dataset.py

import json
import logging
import os
from types import SimpleNamespace
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl import make

from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset

# Reuse Hanabi wrapper + actor core from crossplay_eval
from ah2ac2.tools.crossplay_eval import (
    ActorLstm,
    LogWrapper,
    load_actor_vars_for_vae,
    unwrap_state,
)
from ah2ac2.training.vae_2p_masked import (
    VAE,
    make_trajectory_batcher,
    to_time_major,
    transform_trajectory_batch,
)

"""
e.g.

python -m ah2ac2.tools.latent_search_from_dataset \
  --num_players 2 \
  --vae_ckpt checkpoints/vae/human_vae/latent_dim=16/5k/epoch52_seed0_valacc0.597.safetensors \
  --train_data_path ah2ac2/datasets/data/2_player_games_train.safetensors \
  --diversity_weight 0.7 \
  --latent_dim 16 \
  --max_trajectories 2048 \
  --num_rollouts_sp 128 \
  --num_rollouts_xp 128 \
  --top_k_by_sp 0 \
  --final_set_size 8 \
  --seed 0 \
  --out_dir checkpoints/vae/human_vae/latent_dim=16/5k/latent_search/

python -m ah2ac2.tools.latent_search_from_dataset \
  --num_players 2 \
  --vae_ckpt checkpoints/vae/human_vae/latent_dim=32/100k/vae_on_all_human.safetensors \
  --train_data_path ah2ac2/datasets/data/2_player_games_train.safetensors \
  --diversity_weight 0.7 \
  --latent_dim 32 \
  --max_trajectories 2048 \
  --num_rollouts_sp 128 \
  --num_rollouts_xp 128 \
  --top_k_by_sp 0 \
  --final_set_size 8 \
  --seed 0 \
  --out_dir checkpoints/vae/human_vae/latent_dim=32/100k/latent_search/

python -m ah2ac2.tools.latent_search_from_dataset \
  --num_players 2 \
  --vae_ckpt checkpoints/vae_lowdim_decoder/human_vae/latent_dim=32/3k/epoch60_seed0_valacc0.638.safetensors \
  --train_data_path ah2ac2/datasets/data/2_player_games_train.safetensors \
  --diversity_weight 0.7 \
  --latent_dim 32 \
  --max_trajectories 2048 \
  --num_rollouts_sp 128 \
  --num_rollouts_xp 128 \
  --top_k_by_sp 0 \
  --final_set_size 8 \
  --seed 0 \
  --out_dir checkpoints/vae_lowdim_decoder/human_vae/latent_dim=32/3k/latent_search/ \
  --lstm_features 256 \
  --postprocessing_features 128

python -m ah2ac2.tools.latent_search_from_dataset \
  --num_players 2 \
  --vae_ckpt checkpoints/vae_masked/human_vae/latent_dim32/100k/epoch98_seed0_valacc0.813.safetensors \
  --train_data_path ah2ac2/datasets/data/2_player_games_val.safetensors \
  --diversity_weight 0.7 \
  --latent_dim 32 \
  --max_trajectories 2048 \
  --num_rollouts_sp 128 \
  --num_rollouts_xp 128 \
  --top_k_by_sp 0 \
  --final_set_size 8 \
  --seed 0 \
  --out_dir checkpoints/vae_masked/human_vae/latent_dim32/100k/latent_search/ \
  --lstm_features 512 \
  --postprocessing_features 256

python -m ah2ac2.tools.latent_search_from_dataset \
  --num_players 2 \
  --vae_ckpt checkpoints/vae_lowdim_masked/human_vae/latent_dim32/100k/epoch99_seed0_valacc0.803.safetensors \
  --train_data_path ah2ac2/datasets/data/2_player_games_test.safetensors \
  --diversity_weight 0.4 \
  --latent_dim 32 \
  --max_trajectories 2048 \
  --num_rollouts_sp 128 \
  --num_rollouts_xp 128 \
  --top_k_by_sp 0 \
  --final_set_size 3 \
  --seed 5 \
  --out_dir checkpoints/vae_lowdim_masked/human_vae/latent_dim32/100k/latent_search/ \
  --lstm_features 256 \
  --postprocessing_features 128
"""


# -------------------------
# Env + VAE loading
# -------------------------


def build_env(num_players: int):
    hand_size = 5 if num_players <= 3 else 4
    base_env = make("hanabi", hand_size=hand_size, num_agents=num_players)
    env = LogWrapper(base_env)
    return env


def infer_action_dim(env):
    rng = jax.random.PRNGKey(0)
    obs0, env_state0 = env.reset(rng)
    legal0 = env.get_legal_moves(getattr(env_state0, "env_state", env_state0))
    first_agent = list(env.agents)[0]
    return int(legal0[first_agent].shape[-1])


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

    # Shape inference: as in vae_2p, we construct dummy inputs
    T, B, D = cfg.dummy_TBD
    dummy_o = jnp.ones((T, B, D), jnp.float32)
    dummy_a = jnp.zeros((T, B, 1), jnp.int32)
    params = model.init(rng, obs_seq=dummy_o, act_seq=dummy_a, training=False)["params"]

    # Now load checkpoint weights (params only)
    from safetensors.numpy import load_file as st_load

    flat = st_load(cfg.vae_ckpt)
    flat = {k: jnp.array(v) for k, v in flat.items()}

    from flax.core import freeze
    from flax.traverse_util import unflatten_dict

    def split_key(k: str):
        if "/" in k:
            return tuple(k.split("/"))
        if "," in k:
            return tuple(k.split(","))
        return (k,)

    pairs = {split_key(k): v for k, v in flat.items()}
    tree = unflatten_dict(pairs)  # nested dict

    if "params" in tree:
        params = freeze(tree["params"])
    else:
        params = freeze(tree)

    return model, params


def extract_latents_from_dataset(
    rng,
    model: VAE,
    params,
    ds: HanabiLiveGamesDataset,
    dl: HanabiLiveGamesDataloader,
    num_players: int,
    max_trajectories: int,
) -> Tuple[jnp.ndarray, List[dict]]:
    """
    Returns:
        latents: (N, latent_dim)
        meta: list of dicts with (batch_idx, seq_idx, maybe game_len, etc.)
    """
    get_traj_batch = jax.jit(jax.vmap(make_trajectory_batcher(num_players), in_axes=0))

    latents = []
    meta = []
    num_seen = 0

    for batch_idx, batch in enumerate(dl):
        if num_seen >= max_trajectories:
            break

        rng, batch_rng = jax.random.split(rng)
        traj_keys = jax.random.split(batch_rng, batch.actions.shape[0])
        traj_b = get_traj_batch(traj_keys, batch.decks, batch.actions)

        obs_ba, act_ba, _, _ = transform_trajectory_batch(traj_b, batch.game_len_masks)
        obs_tm, act_tm = to_time_major(obs_ba, act_ba)  # (T, B*, D)

        # Forward VAE once; grab encoder's mu
        _, outs = model.apply(
            {"params": params},
            obs_seq=obs_tm,
            act_seq=act_tm,
            training=False,
            rngs={
                "latent": jax.random.PRNGKey(0),  # not used for mu
                "dropout": jax.random.PRNGKey(1),
            },
        )

        mu = outs["mu"]  # (B*, latent_dim)

        for i in range(mu.shape[0]):
            if num_seen >= max_trajectories:
                break
            z_i = mu[i]
            latents.append(z_i)
            meta.append(
                {
                    "batch_idx": int(batch_idx),
                    "seq_idx": int(i),
                    "game_len": int(batch.game_len_masks[i // ds.num_players].sum()),
                }
            )
            num_seen += 1

    if not latents:
        raise ValueError("No latents extracted; check max_trajectories / dataset size.")

    latents = jnp.stack(latents, axis=0)  # (N, latent_dim)
    return latents, meta


# -------------------------
# JAX-native SP/XP evaluation
# -------------------------


def _make_vae_single_game_fn(env, actor_mod, actor_vars, num_players: int):
    """
    Returns a jitted function single_game(rng, z1, z2) -> scalar return,
    where z1 controls player 0 and z2 controls player 1 (via the usual
    curr_player switching logic from crossplay_eval).
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

            # Same semantics as crossplay_eval: policy index = curr_player
            actions = jax.lax.cond(
                curr_player == 1,
                lambda _: build_actions(actions2),
                lambda _: build_actions(actions1),
                operand=None,
            )

            rng, env_state, obs, legal_moves, reward, dones = _step_env(
                step_rng, env_state, actions
            )

            done = dones["__all__"]
            cum_rew = cum_rew + reward["__all__"]
            return (
                cum_rew,
                done,
                rng,
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

    # JIT over (rng, z1, z2); env/actor_mod/actor_vars/agents are closed over
    return jax.jit(single_game)


def compute_self_play_scores(
    env,
    actor_mod,
    actor_vars,
    latents: jnp.ndarray,
    num_rollouts_sp: int,
    rng_seed: int,
) -> np.ndarray:
    """
    JAX-vectorized self-play:
        For each latent z, evaluate VAE(z) vs VAE(z) over num_rollouts_sp games.
    """
    N = latents.shape[0]
    single_game = _make_vae_single_game_fn(env, actor_mod, actor_vars, num_players=len(env.agents))

    rng = jax.random.PRNGKey(rng_seed)
    rng_latents = jax.random.split(rng, N)

    def per_latent(r, z):
        rollout_keys = jax.random.split(r, num_rollouts_sp)
        returns = jax.vmap(lambda key: single_game(key, z, z))(rollout_keys)
        return jnp.mean(returns)

    sp_scores = jax.vmap(per_latent, in_axes=(0, 0))(rng_latents, latents)
    return np.array(sp_scores, dtype=np.float32)


def select_top_k_by_sp(sp_scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, sp_scores.shape[0])
    return np.argsort(-sp_scores)[:k]


def compute_crossplay_matrix(
    env,
    actor_mod,
    actor_vars,
    latents_top: jnp.ndarray,
    num_rollouts_xp: int,
    rng_seed: int,
) -> np.ndarray:
    """
    JAX-vectorized cross-play:
        For K latents, compute a KxK matrix of XP scores.
        We only simulate pairs (i,j) for i <= j and mirror.
    """
    K = latents_top.shape[0]
    single_game = _make_vae_single_game_fn(env, actor_mod, actor_vars, num_players=len(env.agents))

    # Upper triangle indices (including diagonal)
    tri_i, tri_j = np.triu_indices(K)
    P = len(tri_i)

    z1 = latents_top[tri_i]
    z2 = latents_top[tri_j]

    rng = jax.random.PRNGKey(rng_seed)
    rng_pairs = jax.random.split(rng, P)

    def per_pair(r, zi, zj):
        rollout_keys = jax.random.split(r, num_rollouts_xp)
        returns = jax.vmap(lambda key: single_game(key, zi, zj))(rollout_keys)
        return jnp.mean(returns)

    xp_scores = jax.vmap(per_pair, in_axes=(0, 0, 0))(rng_pairs, z1, z2)  # (P,)
    xp_scores = np.array(xp_scores, dtype=np.float32)

    xp = np.zeros((K, K), dtype=np.float32)
    xp[tri_i, tri_j] = xp_scores
    xp[tri_j, tri_i] = xp_scores
    return xp


# -------------------------
# Geometry-based k-center selection
# -------------------------


def select_diverse_latents_geom(
    latents_top: jnp.ndarray,
    sp_scores_top: np.ndarray,
    final_set_size: int,
    diversity_weight: float = 0.7,
    eps: float = 1e-6,
) -> List[int]:
    """
    Greedy k-center style selection in (whitened) latent space, with SP tradeoff.

    Args:
        latents_top: (K, L) candidate latents (top-K by SP).
        sp_scores_top: (K,) self-play scores for those latents.
        final_set_size: desired subset size (<= K).
        diversity_weight: α in [0,1]. α=1.0 -> pure diversity after seeding;
                          α=0.0 -> pure self-play.
        eps: small jitter for numerical stability.

    Returns:
        selected: list of indices 0..K-1 into the top-K slice.
    """
    latents_np = np.asarray(latents_top, dtype=np.float32)
    sp = np.asarray(sp_scores_top, dtype=np.float32)

    K = latents_np.shape[0]
    if K == 0 or final_set_size <= 0:
        return []

    final_set_size = min(final_set_size, K)

    # --- 1) Whiten latents to approximate Mahalanobis distance ---
    mu = latents_np.mean(axis=0, keepdims=True)  # (1, L)
    X = latents_np - mu  # (K, L)

    # Covariance estimate + jitter
    Ldim = X.shape[1]
    cov = (X.T @ X) / max(K, 1) + eps * np.eye(Ldim, dtype=np.float32)

    # Eigen-decomposition for cov^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(cov)  # cov = V diag(e) V^T
    inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals + eps)
    cov_inv_sqrt = eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T

    # Whitened latents
    Y = (cov_inv_sqrt @ X.T).T  # (K, L)

    # --- 2) Pairwise distances in whitened space ---
    # d_ij = ||Y_i - Y_j||_2
    diff = Y[:, None, :] - Y[None, :, :]  # (K, K, L)
    dist = np.linalg.norm(diff, axis=-1)  # (K, K)

    # Normalise distances into [0,1] for scoring
    d_max = float(dist.max())
    if d_max > 0.0:
        dist_norm = dist / d_max
    else:
        dist_norm = dist

    # --- 3) Normalise SP to [0,1] ---
    sp_min = float(sp.min())
    sp_max = float(sp.max())
    if sp_max > sp_min:
        sp_norm = (sp - sp_min) / (sp_max - sp_min)
    else:
        sp_norm = np.zeros_like(sp, dtype=np.float32)

    # --- 4) Seed with the best SP latent ---
    first = int(np.argmax(sp))
    selected = [first]
    remaining = set(range(K)) - {first}

    # --- 5) Greedy k-center with SP/diversity tradeoff ---
    while len(selected) < final_set_size and remaining:
        best_i = None
        best_score = -np.inf

        for i in remaining:
            # distance to closest already-selected point (k-center objective)
            if len(selected) == 1:
                min_dist = float(dist_norm[i, selected[0]])
            else:
                min_dist = float(dist_norm[i, selected].min())

            diversity_term = min_dist  # ~[0,1]
            quality_term = float(sp_norm[i])  # ~[0,1]

            score = diversity_weight * diversity_term + (1.0 - diversity_weight) * quality_term

            if score > best_score:
                best_score = score
                best_i = i

        if best_i is None:
            break
        selected.append(best_i)
        remaining.remove(best_i)

    return selected


# -------------------------
# CLI entrypoint
# -------------------------


def main():
    import argparse

    # --- Logging setup ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--num_players", type=int, default=2)
    # VAE / arch
    parser.add_argument("--vae_ckpt", type=str, required=True)
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--preprocessing_features", type=int, nargs="+", default=[1024])
    parser.add_argument("--lstm_features", type=int, nargs="+", default=[512])
    parser.add_argument("--postprocessing_features", type=int, nargs="+", default=[256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--act_fn", type=str, default="gelu")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--prior_json", type=str, default=None
    )  # unused here, kept for compatibility

    # search
    parser.add_argument("--max_trajectories", type=int, default=128)
    parser.add_argument("--num_rollouts_sp", type=int, default=64)
    parser.add_argument("--num_rollouts_xp", type=int, default=64)
    parser.add_argument("--top_k_by_sp", type=int, default=16)
    parser.add_argument("--final_set_size", type=int, default=8)
    parser.add_argument("--diversity_weight", type=float, default=0.7)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, required=True)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Env + action_dim
    env = build_env(args.num_players)
    action_dim = infer_action_dim(env)

    # 2) Dataset + loader
    rng = jax.random.PRNGKey(args.seed)
    ds = HanabiLiveGamesDataset(file=args.train_data_path)
    rng, loader_rng = jax.random.split(rng)
    dl = HanabiLiveGamesDataloader(ds, args.batch_size, loader_rng)

    # 3) Dummy T,B,D for VAE init
    dummy_batch = next(iter(dl))
    rng, dummy_rng = jax.random.split(rng)
    get_traj_batch = jax.jit(jax.vmap(make_trajectory_batcher(args.num_players), in_axes=0))
    dummy_keys = jax.random.split(dummy_rng, args.batch_size)
    dummy_traj = get_traj_batch(dummy_keys, dummy_batch.decks, dummy_batch.actions)
    dummy_obs_ba, dummy_act_ba, _, _ = transform_trajectory_batch(
        dummy_traj, dummy_batch.game_len_masks
    )
    dummy_obs_tm, dummy_act_tm = to_time_major(dummy_obs_ba, dummy_act_ba)
    T, B, D = dummy_obs_tm.shape

    # 4) Config shim for VAE loader
    cfg = SimpleNamespace(
        preprocessing_features=tuple(args.preprocessing_features),
        lstm_features=tuple(args.lstm_features),
        postprocessing_features=tuple(args.postprocessing_features),
        action_dim=action_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        act_fn=args.act_fn,
        beta=args.beta,
        dummy_TBD=(T, B, D),
        vae_ckpt=args.vae_ckpt,
    )

    rng, vae_rng = jax.random.split(rng)
    vae_model, vae_params = load_vae_model_and_params(vae_rng, cfg)

    # 5) Extract latents
    latents, meta = extract_latents_from_dataset(
        rng,
        vae_model,
        vae_params,
        ds,
        dl,
        num_players=args.num_players,
        max_trajectories=args.max_trajectories,
    )

    np.save(os.path.join(args.out_dir, "candidate_z.npy"), np.array(latents))
    with open(os.path.join(args.out_dir, "candidate_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Extracted {latents.shape[0]} latents of dim {latents.shape[1]}.")
    logger.info(f"Saved candidate_z.npy and candidate_meta.json to {args.out_dir}.")

    # 6) Build actor module + decoder params for VAE policy
    actor_mod = ActorLstm(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=args.act_fn,
    )
    actor_vars = load_actor_vars_for_vae(args.vae_ckpt)

    # 7) Self-play scores (JAX-native)
    sp_scores = compute_self_play_scores(
        env=env,
        actor_mod=actor_mod,
        actor_vars=actor_vars,
        latents=latents,
        num_rollouts_sp=args.num_rollouts_sp,
        rng_seed=args.seed,
    )
    # We'll use the NumPy view for indexing & saving
    latents_np = np.array(latents)

    np.save(os.path.join(args.out_dir, "sp_scores.npy"), sp_scores)

    logger.info(
        f"Computed self-play scores for {len(sp_scores)} latents "
        f"(mean={float(sp_scores.mean()):.3f}, "
        f"min={float(sp_scores.min()):.3f}, "
        f"max={float(sp_scores.max()):.3f})."
    )
    logger.info(f"Saved sp_scores.npy to {args.out_dir}.")

    # 8) Optional SP filtering (gentle, not hard top-K)
    if args.top_k_by_sp > 0:
        # interpret as "keep this many best" but we’ll run geometry over *that* pool
        top_idx = select_top_k_by_sp(sp_scores, args.top_k_by_sp)
        pool_latents = latents_np[top_idx]
        pool_scores = sp_scores[top_idx]
        pool_indices = top_idx
    else:
        # no truncation: geometry over all
        pool_latents = latents_np
        pool_scores = sp_scores
        pool_indices = np.arange(len(latents_np))

    # 9) Geometry-based diverse subset from the pool
    selected_rel = select_diverse_latents_geom(
        latents_top=pool_latents,
        sp_scores_top=pool_scores,
        final_set_size=args.final_set_size,
        diversity_weight=args.diversity_weight,
    )
    selected_abs = [int(pool_indices[i]) for i in selected_rel]

    # 9b) Save the selected latents + their SP scores
    selected_latents = latents_np[np.array(selected_abs, dtype=int)]
    selected_sp_scores = sp_scores[np.array(selected_abs, dtype=int)]

    np.save(os.path.join(args.out_dir, "selected_z.npy"), selected_latents)
    np.save(os.path.join(args.out_dir, "selected_sp_scores.npy"), selected_sp_scores)

    with open(os.path.join(args.out_dir, "selected_indices.json"), "w") as f:
        json.dump(
            {
                "selected_relative": selected_rel,
                "selected_absolute": selected_abs,
            },
            f,
            indent=2,
        )

    logger.info(
        f"Selected {len(selected_abs)} latents after geometry-based k-center. "
        f"Indices (absolute): {selected_abs}"
    )

    # 10) Optional XP diagnostics for the final selected set
    if args.num_rollouts_xp > 0 and len(selected_abs) > 0:
        # use the already-constructed NumPy array for indexing, then convert to jnp
        latents_selected = jnp.array(selected_latents)
        xp_matrix_sel = compute_crossplay_matrix(
            env=env,
            actor_mod=actor_mod,
            actor_vars=actor_vars,
            latents_top=latents_selected,
            num_rollouts_xp=args.num_rollouts_xp,
            rng_seed=args.seed + 9999,
        )
        np.save(os.path.join(args.out_dir, "xp_matrix_selected.npy"), xp_matrix_sel)

        if xp_matrix_sel.shape[0] > 1:
            Ksel = xp_matrix_sel.shape[0]
            off_diag = xp_matrix_sel[~np.eye(Ksel, dtype=bool)]
            logger.info(
                f"[XP-selected] shape={xp_matrix_sel.shape}. "
                f"Off-diagonal mean={float(off_diag.mean()):.3f}, "
                f"min={float(off_diag.min()):.3f}, "
                f"max={float(off_diag.max()):.3f}."
            )
        else:
            logger.info("[XP-selected] Only one latent selected; no off-diagonal XP.")
    else:
        logger.info(
            "Skipping XP diagnostics for selected set (num_rollouts_xp <= 0 or no selection)."
        )

    logger.info("Done.")
    print("Done.")
    print("Selected latent indices (absolute):", selected_abs)


if __name__ == "__main__":
    main()
