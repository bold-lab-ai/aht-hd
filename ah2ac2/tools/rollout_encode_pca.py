#!/usr/bin/env python3
"""
rollout_encode_pca.py

Generate greedy rollouts from a single policy (RL_BC or RL_VAE), encode each episode
through a saved VAE *encoder* to get a latent vector, then run PCA to 2D and
save the points as .npy.

Outputs:
  - --out_npy: (N,2) PCA coordinates
Optional:
  - --out_latents_npy: (N,L) raw episode latents (before PCA)

Notes / assumptions (match your codebase conventions):
  - Policy is run in self-play: same policy controls both seats.
  - Observations come from LogWrapper env (same as RL training).
  - Encoder aggregation is *mask-weighted over time* to avoid padding bias.
  - Episode latent = mean over players of their per-seat latents.


e.g.
python ah2ac2/tools/rollout_encode_pca.py \
  --policy_type rl_bc \
  --rl_ckpt checkpoints/bc_test_proxies/bc_proxy_1/seed1_score23.723_klt0.05_2p.safetensors \
  --vae_ckpt checkpoints/vae_masked/human_vae/latent_dim32/100k/epoch98_seed0_valacc0.813.safetensors \
  --vae_latent_dim 32 \
  --episodes 256 \
  --max_steps 80 \
  --seed 0 \
  --out_npy outputs/pca/bc_proxy1_points.npy

python ah2ac2/tools/rollout_encode_pca.py \
  --policy_type rl_vae \
  --rl_ckpt checkpoints/test_proxies/es_final_Z_i002_SP=17.266_XPavg=2.453/seed1_score21.877_klt0.064_2p.safetensors \
  --z_npy outputs/es_refine_test/es_final_Z_i002_SP=17.266_XPavg=2.453.npy \
  --vae_ckpt checkpoints/vae_masked/human_vae/latent_dim32/100k/epoch98_seed0_valacc0.813.safetensors \
  --vae_latent_dim 32 \
  --episodes 256 \
  --max_steps 80 \
  --seed 0 \
  --out_npy outputs/pca/latent_proxy3_points.npy
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.traverse_util import unflatten_dict
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper

from ah2ac2.nn.checkpoint_utils import unwrap_vae_params

# ---- your project modules ----
from ah2ac2.nn.multi_layer_lstm import LstmCellWithHiddenStateReset

# ==========================================================
# Load safetensors -> Flax tree (same as your harness)
# ==========================================================


def load_flax_safetensors(path: str):
    from safetensors.numpy import load_file as st_load

    flat = st_load(path)  # Dict[str, np.ndarray]
    flat = {k: jnp.array(v) for k, v in flat.items()}

    def split_key(k: str):
        if "/" in k:
            return tuple(k.split("/"))
        if "," in k:
            return tuple(k.split(","))
        return (k,)

    pairs = {split_key(k): v for k, v in flat.items()}
    tree = unflatten_dict(pairs)
    if "params" not in tree:
        tree = {"params": tree}
    return freeze(tree)


# ==========================================================
# Policy network (actor-only) — adapted from your crossplay_eval.py
# ==========================================================


def zeros_lstm_carry(lstm_features: List[int], batch_size: int):
    cs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in lstm_features]
    hs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in lstm_features]
    return jnp.stack(cs), jnp.stack(hs)


class ActorLstm(nn.Module):
    preprocessing_features: List[int]
    lstm_features: List[int]
    postprocessing_features: List[int]
    action_dim: int
    dropout_rate: float
    activation_fn_name: str

    def setup(self):
        self.lstm = nn.scan(
            LstmCellWithHiddenStateReset,
            variable_broadcast="params",
            in_axes=0,
            out_axes=0,
            split_rngs={"params": False},
        )
        self.act_fn = {"relu": nn.relu, "gelu": nn.gelu}.get(self.activation_fn_name, nn.relu)

    @nn.compact
    def __call__(self, carry, x, z: jnp.ndarray = None, training: bool = False):
        # x: (obs, resets, avail)  where obs:(T,B,D), resets:(T,B), avail:(T,B,A)
        obs, resets, avail = x

        if z is not None:
            T = obs.shape[0]
            if z.ndim == 2:  # (B,L) -> (T,B,L)
                z = jnp.broadcast_to(z[None, ...], (T,) + z.shape)
            obs = jnp.concatenate([obs, z], axis=-1)

        xh = obs
        for feature_size in self.preprocessing_features:
            xh = nn.Dense(feature_size)(xh)
            xh = nn.LayerNorm()(xh)
            xh = self.act_fn(xh)

        new_cs, new_hs = [], []
        for i, feature_size in enumerate(self.lstm_features):
            layer = self.lstm(feature_size)
            layer_carry = jax.tree.map(lambda c: c[i], carry)
            layer_carry, xh = layer(layer_carry, (xh, resets))
            new_c, new_h = layer_carry
            new_cs.append(new_c)
            new_hs.append(new_h)

        for feature_size in self.postprocessing_features:
            xh = nn.Dense(feature_size)(xh)
            xh = self.act_fn(xh)
            xh = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(xh)

        logits = nn.Dense(self.action_dim)(xh)  # (T,B,A)
        # mask in-network (keep consistent with training usage)
        logits = logits - (1.0 - avail) * 1e10

        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_carry, logits

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return zeros_lstm_carry(self.lstm_features, batch_size)


class ActorLstmPolicyOut(nn.Module):
    preprocessing_features: List[int]
    lstm_features: List[int]
    postprocessing_features: List[int]
    action_dim: int
    dropout_rate: float
    activation_fn_name: str

    def setup(self):
        self.lstm = nn.scan(
            LstmCellWithHiddenStateReset,
            variable_broadcast="params",
            in_axes=0,
            out_axes=0,
            split_rngs={"params": False},
        )
        self.act_fn = {"relu": nn.relu, "gelu": nn.gelu}.get(self.activation_fn_name, nn.relu)

    @nn.compact
    def __call__(self, carry, x, z: jnp.ndarray = None, training: bool = False):
        obs, resets, avail = x

        if z is not None:
            T = obs.shape[0]
            if z.ndim == 2:
                z = jnp.broadcast_to(z[None, ...], (T,) + z.shape)
            obs = jnp.concatenate([obs, z], axis=-1)

        xh = obs
        for feature_size in self.preprocessing_features:
            xh = nn.Dense(feature_size)(xh)
            xh = nn.LayerNorm()(xh)
            xh = self.act_fn(xh)

        new_cs, new_hs = [], []
        for i, feature_size in enumerate(self.lstm_features):
            layer = self.lstm(feature_size)
            layer_carry = jax.tree.map(lambda c: c[i], carry)
            layer_carry, xh = layer(layer_carry, (xh, resets))
            new_c, new_h = layer_carry
            new_cs.append(new_c)
            new_hs.append(new_h)

        for feature_size in self.postprocessing_features:
            xh = nn.Dense(feature_size)(xh)
            xh = self.act_fn(xh)
            xh = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(xh)

        logits = nn.Dense(self.action_dim, name="policy_out")(xh)
        logits = logits - (1.0 - avail) * 1e10

        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_carry, logits

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return zeros_lstm_carry(self.lstm_features, batch_size)


def actor_uses_policy_out(actor_params_tree) -> bool:
    p = unfreeze(actor_params_tree)

    def has_policy_out(d):
        if not isinstance(d, dict):
            return False
        if "policy_out" in d:
            return True
        return any(has_policy_out(v) for v in d.values() if isinstance(v, dict))

    return has_policy_out(p)


def load_actor_vars_for_rl(params_path: str):
    full_vars = load_flax_safetensors(params_path)
    p = unfreeze(full_vars)
    assert "params" in p and "actor" in p["params"], f"'actor' not found in {params_path}"
    actor_params = freeze(p["params"]["actor"])
    return freeze({"params": actor_params})


# ==========================================================
# RL_VAE latent z utilities (reuse your prior format)
# ==========================================================


def z_from_prior_mean(prior_json_path: str):
    with open(prior_json_path, "r") as f:
        prior = json.load(f)
    weights = jnp.array(prior["weights"], dtype=jnp.float32)
    weights = weights / weights.sum()
    means = jnp.array(prior["means"], dtype=jnp.float32)  # (K,L)
    return jnp.tensordot(weights, means, axes=1)


def sample_z_from_prior(rng, prior_json_path: str):
    with open(prior_json_path, "r") as f:
        prior = json.load(f)
    weights = jnp.array(prior["weights"], dtype=jnp.float32)
    weights = weights / weights.sum()
    means = jnp.array(prior["means"], dtype=jnp.float32)  # (K,L)
    Ls = jnp.array(prior["cov_cholesky"], dtype=jnp.float32)  # (K,L,L)
    k = jax.random.categorical(rng, jnp.log(weights + 1e-8))
    mu = means[k]
    L = Ls[k]
    eps = jax.random.normal(rng, (mu.shape[0],))
    return mu + L @ eps


def infer_z_for_rl_vae(
    *,
    z_npy: str | None,
    prior_json: str | None,
    z_seed: int | None,
    use_prior_mean: bool,
) -> jnp.ndarray:
    if z_npy is not None and os.path.exists(z_npy):
        return jnp.array(np.load(z_npy)).astype(jnp.float32)
    if use_prior_mean:
        assert prior_json is not None, "use_prior_mean requires --prior_json"
        return z_from_prior_mean(prior_json)
    assert prior_json is not None and z_seed is not None, (
        "Need --prior_json and --z_seed (or --z_npy / --use_prior_mean)"
    )
    return sample_z_from_prior(jax.random.PRNGKey(int(z_seed)), prior_json)


# ==========================================================
# VAE encoder-only (mask-weighted time aggregation)
#   This is the key bit: we reimplement the *encoder path* so we can apply a mask.
# ==========================================================

# If your obs corruption/masking is irrelevant at eval, we skip it entirely here (training=False path).
# Encoder expects obs_seq: (T,B,658) in the same layout the VAE was trained on.


class _MLP(nn.Module):
    features: Tuple[int, ...]
    act: str = "relu"

    @nn.compact
    def __call__(self, x):
        act_fn = dict(relu=nn.relu, gelu=nn.gelu, tanh=nn.tanh)[self.act]
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = act_fn(x)
        return nn.Dense(self.features[-1])(x)


class EncoderOnly(nn.Module):
    preprocessing_features: Tuple[int, ...]
    latent_dim: int
    act_fn: str = "relu"

    @nn.compact
    def __call__(self, obs_seq: jnp.ndarray, mask_tb: jnp.ndarray):
        """
        obs_seq: (T,B,D)
        mask_tb: (T,B) in {0,1} where 1 indicates valid timestep (pre-terminal)
        Returns:
          mu_b: (B,L) mask-weighted over time
          lv_b: (B,L) mask-weighted over time
        """
        T, B, _ = obs_seq.shape
        enc_in = obs_seq.reshape(T * B, -1)
        h = _MLP(tuple(self.preprocessing_features) + (128,), self.act_fn)(enc_in)
        mu_t = nn.Dense(self.latent_dim)(h).reshape(T, B, self.latent_dim)
        lv_t = nn.Dense(self.latent_dim)(h).reshape(T, B, self.latent_dim)

        m = mask_tb[..., None].astype(mu_t.dtype)  # (T,B,1)
        denom = jnp.maximum(m.sum(axis=0), 1.0)  # (B,1)

        mu_b = (mu_t * m).sum(axis=0) / denom
        lv_b = (lv_t * m).sum(axis=0) / denom
        return mu_b, lv_b


def load_vae_params_full(vae_ckpt: str):
    # full params tree for the trained VAE
    return load_flax_safetensors(vae_ckpt)


# ==========================================================
# Rollout generation (vectorized over episodes)
# ==========================================================


def unwrap_state(s):
    return getattr(s, "env_state", s)


def build_env(num_players: int):
    hand_size = 5 if num_players <= 3 else 4
    base_env = make("hanabi", hand_size=hand_size, num_agents=num_players)
    return LogWrapper(base_env)


def build_policy(
    *,
    policy_type: str,  # "rl_bc" or "rl_vae"
    rl_ckpt: str,
    act_fn: str,
    preprocessing_features: List[int],
    lstm_features: List[int],
    postprocessing_features: List[int],
    dropout: float,
    action_dim: int,
    z: jnp.ndarray,  # (L,) for rl_vae, empty for rl_bc
):
    actor_vars = load_actor_vars_for_rl(rl_ckpt)
    actor_params = actor_vars["params"]

    if actor_uses_policy_out(actor_params):
        actor = ActorLstmPolicyOut(
            preprocessing_features=preprocessing_features,
            lstm_features=lstm_features,
            postprocessing_features=postprocessing_features,
            action_dim=action_dim,
            dropout_rate=dropout,
            activation_fn_name=act_fn,
        )
    else:
        actor = ActorLstm(
            preprocessing_features=preprocessing_features,
            lstm_features=lstm_features,
            postprocessing_features=postprocessing_features,
            action_dim=action_dim,
            dropout_rate=dropout,
            activation_fn_name=act_fn,
        )

    # Normalize vars
    actor_vars = freeze({"params": unfreeze(actor_vars)["params"]})

    return actor, actor_vars, z


def rollout_one_episode(
    rng: jnp.ndarray,
    env,
    actor: nn.Module,
    actor_vars,
    z_vec: jnp.ndarray,
    max_steps: int,
):
    """
    Run one greedy episode in self-play (same actor for both seats).

    Returns:
      obs_seq: (T,B,D)
      act_seq: (T,B,1) int32
      mask_tb: (T,B) float32 (valid timesteps)
    """
    rng, r0 = jax.random.split(rng)
    obs, env_state = env.reset(r0)
    B = len(env.agents)
    first_agent = env.agents[0]
    _D = int(obs[first_agent].shape[-1])

    carry0 = actor.initialize_carry(B)

    def step_fn(carry, _t):
        rng, env_state, obs_dict, carry_actor, done = carry

        # build tensors
        obs_b = jnp.stack([obs_dict[a] for a in env.agents], axis=0)  # (B,D)
        obs_tb = obs_b[None, ...]  # (1,B,D)

        # resets/dones
        resets_tb = jnp.full((1, B), done, dtype=jnp.bool_)  # (1,B)

        # legal moves
        legal = env.get_legal_moves(unwrap_state(env_state))
        avail_b = jnp.stack([legal[a] for a in env.agents], axis=0)  # (B,A)
        avail_tb = avail_b[None, ...]  # (1,B,A)

        # z batch
        if z_vec.size == 0:
            z_b = None
        else:
            z_b = jnp.broadcast_to(z_vec[None, :], (B, z_vec.shape[0]))  # (B,L)

        # actor forward (T=1)
        carry_actor2, logits_tb = actor.apply(
            actor_vars,
            carry_actor,
            (obs_tb, resets_tb, avail_tb),
            z=z_b,
            training=False,
        )

        # greedy action
        a_tb = jnp.argmax(logits_tb, axis=-1).astype(jnp.int32)  # (1,B)
        a_b = a_tb.squeeze(0)  # (B,)

        # map to dict
        act_dict = {agent_name: a_b[i] for i, agent_name in enumerate(env.agents)}

        # step env
        rng, r1 = jax.random.split(rng)
        new_obs, new_env_state, reward, dones, infos = env.step(r1, env_state, act_dict)
        new_done = dones["__all__"]

        # mask: timestep is valid iff we were not already done entering the step
        m = 1.0 - done.astype(jnp.float32)  # scalar
        mask_tb = jnp.full((1, B), m, dtype=jnp.float32)  # (1,B)

        # action sequence expects (T,B,1)
        act_t_b1 = a_tb[..., None]  # (1,B,1)
        obs_t_bd = obs_tb  # (1,B,D)

        new_carry = (rng, new_env_state, new_obs, carry_actor2, new_done)
        outputs = (obs_t_bd, act_t_b1, mask_tb)
        return new_carry, outputs

    carry_init = (rng, env_state, obs, carry0, jnp.array(False))
    (_, _, _, _, _), (obs_seq, act_seq, mask_tb) = jax.lax.scan(
        step_fn, carry_init, xs=None, length=max_steps
    )

    # squeeze time already present: obs_seq is (T,1,B,D)? no: we returned (1,B,D) each step
    # scan stacks over time -> (T,1,B,D). So drop the leading singleton time-in-step dimension.
    obs_seq = obs_seq.squeeze(1)  # (T,B,D)
    act_seq = act_seq.squeeze(1)  # (T,B,1)
    mask_tb = mask_tb.squeeze(1)  # (T,B)
    return obs_seq, act_seq, mask_tb


# ==========================================================
# PCA (pure numpy)
# ==========================================================


def pca_2d(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    x: (N,L)
    returns:
      coords: (N,2)
      info: dict with mean, components, singular_values
    """
    x_mean = x.mean(axis=0, keepdims=True)
    xc = x - x_mean
    # SVD on centered data
    U, S, Vt = np.linalg.svd(xc, full_matrices=False)
    comps = Vt[:2].T  # (L,2)
    coords = xc @ comps
    info = {
        "mean": x_mean.squeeze(0),
        "components": comps,
        "singular_values": S[:2],
    }
    return coords, info


# ==========================================================
# Main
# ==========================================================


def main():
    ap = argparse.ArgumentParser()

    # --- policy ---
    ap.add_argument("--policy_type", choices=["rl_bc", "rl_vae"], required=True)
    ap.add_argument("--rl_ckpt", type=str, required=True)
    ap.add_argument("--act_fn", choices=["relu", "gelu"], default="gelu")

    # arch (must match RL checkpoint)
    ap.add_argument("--preprocessing_features", type=int, nargs="+", default=[1024])
    ap.add_argument("--lstm_features", type=int, nargs="+", default=[512])
    ap.add_argument("--postprocessing_features", type=int, nargs="+", default=[256])
    ap.add_argument("--dropout", type=float, default=0.0)

    # rl_vae z
    ap.add_argument("--z_npy", type=str, default=None)
    ap.add_argument("--prior_json", type=str, default=None)
    ap.add_argument("--z_seed", type=int, default=None)
    ap.add_argument("--use_prior_mean", action="store_true", default=False)

    # --- vae encoder ---
    ap.add_argument("--vae_ckpt", type=str, required=True)
    ap.add_argument("--vae_latent_dim", type=int, required=True)
    ap.add_argument("--vae_act_fn", choices=["relu", "gelu", "tanh"], default="relu")
    ap.add_argument("--vae_preprocessing_features", type=int, nargs="+", default=[1024])

    # --- rollout ---
    ap.add_argument("--num_players", type=int, default=2)
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)

    # --- outputs ---
    ap.add_argument(
        "--out_npy", type=str, required=True, help="Path to save (N,2) PCA coords .npy"
    )
    ap.add_argument(
        "--out_latents_npy",
        type=str,
        default=None,
        help="Optional path to save (N,L) latents .npy",
    )
    ap.add_argument(
        "--out_pca_info_npz",
        type=str,
        default=None,
        help="Optional .npz with PCA mean/components/S",
    )

    args = ap.parse_args()

    # env
    env = build_env(args.num_players)

    # infer action dim
    rng0 = jax.random.PRNGKey(args.seed)
    obs0, es0 = env.reset(rng0)
    legal0 = env.get_legal_moves(unwrap_state(es0))
    first_agent = env.agents[0]
    action_dim = int(legal0[first_agent].shape[-1])

    # z for policy
    if args.policy_type == "rl_vae":
        z = infer_z_for_rl_vae(
            z_npy=args.z_npy,
            prior_json=args.prior_json,
            z_seed=args.z_seed,
            use_prior_mean=args.use_prior_mean,
        )
    else:
        z = jnp.zeros((0,), dtype=jnp.float32)

    # build policy
    actor, actor_vars, z_vec = build_policy(
        policy_type=args.policy_type,
        rl_ckpt=args.rl_ckpt,
        act_fn=args.act_fn,
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        dropout=args.dropout,
        action_dim=action_dim,
        z=z,
    )

    # load VAE params + build encoder-only with matching param names
    vae_vars = load_vae_params_full(args.vae_ckpt)

    encoder = EncoderOnly(
        preprocessing_features=tuple(args.vae_preprocessing_features),
        latent_dim=int(args.vae_latent_dim),
        act_fn=args.vae_act_fn,
    )

    # NOTE: we need to pass the *correct encoder subtree params*.
    # Your saved VAE checkpoint likely has encoder params at top-level inside params.
    # We just pass {"params": <full params>} and rely on matching names (EncoderOnly matches the encoder pieces).
    # If your checkpoint nests the model under e.g. "VAE_0", you may need to unwrap.
    enc_params = unfreeze(vae_vars["params"])
    # common unwrap: sometimes saved under module name; add more cases if needed
    enc_params = unwrap_vae_params(enc_params)
    enc_vars = freeze({"params": enc_params})

    # jit rollout+encode for speed
    def rollout_and_encode(rkey):
        obs_seq, act_seq, mask_tb = rollout_one_episode(
            rkey, env, actor, actor_vars, z_vec, args.max_steps
        )
        # encode per-seat
        mu_b, lv_b = encoder.apply(enc_vars, obs_seq, mask_tb)  # (B,L), (B,L)
        # policy/episode latent: average across seats
        mu_ep = mu_b.mean(axis=0)  # (L,)
        return mu_ep

    rollout_and_encode_jit = jax.jit(rollout_and_encode)
    rollout_and_encode_vmap = jax.jit(jax.vmap(rollout_and_encode_jit))

    keys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    latents = rollout_and_encode_vmap(keys)  # (N,L)
    latents_np = np.array(latents)

    coords, info = pca_2d(latents_np)  # (N,2)

    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    np.save(args.out_npy, coords)

    if args.out_latents_npy is not None:
        os.makedirs(os.path.dirname(args.out_latents_npy) or ".", exist_ok=True)
        np.save(args.out_latents_npy, latents_np)

    if args.out_pca_info_npz is not None:
        os.makedirs(os.path.dirname(args.out_pca_info_npz) or ".", exist_ok=True)
        np.savez(
            args.out_pca_info_npz,
            mean=info["mean"],
            components=info["components"],
            singular_values=info["singular_values"],
        )

    print(f"Saved PCA coords: {args.out_npy}  shape={coords.shape}")
    if args.out_latents_npy:
        print(f"Saved raw latents: {args.out_latents_npy}  shape={latents_np.shape}")
    if args.out_pca_info_npz:
        print(f"Saved PCA info: {args.out_pca_info_npz}")


if __name__ == "__main__":
    main()
