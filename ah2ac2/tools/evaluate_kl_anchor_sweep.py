#!/usr/bin/env python3
"""
evaluate_kl_anchor_sweep.py

Evaluate external fidelity across reference and reconstruction sweeps.
Run as a module with --help for input paths and supported generators.
"""

import argparse

# ------------------------------------------------
# (new) tiny OBL helpers — kept local and minimal
# ------------------------------------------------
import functools
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper
from safetensors.numpy import load_file as st_load

from ah2ac2.tools import crossplay_eval, kl_eval
from ah2ac2.tools.eval_bc_on_val import evaluate_bc_on_val
from ah2ac2.tools.eval_ppo_on_val import evaluate_ppo_on_val
from ah2ac2.tools.kl_eval import load_actor_vars_for_new_obl


def _unflatten_dict_obl(flat: Dict[str, np.ndarray], sep: str = ",") -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    for k, v in flat.items():
        parts = k.split(sep)
        d = root
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = v
    return root


class _OBLMultiLayerLSTM(nn.Module):
    num_layers: int
    features: int

    @nn.compact
    def __call__(self, carry, x):
        c, h = carry
        new_cs = []
        new_hs = []
        y = x
        for i in range(self.num_layers):
            lstm = nn.LSTMCell(self.features, name=f"l{i}")
            (nc, nh), y = lstm((c[i], h[i]), y)
            new_cs.append(nc)
            new_hs.append(nh)
        return (jnp.stack(new_cs, 0), jnp.stack(new_hs, 0)), y

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, batch_dims: Tuple[int, ...]):
        shape = (self.num_layers,) + batch_dims + (self.features,)
        return jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.float32)


class _OBLR2D2(nn.Module):
    hid_dim: int = 512
    out_dim: int = 21
    num_lstm_layer: int = 2

    @nn.compact
    def __call__(self, carry, inputs):
        priv_s, publ_s = inputs

        priv_o = nn.Dense(self.hid_dim, name="priv_net_dense_0")(priv_s)
        priv_o = nn.relu(priv_o)
        priv_o = nn.Dense(self.hid_dim, name="priv_net_dense_1")(priv_o)
        priv_o = nn.relu(priv_o)
        priv_o = nn.Dense(self.hid_dim, name="priv_net_dense_2")(priv_o)
        priv_o = nn.relu(priv_o)

        x = nn.Dense(self.hid_dim, name="publ_net_dense_0")(publ_s)
        x = nn.relu(x)
        carry, publ_o = _OBLMultiLayerLSTM(
            num_layers=self.num_lstm_layer,
            features=self.hid_dim,
            name="lstm",
        )(carry, x)

        o = priv_o * publ_o
        logits = nn.Dense(self.out_dim, name="fc_a")(o)
        return carry, logits

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, batch_dims: Tuple[int, ...]):
        return _OBLMultiLayerLSTM(
            num_layers=self.num_lstm_layer,
            features=self.hid_dim,
            name="lstm",
        ).initialize_carry(rng, batch_dims)


def _load_obl_params(path: str) -> Dict[str, Any]:
    flat = st_load(path)
    return _unflatten_dict_obl(flat, sep=",")


def _make_hanabi_env(num_players: int, use_raw_obs: bool, seed: int = 0):
    hand_size = 5 if num_players <= 3 else 4
    base_env = make("hanabi", hand_size=hand_size, num_agents=num_players)
    return base_env if use_raw_obs else LogWrapper(base_env)


# ---------------------------
# new_obl helpers (ActorCritic-style LSTM)
# ---------------------------

NEW_OBL_CONFIG: Dict[str, Any] = {
    "ENV_NAME": "hanabi",
    "ENV_KWARGS": {"num_agents": 2},
    "RNN": "LSTM",
    "NUM_RNN_LAYERS": 2,
    "NUM_FF_LAYERS": 0,
    "LAYER_WIDTH": 512,
    "PUBLIC_RNN": True,
    "CRITIC_CENTRALIZED": False,
    "ACTOR_FF_STREAM": True,
    "CRITIC_FF_STREAM": True,
    "MULTIPLY_PUBLIC_PRIVATE": True,
}


class ScannedMultiLayerLSTM(nn.Module):
    num_layers: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry_LBW_, x_B_):
        ins_BW, resets_B = x_B_

        # reset hidden state where resets_B is True
        carry_LBW_ = tuple(
            jnp.where(
                resets_B[None, :, None],
                self.initialize_hidden_state(self.num_layers, ins_BW.shape[0], ins_BW.shape[1])[i],
                carry_LBW_[i],
            )
            for i in range(len(carry_LBW_))
        )

        cs_LBW, hs_LBW = carry_LBW_
        new_cs_LBW, new_hs_LBW = jnp.zeros_like(cs_LBW), jnp.zeros_like(hs_LBW)
        y_BW = ins_BW

        for layer in range(self.num_layers):
            (new_c_BW, new_h_BW), y_BW = nn.LSTMCell(features=ins_BW.shape[1])(
                (cs_LBW[layer], hs_LBW[layer]), y_BW
            )
            new_cs_LBW = new_cs_LBW.at[layer].set(new_c_BW)
            new_hs_LBW = new_hs_LBW.at[layer].set(new_h_BW)

        return (new_cs_LBW, new_hs_LBW), y_BW

    @staticmethod
    def initialize_hidden_state(num_rnn_layers, batch_size, hidden_size):
        single = jnp.zeros((num_rnn_layers, batch_size, hidden_size))
        return (single, single)


class ActorCritic(nn.Module):
    action_dim: int
    config: Dict[str, Any]

    def setup(self):
        assert self.config["RNN"] in [False, "LSTM", "GRU"]
        NUM_PLAYERS = self.config["ENV_KWARGS"].get("num_agents", 2)
        NUM_COLORS = self.config["ENV_KWARGS"].get("num_colors", 5)
        NUM_RANKS = self.config["ENV_KWARGS"].get("num_ranks", 5)
        HAND_SIZE = 5 if NUM_PLAYERS < 4 else 4
        self.OBS_PRIVATE_SIZE = (NUM_PLAYERS - 1) * HAND_SIZE * NUM_COLORS * NUM_RANKS

    @nn.compact
    def __call__(self, hidden_state_LBW_, x_TB_):
        obs_TBO, hand_TBY, dones_TB, avail_actions_TBA = x_TB_

        if self.config["PUBLIC_RNN"]:
            obs_rnn_TB_ = obs_TBO[..., self.OBS_PRIVATE_SIZE :]
        else:
            obs_rnn_TB_ = obs_TBO

        obs_actor_TBO = obs_TBO
        obs_critic_TB_ = obs_TBO  # CRITIC_CENTRALIZED = False

        embedding_obs_actor_TBW = nn.relu(
            nn.Dense(
                self.config["LAYER_WIDTH"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(obs_actor_TBO)
        )
        embedding_obs_critic_TBW = nn.relu(
            nn.Dense(
                self.config["LAYER_WIDTH"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(obs_critic_TB_)
        )
        for _ in range(self.config["NUM_RNN_LAYERS"]):
            embedding_obs_actor_TBW = nn.relu(
                nn.Dense(
                    self.config["LAYER_WIDTH"],
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(embedding_obs_actor_TBW)
            )
            embedding_obs_critic_TBW = nn.relu(
                nn.Dense(
                    self.config["LAYER_WIDTH"],
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(embedding_obs_critic_TBW)
            )

        embedding_obs_rnn_TBW = nn.relu(
            nn.Dense(
                self.config["LAYER_WIDTH"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(obs_rnn_TB_)
        )
        rnn_in_TB_ = (embedding_obs_rnn_TBW, dones_TB)
        hidden_state_LBW_, embedding_obs_rnn_TBW = ScannedMultiLayerLSTM(
            self.config["NUM_RNN_LAYERS"]
        )(hidden_state_LBW_, rnn_in_TB_)

        actor_mean_TBW = embedding_obs_rnn_TBW
        critic_mean_TBW = embedding_obs_rnn_TBW

        if self.config["ACTOR_FF_STREAM"]:
            if self.config["MULTIPLY_PUBLIC_PRIVATE"]:
                actor_mean_TBW = embedding_obs_actor_TBW * actor_mean_TBW
            else:
                actor_mean_TBW = jnp.concatenate(
                    [embedding_obs_actor_TBW, actor_mean_TBW], axis=-1
                )
        if self.config["CRITIC_FF_STREAM"]:
            if self.config["MULTIPLY_PUBLIC_PRIVATE"]:
                critic_mean_TBW = embedding_obs_critic_TBW * critic_mean_TBW
            else:
                critic_mean_TBW = jnp.concatenate(
                    [embedding_obs_critic_TBW, critic_mean_TBW], axis=-1
                )

        for _ in range(self.config["NUM_FF_LAYERS"]):
            actor_mean_TBW = nn.relu(
                nn.Dense(
                    self.config["LAYER_WIDTH"], kernel_init=orthogonal(2), bias_init=constant(0.0)
                )(actor_mean_TBW)
            )
            critic_mean_TBW = nn.relu(
                nn.Dense(
                    self.config["LAYER_WIDTH"], kernel_init=orthogonal(2), bias_init=constant(0.0)
                )(critic_mean_TBW)
            )

        action_logits_TBA = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean_TBW)
        action_logits_TBA = action_logits_TBA - 1e10 * (1 - avail_actions_TBA)

        critic_TB1 = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(
            critic_mean_TBW
        )
        critic_TB = critic_TB1.squeeze(-1)

        return hidden_state_LBW_, action_logits_TBA, critic_TB

    @nn.nowrap
    def initialize_hidden_state(self, batch_size: int):
        return ScannedMultiLayerLSTM.initialize_hidden_state(
            NEW_OBL_CONFIG["NUM_RNN_LAYERS"], batch_size, NEW_OBL_CONFIG["LAYER_WIDTH"]
        )


def _load_new_obl_params(path: str) -> Dict[str, Any]:
    return load_actor_vars_for_new_obl(path)


# ---------------------------
# Small helpers (original)
# ---------------------------


def make_hanabi_env(num_players: int, use_raw_obs: bool, seed: int = 0):
    return _make_hanabi_env(num_players, use_raw_obs, seed)


def parse_bc_score_from_name(fname: str) -> float:
    m = re.search(r"score([0-9]+(?:\.[0-9]+)?)", fname)
    if not m:
        return None
    return float(m.group(1))


def find_single_bc_file(bc_dir: str) -> str:
    cands = [f for f in os.listdir(bc_dir) if f.endswith(".safetensors") and "score" in f]
    if len(cands) == 0:
        raise FileNotFoundError(f"No BC .safetensors with 'score' in {bc_dir}")
    if len(cands) > 1:
        cands = sorted(cands)
    return os.path.join(bc_dir, cands[0])


def iter_rl_checkpoints_under(rl_root: str):
    for sub in os.listdir(rl_root):
        if not sub.startswith("kl_target"):
            continue
        subdir = os.path.join(rl_root, sub)
        if not os.path.isdir(subdir):
            continue
        ckpts = [
            f for f in os.listdir(subdir) if f.startswith("seed1_") and f.endswith(".safetensors")
        ]
        if not ckpts:
            continue
        ckpt_path = os.path.join(subdir, ckpts[0])
        yield sub, ckpt_path


def _act_for_policy_type(policy_type: str) -> str:
    """Map policy type -> activation function used at training time."""
    if policy_type in ("bc", "rl_bc"):
        return "gelu"
    if policy_type in ("ippo",):
        return "relu"
    if policy_type in ("obl", "new_obl"):
        # both OBL variants use ReLU everywhere
        return "relu"
    # default
    return "relu"


def kl_to_generator(
    gen_type: str,
    gen_path: str,
    eval_type: str,
    eval_path: str,
    num_players: int = 2,
    num_rollouts: int = 128,
    seed: int = 0,
):
    # figure out activations
    gen_act = _act_for_policy_type(gen_type)
    eval_act = _act_for_policy_type(eval_type)

    args = SimpleNamespace(
        # env
        num_players=num_players,
        num_rollouts=num_rollouts,
        seed=seed,
        use_jit=1,
        # nets (global defaults; per-policy overrides below)
        preprocessing_features=[1024],
        lstm_features=[512],
        postprocessing_features=[256],
        dropout=0.0,
        act_fn="relu",
        gen_act_fn=gen_act,
        eval_act_fn=eval_act,
        # prior / z
        prior_json=None,
        # generator fields
        gen_type=gen_type,
        gen_params=None,
        gen_vae_ckpt=None,
        gen_bc_params=None,
        gen_ippo_ckpt=None,
        gen_yaml_with_z=None,
        gen_z_npy=None,
        gen_z_seed=None,
        gen_use_prior_mean=False,
        gen_obl_params=None,
        gen_new_obl_params=None,
        gen_mode="sample",
        # eval fields
        eval_type=eval_type,
        eval_params=None,
        eval_vae_ckpt=None,
        eval_bc_params=None,
        eval_ippo_ckpt=None,
        eval_yaml_with_z=None,
        eval_z_npy=None,
        eval_z_seed=None,
        eval_use_prior_mean=False,
        eval_obl_params=None,
        eval_new_obl_params=None,
    )

    # wire the paths
    if gen_type == "bc":
        args.gen_bc_params = gen_path
    elif gen_type == "ippo":
        args.gen_ippo_ckpt = gen_path
    elif gen_type in ("rl", "rl_bc"):
        args.gen_params = gen_path
    elif gen_type == "obl":
        args.gen_obl_params = gen_path
    elif gen_type == "new_obl":
        args.gen_new_obl_params = gen_path
    else:
        raise ValueError(f"Unsupported gen_type {gen_type}")

    if eval_type == "bc":
        args.eval_bc_params = eval_path
    elif eval_type == "ippo":
        args.eval_ippo_ckpt = eval_path
    elif eval_type in ("rl", "rl_bc"):
        args.eval_params = eval_path
    else:
        raise ValueError(f"Unsupported eval_type {eval_type}")

    use_raw_obs = any(t in ("bc", "obl", "new_obl") for t in (args.gen_type, args.eval_type))
    env = make_hanabi_env(num_players, use_raw_obs, seed=seed)

    rng = jax.random.PRNGKey(seed)
    rng_keys = jax.random.split(rng, num_rollouts)
    (
        kl_vals,
        steps,
        mean_total,
        sem_total,
        mean_per_step,
        sem_per_step,
    ) = kl_eval.run_many(rng_keys, args, env)
    return float(mean_total)


# ---------------------------
# (new) OBL self-play
# ---------------------------


def _obl_selfplay(obl_path: str, num_players: int, num_rollouts: int, seed: int = 0):
    """
    Roll out the OBL policy exactly the way your rollout_obl_to_safetensors.py does:
    - use the obs returned by env.reset / env.step
    - build batched obs from the dict
    - mask logits with legal moves
    - act greedily
    """
    # load OBL params (the same unflatten we defined earlier in this file)
    params = _load_obl_params(obl_path)

    # raw hanabi env, like your script
    env = make("hanabi", num_agents=num_players)
    agents = list(env.agents)

    # get action dim from a fresh reset
    rng0 = jax.random.PRNGKey(seed)
    obs0, state0 = env.reset(rng0)
    legals0 = env.get_legal_moves(state0)
    action_dim = int(np.asarray(legals0[agents[0]]).shape[-1])

    # build OBL model (same structure we defined above)
    model = _OBLR2D2(out_dim=action_dim)

    # figure out obs dims from env, like your script
    dummy_obs_b = jnp.stack([obs0[a] for a in agents], 0)  # (B, D)
    publ_offset = 125
    assert dummy_obs_b.shape[-1] > publ_offset, (
        f"Obs dim {dummy_obs_b.shape[-1]} <= publ offset {publ_offset}"
    )
    dummy_priv = dummy_obs_b
    dummy_publ = dummy_obs_b[:, publ_offset:]

    # init once for shapes
    dummy_carry = model.initialize_carry(jax.random.PRNGKey(0), (num_players,))
    _ = model.init(jax.random.PRNGKey(1), dummy_carry, (dummy_priv, dummy_publ))

    def _one(key):
        # fresh episode
        key, k_reset = jax.random.split(key)
        obs, state = env.reset(k_reset)

        # OBL hidden state
        carry = model.initialize_carry(jax.random.PRNGKey(0), (num_players,))

        score = jnp.array(0.0, dtype=jnp.float32)
        done = jnp.array(False)

        def cond_fn(vals):
            # vals = (key, state, obs, done, score, carry)
            return jnp.logical_not(vals[3])

        def body_fn(vals):
            key, state, obs, done, score, carry = vals

            # turn obs dict -> (B, D)
            obs_b = jnp.stack([obs[a] for a in agents], 0)
            priv_s = obs_b
            publ_s = obs_b[:, publ_offset:]

            # forward OBL
            carry, logits = model.apply(params, carry, (priv_s, publ_s))

            # legal moves
            legals = env.get_legal_moves(state)
            legals_b = jnp.stack([legals[a] for a in agents], 0)

            # mask and act greedily
            logits = logits - (1.0 - legals_b) * 1e10
            acts_b = jnp.argmax(logits, axis=-1)

            act_dict = {agents[i]: acts_b[i] for i in range(num_players)}

            key, k_step = jax.random.split(key)
            obs2, state2, reward, dones, infos = env.step(k_step, state, act_dict)

            done2 = dones["__all__"]
            score2 = score + reward["__all__"]

            return (key, state2, obs2, done2, score2, carry)

        key, state, obs, done, score, carry = jax.lax.while_loop(
            cond_fn, body_fn, (key, state, obs, done, score, carry)
        )
        return score

    # vmapped eval
    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, num_rollouts)
    scores = jax.vmap(_one)(keys)
    return float(jnp.mean(scores))


def _new_obl_selfplay(obl_path: str, num_players: int, num_rollouts: int, seed: int = 0):
    params = _load_new_obl_params(obl_path)

    env = make("hanabi", num_agents=num_players)
    agents = list(env.agents)

    rng0 = jax.random.PRNGKey(seed)
    obs0, state0 = env.reset(rng0)
    legals0 = env.get_legal_moves(state0)
    action_dim = int(np.asarray(legals0[agents[0]]).shape[-1])

    model = ActorCritic(action_dim=action_dim, config=NEW_OBL_CONFIG)

    def _one(key):
        key, k_reset = jax.random.split(key)
        obs, state = env.reset(k_reset)

        carry = model.initialize_hidden_state(num_players)
        score = jnp.array(0.0, dtype=jnp.float32)
        done = jnp.array(False)

        def cond_fn(vals):
            return jnp.logical_not(vals[3])  # not done

        def body_fn(vals):
            key, state, obs, done, score, carry = vals

            obs_b = jnp.stack([obs[a] for a in agents], 0)  # (B, D)
            obs_TBO = obs_b[None, ...]
            hand_TBY = jnp.zeros((1, num_players, 1))
            dones_TB = jnp.zeros((1, num_players), dtype=jnp.bool_)

            legals = env.get_legal_moves(state)
            legals_b = jnp.stack([legals[a] for a in agents], 0)
            avail_TBA = legals_b[None, ...]

            carry, logits_TBA, _ = model.apply(
                params, carry, (obs_TBO, hand_TBY, dones_TB, avail_TBA)
            )
            logits_ba = logits_TBA[0]
            acts_b = jnp.argmax(logits_ba, axis=-1)

            act_dict = {agents[i]: acts_b[i] for i in range(num_players)}

            key, k_step = jax.random.split(key)
            obs2, state2, reward, dones, infos = env.step(k_step, state, act_dict)

            done2 = dones["__all__"]
            score2 = score + reward["__all__"]

            return (key, state2, obs2, done2, score2, carry)

        key, state, obs, done, score, carry = jax.lax.while_loop(
            cond_fn, body_fn, (key, state, obs, done, score, carry)
        )
        return score

    rng = jax.random.PRNGKey(seed)
    keys = jax.random.split(rng, num_rollouts)
    scores = jax.vmap(_one)(keys)
    return float(jnp.mean(scores))


def generator_selfplay(
    gen_type: str,
    gen_path: str,
    num_players: int = 2,
    num_rollouts: int = 128,
    seed: int = 0,
):
    # handle OBL and new_OBL explicitly
    if gen_type == "obl":
        return _obl_selfplay(gen_path, num_players, num_rollouts, seed)
    if gen_type == "new_obl":
        return _new_obl_selfplay(gen_path, num_players, num_rollouts, seed)

    # --- everything below is your original IPPO/BC/RL selfplay code ---
    pol_act = _act_for_policy_type(gen_type)

    args = SimpleNamespace(
        # env
        num_players=num_players,
        num_rollouts=num_rollouts,
        seed=seed,
        use_jit=1,
        # nets
        preprocessing_features=[1024],
        lstm_features=[512],
        postprocessing_features=[256],
        dropout=0.0,
        act_fn=pol_act,
        # policy 1
        pol1_type=gen_type,
        pol1_params=None,
        pol1_vae_ckpt=None,
        pol1_bc_params=None,
        pol1_ippo_ckpt=None,
        pol1_yaml_with_z=None,
        pol1_z_npy=None,
        pol1_z_seed=None,
        pol1_use_prior_mean=False,
        pol1_act_fn=pol_act,
        # policy 2 (same)
        pol2_type=gen_type,
        pol2_params=None,
        pol2_vae_ckpt=None,
        pol2_bc_params=None,
        pol2_ippo_ckpt=None,
        pol2_yaml_with_z=None,
        pol2_z_npy=None,
        pol2_z_seed=None,
        pol2_use_prior_mean=False,
        pol2_act_fn=pol_act,
        # eval behaviour
        eval_mode="greedy",
        temperature=1.0,
    )

    _pol1_preprocessing_features = (None,)
    _pol1_lstm_features = (None,)
    _pol1_postprocessing_features = (None,)
    _pol2_preprocessing_features = (None,)
    _pol2_lstm_features = (None,)
    _pol2_postprocessing_features = (None,)

    if gen_type == "bc":
        args.pol1_bc_params = gen_path
        args.pol2_bc_params = gen_path
    elif gen_type == "ippo":
        args.pol1_ippo_ckpt = gen_path
        args.pol2_ippo_ckpt = gen_path
    elif gen_type in ("rl", "rl_bc"):
        args.pol1_params = gen_path
        args.pol2_params = gen_path
    else:
        raise ValueError(f"Unsupported generator type {gen_type}")

    use_raw_obs = any(t in ("bc",) for t in (args.pol1_type, args.pol2_type))
    env = make_hanabi_env(num_players, use_raw_obs, seed=seed)

    rng0 = jax.random.PRNGKey(seed)
    _rng = jax.random.split(rng0, 1)[0]
    obs0, env_state0 = env.reset(_rng)
    legal0 = env.get_legal_moves(crossplay_eval.unwrap_state(env_state0))
    first_agent = list(env.agents)[0]
    args.action_dim = int(legal0[first_agent].shape[-1])

    rng = jax.random.PRNGKey(seed)
    rng_keys = jax.random.split(rng, args.num_rollouts)
    mean, sem = crossplay_eval.play_games(rng_keys, args, env)
    return float(mean)


def _maybe_split_commas(lst):
    """If user passed a single comma-separated item, split it."""
    if len(lst) == 1 and ("," in lst[0]):
        return [x for x in lst[0].split(",") if x]
    return lst


# ---------------------------
# Main driver
# ---------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--generator_type",
        required=True,
        choices=["rl", "vae", "bc", "ippo", "rl_bc", "obl", "new_obl"],
    )
    p.add_argument("--generator_path", required=True)
    p.add_argument("--rl_roots", nargs="+", required=True)
    p.add_argument("--bc_roots", nargs="+", required=True)
    p.add_argument("--num_rollouts", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_npy", required=True)
    p.add_argument(
        "--val_data_path",
        type=str,
        required=True,
        help="Path to validation safetensors dataset used to evaluate BC val accuracy.",
    )
    p.add_argument(
        "--selection_metric",
        type=str,
        default="kl",
        choices=["kl", "cross_entropy"],
        help=(
            "Metric to select best kl_target: "
            "'kl' (lower is better; uses KL to generator) or "
            "'cross_entropy' (lower is better; uses validation cross-entropy)."
        ),
    )

    args = p.parse_args()

    # make comma-friendly
    args.rl_roots = _maybe_split_commas(args.rl_roots)
    args.bc_roots = _maybe_split_commas(args.bc_roots)

    if len(args.rl_roots) != len(args.bc_roots):
        raise ValueError("rl_roots and bc_roots must have the same length (1-1 correspondence).")

    # 1) generator SP
    gen_sp = generator_selfplay(
        args.generator_type,
        args.generator_path,
        num_players=2,
        num_rollouts=args.num_rollouts,
        seed=args.seed,
    )
    print(f"[info] generator self-play score: {gen_sp:.3f}")

    datapoints = []

    # 2) per pair
    for rl_root, bc_root in zip(args.rl_roots, args.bc_roots):
        print(f"[info] processing RL root={rl_root} with BC root={bc_root}")

        # Allow bc_root to be either a directory or a concrete .safetensors file
        if os.path.isdir(bc_root):
            bc_file = find_single_bc_file(bc_root)
        elif bc_root.endswith(".safetensors") and os.path.isfile(bc_root):
            bc_file = bc_root
        else:
            raise ValueError(
                f"bc_root must be either a directory or a .safetensors file, got: {bc_root}"
            )

        bc_sp = parse_bc_score_from_name(os.path.basename(bc_file))
        if bc_sp is not None:
            print(f"      BC file: {bc_file}  (SP={bc_sp:.3f})")

        # --- BC metrics on generator vs val set ---
        if args.selection_metric == "kl":
            # We care about KL(BC || generator)
            bc_kl = kl_to_generator(
                gen_type=args.generator_type,
                gen_path=args.generator_path,
                eval_type="bc",
                eval_path=bc_file,
                num_players=2,
                num_rollouts=args.num_rollouts,
                seed=args.seed,
            )
        else:
            # In cross-entropy mode we don't compute any KLs
            bc_kl = None

        bc_val_metrics = evaluate_bc_on_val(
            bc_params_path=bc_file,
            val_data_path=args.val_data_path,
            config_path="ah2ac2/training/config/bc_2p.yaml",
            seed=args.seed,
        )
        bc_val_nonnoop = float(bc_val_metrics["non_noop_accuracy"])
        bc_val_xent = float(bc_val_metrics["cross_entropy"])

        # --- logging ---
        if args.selection_metric == "kl":
            if args.generator_type == "obl":
                print(
                    f"      BC: disagreement={bc_kl:.4f} | "
                    f"val_nonnoop_acc={bc_val_nonnoop:.4f} | "
                    f"val_xent={bc_val_xent:.4f}"
                )
            else:
                print(
                    f"      BC: KL(BC || gen)={bc_kl:.4f} nats | "
                    f"val_nonnoop_acc={bc_val_nonnoop:.4f} | "
                    f"val_xent={bc_val_xent:.4f}"
                )
        else:  # selection_metric == "cross_entropy"
            print(f"      BC: val_nonnoop_acc={bc_val_nonnoop:.4f} | val_xent={bc_val_xent:.4f}")

        best_x = None
        best_score = None  # score according to selection_metric

        for kl_name, rl_ckpt_path in iter_rl_checkpoints_under(rl_root):
            if args.selection_metric == "kl":
                # --- KL mode: KL(generator || RL) only, no validation metrics for RL ---
                rl_kl = kl_to_generator(
                    gen_type=args.generator_type,
                    gen_path=args.generator_path,
                    eval_type="rl_bc",
                    eval_path=rl_ckpt_path,
                    num_players=2,
                    num_rollouts=args.num_rollouts,
                    seed=args.seed,
                )

                if args.generator_type == "obl":
                    print(f"         {kl_name}: disagreement={rl_kl:.4f}")
                else:
                    print(f"         {kl_name}: KL={rl_kl:.4f} nats")

                # selection by KL (lower is better)
                score = rl_kl
                is_better = (best_score is None) or (score < best_score)

            else:  # selection_metric == "cross_entropy"
                # --- val cross-entropy mode: no KL vs generator ---
                ppo_val_metrics = evaluate_ppo_on_val(
                    ppo_params_path=rl_ckpt_path,
                    val_data_path=args.val_data_path,
                    config_path="ah2ac2/training/config/rl_ppo_bc_anchor_2p.yaml",
                    seed=args.seed,
                )
                ppo_val_nonnoop = float(ppo_val_metrics["non_noop_accuracy"])
                ppo_val_xent = float(ppo_val_metrics["cross_entropy"])

                print(
                    f"         {kl_name}: "
                    f"PPO_val_nonnoop_acc={ppo_val_nonnoop:.4f} | "
                    f"PPO_val_xent={ppo_val_xent:.4f}"
                )

                # selection by cross-entropy (lower is better)
                score = ppo_val_xent
                is_better = (best_score is None) or (score < best_score)

            if is_better:
                best_score = score
                best_x = kl_name

        if best_x is None:
            print(f"      WARNING: no kl_targetX folders found under {rl_root}, skipping")
            continue

        m = re.match(r"kl_target([0-9.]+)", best_x)
        if not m:
            raise ValueError(f"Could not parse X from folder name {best_x}")
        x_star = float(m.group(1))

        # if bc_sp is not None:
        #     y_val = gen_sp - bc_sp
        #     print(f"      -> best X*: {x_star}  |  y = gen_SP - bc_SP = {y_val:.3f}")
        # else:
        #     y_val = bc_val_nonnoop
        #     print(f"      -> best X*: {x_star}  |  y = bc_nonnoop_accuracy = {y_val:.4f}")

        y_val = bc_val_nonnoop
        print(f"      -> best X*: {x_star}  |  y = bc_nonnoop_accuracy = {y_val:.4f}")

        datapoints.append([x_star, y_val])

    datapoints = np.array(datapoints, dtype=np.float32)
    np.save(args.output_npy, datapoints)
    print(f"[done] wrote {datapoints.shape[0]} datapoints to {args.output_npy}")


if __name__ == "__main__":
    main()
