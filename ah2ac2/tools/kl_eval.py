#!/usr/bin/env python3
# kl_eval.py
import argparse
import functools
import json
import os
from typing import Any, Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.linen.initializers import constant, orthogonal
from flax.traverse_util import unflatten_dict
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper
from safetensors.numpy import load_file as st_load

from ah2ac2.nn.multi_layer_lstm import LstmCellWithHiddenStateReset, MultiLayerLstm

"""
e.g.
python ah2ac2/tools/kl_eval.py \
  --num_players 2 --num_rollouts 1024 --seed 0 \
  --gen_type ippo \
  --gen_ippo_ckpt ah2ac2/models/proud-music/op_ippo_ff/checkpoint_1.pkl \
  --eval_type bc \
  --eval_bc_params checkpoints/bc/op_bc_proudmusic_chpt1/1.5k/epoch47_seed0_score16.241_2p.safetensors \
  --gen_act_fn relu \
    --eval_act_fn gelu \
  --gen_mode sample

python ah2ac2/tools/kl_eval.py \
  --num_players 2 --num_rollouts 1024 --seed 0 \
  --gen_type ippo \
  --gen_ippo_ckpt ah2ac2/models/proud-music/op_ippo_ff/checkpoint_1.pkl \
  --eval_type rl_bc \
  --eval_params checkpoints/kl_anchored/op_bc_proudmusic_chpt1_2k_19.158/kl_target0.075/seed1_score23.889_klt0.075_2p.safetensors \
  --gen_act_fn relu \
  --eval_act_fn gelu \
  --gen_mode sample
"""


# -------------------------
# Utilities
# -------------------------
def unwrap_state(s):
    return getattr(s, "env_state", s)


def load_flax_safetensors(path: str):
    from safetensors.numpy import load_file as st_load_inner

    flat = st_load_inner(path)
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


def zeros_lstm_carry(lstm_features: List[int], batch_size: int):
    cs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in lstm_features]
    hs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in lstm_features]
    return jnp.stack(cs), jnp.stack(hs)


def z_from_prior_mean(prior_json_path: str):
    with open(prior_json_path, "r") as f:
        prior = json.load(f)
    weights = jnp.array(prior["weights"], dtype=jnp.float32)
    weights = weights / weights.sum()
    means = jnp.array(prior["means"], dtype=jnp.float32)
    return jnp.tensordot(weights, means, axes=1)


def sample_z_from_prior(rng, prior_json_path: str):
    with open(prior_json_path, "r") as f:
        prior = json.load(f)
    ptype = prior.get("type", "gmm-fullcov")
    if ptype != "gmm-fullcov":
        raise ValueError(f"Unsupported prior type: {ptype}")
    weights = jnp.array(prior["weights"], dtype=jnp.float32)
    weights = weights / weights.sum()
    means = jnp.array(prior["means"], dtype=jnp.float32)
    Ls = jnp.array(prior["cov_cholesky"], dtype=jnp.float32)
    k = jax.random.categorical(rng, jnp.log(weights + 1e-8))
    mu = means[k]
    L = Ls[k]
    eps = jax.random.normal(rng, (mu.shape[0],))
    return mu + L @ eps


# -------------------------
# (new) OBL bits
# -------------------------


def _sample_deck_pairs(key: jax.Array) -> jnp.ndarray:
    mult = [3, 2, 2, 2, 1]
    colors, ranks = [], []
    for c in range(5):
        for r, n in enumerate(mult):
            colors.extend([c] * n)
            ranks.extend([r] * n)
    deck = jnp.stack(
        [jnp.array(colors, dtype=jnp.int32), jnp.array(ranks, dtype=jnp.int32)],
        axis=1,
    )
    perm = jax.random.permutation(key, deck.shape[0])
    return deck[perm]


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


class OBLMultiLayerLSTM(nn.Module):
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


class OBLR2D2(nn.Module):
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
        carry, publ_o = OBLMultiLayerLSTM(
            num_layers=self.num_lstm_layer,
            features=self.hid_dim,
            name="lstm",
        )(carry, x)

        o = priv_o * publ_o
        logits = nn.Dense(self.out_dim, name="fc_a")(o)
        return carry, logits

    @nn.nowrap
    def initialize_carry(self, rng: jax.Array, batch_dims: Tuple[int, ...]):
        return OBLMultiLayerLSTM(
            num_layers=self.num_lstm_layer,
            features=self.hid_dim,
            name="lstm",
        ).initialize_carry(rng, batch_dims)


def load_actor_vars_for_obl(params_path: str) -> Dict[str, Any]:
    flat = st_load(params_path)
    nested = _unflatten_dict_obl(flat, sep=",")
    return freeze(nested)


# -------------------------
# "new_obl" bits – ActorCritic LSTM with PUBLIC_RNN etc.
# -------------------------

# Match the config you gave for new_obl
NEW_OBL_CONFIG: Dict[str, Any] = {
    "ENV_NAME": "hanabi",
    "ENV_KWARGS": {
        "num_agents": 2,  # we only use 2p here
        # num_colors/num_ranks default to 5 inside the module
    },
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
        # private obs are other players' hands
        self.OBS_PRIVATE_SIZE = (NUM_PLAYERS - 1) * HAND_SIZE * NUM_COLORS * NUM_RANKS

    @nn.compact
    def __call__(self, hidden_state_LBW_, x_TB_):
        obs_TBO, hand_TBY, dones_TB, avail_actions_TBA = x_TB_

        if self.config["PUBLIC_RNN"]:
            obs_rnn_TB_ = obs_TBO[..., self.OBS_PRIVATE_SIZE :]
        else:
            obs_rnn_TB_ = obs_TBO

        obs_actor_TBO = obs_TBO

        if self.config["CRITIC_CENTRALIZED"]:
            obs_critic_TB_ = jnp.concatenate([hand_TBY, obs_TBO], axis=-1)
        else:
            obs_critic_TB_ = obs_TBO

        # FF streams
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

        if self.config["RNN"] not in ["LSTM", "GRU"]:
            actor_mean_TBW = embedding_obs_actor_TBW
            critic_mean_TBW = embedding_obs_critic_TBW
        else:
            embedding_obs_rnn_TBW = nn.relu(
                nn.Dense(
                    self.config["LAYER_WIDTH"],
                    kernel_init=orthogonal(np.sqrt(2)),
                    bias_init=constant(0.0),
                )(obs_rnn_TB_)
            )

            rnn_in_TB_ = (embedding_obs_rnn_TBW, dones_TB)
            if self.config["RNN"] == "LSTM":
                hidden_state_LBW_, embedding_obs_rnn_TBW = ScannedMultiLayerLSTM(
                    self.config["NUM_RNN_LAYERS"]
                )(hidden_state_LBW_, rnn_in_TB_)
            else:
                raise NotImplementedError("new_obl only implemented for LSTM")

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

    @staticmethod
    def initialize_hidden_state(rnn: str, num_rnn_layers: int, batch_size: int, hidden_size: int):
        if rnn == "LSTM":
            return ScannedMultiLayerLSTM.initialize_hidden_state(
                num_rnn_layers, batch_size, hidden_size
            )
        else:
            raise NotImplementedError("new_obl only implemented for LSTM")


def load_actor_vars_for_new_obl(params_path: str):
    """
    new_obl weights saved as safetensors; we reuse the generic loader.
    """
    return load_flax_safetensors(params_path)


# -------------------------
# Networks (original)
# -------------------------
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
        x, resets = x
        if z is not None:
            T = x.shape[0]
            if z.ndim == 2:
                z = jnp.broadcast_to(z[None, ...], (T,) + z.shape)
            elif z.ndim == 3 and z.shape[0] == 1 and T != 1:
                z = jnp.broadcast_to(z, (T,) + z.shape[1:])
            x = jnp.concatenate([x, z], axis=-1)

        for fs in self.preprocessing_features:
            x = nn.Dense(fs)(x)
            x = nn.LayerNorm()(x)
            x = self.act_fn(x)

        new_cs, new_hs = [], []
        for i, fs in enumerate(self.lstm_features):
            layer = self.lstm(fs)
            layer_carry = jax.tree.map(lambda c: c[i], carry)
            layer_carry, x = layer(layer_carry, (x, resets))
            new_c, new_h = layer_carry
            new_cs.append(new_c)
            new_hs.append(new_h)

        for fs in self.postprocessing_features:
            x = nn.Dense(fs)(x)
            x = self.act_fn(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        logits = nn.Dense(self.action_dim)(x)

        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_carry, logits, x

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return zeros_lstm_carry(self.lstm_features, batch_size)


class ActorLstmBCAdapter(nn.Module):
    preprocessing_features: List[int]
    lstm_features: List[int]
    postprocessing_features: List[int]
    action_dim: int
    dropout_rate: float
    activation_fn_name: str

    def setup(self):
        self.net = MultiLayerLstm(
            preprocessing_features=self.preprocessing_features,
            lstm_features=self.lstm_features,
            postprocessing_features=self.postprocessing_features,
            action_dim=self.action_dim,
            dropout_rate=self.dropout_rate,
            activation_fn_name=self.activation_fn_name,
        )

    @nn.compact
    def __call__(self, carry, x, z: jnp.ndarray = None, training: bool = False):
        obs, resets = x
        x_bt = jnp.swapaxes(obs, 0, 1)
        carry, logits_bt = self.net(x=x_bt, carry=carry, training=training)
        logits = jnp.swapaxes(logits_bt, 0, 1)
        return carry, logits, logits

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return self.net.initialize_carry(batch_size)


class IPPOAdapter(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, carry, x, z: jnp.ndarray = None, training: bool = False):
        if len(x) == 2:
            obs, _resets = x
            avail = None
        else:
            obs, _resets, avail = x
        obs_b = obs.squeeze(0)
        h = nn.Dense(512)(obs_b)
        h = nn.relu(h)
        h = nn.Dense(512)(h)
        h = nn.relu(h)
        logits = nn.Dense(self.action_dim)(h)
        if avail is not None:
            logits = logits - (1.0 - avail.squeeze(0)) * 1e10
        return carry, logits[None, ...], logits[None, ...]

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return None


# -------------------------
# Loaders / builders (mirror XP)
# -------------------------
def build_actor_module(args, action_dim: int, act_fn_name: str):
    return ActorLstm(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=act_fn_name or args.act_fn,
    )


def build_bc_actor_module(args, action_dim: int, act_fn_name: str):
    return ActorLstmBCAdapter(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=act_fn_name or args.act_fn,
    )


def ippo_load_checkpoint(checkpoint_path: str):
    import pickle

    with open(checkpoint_path, "rb") as f:
        params = pickle.load(f)
    if "params" in params and isinstance(params["params"], dict) and "layer1" in params["params"]:
        key_map = {
            "layer1": "Dense_0",
            "layer2": "Dense_1",
            "layer3": "Dense_2",
            "layer4": "Dense_3",
            "layer5": "Dense_4",
        }
        params["params"] = {key_map[k]: v for k, v in params["params"].items() if k in key_map}
    return freeze(params)


def load_actor_vars_for_ippo(pickle_path: str):
    return ippo_load_checkpoint(pickle_path)


def load_actor_vars_for_bc(params_path: str):
    full_vars = load_flax_safetensors(params_path)
    p = unfreeze(full_vars)
    assert "params" in p
    return freeze({"params": {"net": p["params"]}})


def load_actor_vars_for_rl(params_path: str):
    full_vars = load_flax_safetensors(params_path)
    p = unfreeze(full_vars)
    assert "params" in p
    if "actor" not in p["params"]:
        raise ValueError(f"'actor' not found in params of {params_path}")

    actor_block = p["params"]["actor"]
    if isinstance(actor_block, dict) and "policy" in actor_block:
        actor_block = actor_block["policy"]
    elif isinstance(actor_block, dict) and len(actor_block) == 1:
        only_k = next(iter(actor_block))
        if isinstance(actor_block[only_k], dict):
            actor_block = actor_block[only_k]
    if isinstance(actor_block, dict) and "policy_out" in actor_block:
        actor_block["Dense_2"] = actor_block["policy_out"]
        if "adv_out" in actor_block:
            actor_block.pop("adv_out")

    return freeze({"params": actor_block})


def extract_decoder_params(vae_vars):
    p = unfreeze(vae_vars)
    assert "params" in p
    params = p["params"]
    if "decoder" in params:
        dec = params["decoder"]
    elif "MultiLayerLstm_0" in params:
        dec = params["MultiLayerLstm_0"]
    else:
        dec = params
    return freeze(dec), freeze({"params": {"decoder": dec}})


def load_actor_vars_for_vae(vae_checkpoint_path: str):
    vae_vars = load_flax_safetensors(vae_checkpoint_path)
    decoder_params, _ = extract_decoder_params(vae_vars)
    return freeze({"params": decoder_params})


def infer_z_for_rl(
    params_path: str,
    z_npy: str = None,
    yaml_with_z: str = None,
    prior_json: str = None,
    z_seed: int = None,
    use_prior_mean: bool = False,
):
    if z_npy and os.path.exists(z_npy):
        return jnp.array(np.load(z_npy))
    if yaml_with_z and os.path.exists(yaml_with_z):
        import yaml

        with open(yaml_with_z, "r") as f:
            cfg = yaml.safe_load(f)
        if "sampled_z" in cfg:
            return jnp.array(cfg["sampled_z"], dtype=jnp.float32)
    if use_prior_mean:
        if prior_json is None:
            raise ValueError("use_prior_mean=True requires --prior_json.")
        return z_from_prior_mean(prior_json)
    if prior_json and z_seed is not None:
        rng = jax.random.PRNGKey(int(z_seed))
        return sample_z_from_prior(rng, prior_json)
    raise ValueError(
        "For RL: provide --*_z_npy | --*_yaml_with_z | --*_use_prior_mean | (--prior_json & --*_z_seed)."
    )


def infer_z_for_vae(
    prior_json: str, z_npy: str = None, z_seed: int = None, use_prior_mean: bool = False
):
    if z_npy and os.path.exists(z_npy):
        return jnp.array(np.load(z_npy))
    if use_prior_mean:
        if prior_json is None:
            raise ValueError("use_prior_mean=True requires --prior_json.")
        return z_from_prior_mean(prior_json)
    if z_seed is None:
        raise ValueError(
            "For VAE: provide --*_z_npy | --*_use_prior_mean | --*_z_seed (with --prior_json)."
        )
    if prior_json is None:
        raise ValueError("Sampling z requires --prior_json for the GMM prior.")
    rng = jax.random.PRNGKey(int(z_seed))
    return sample_z_from_prior(rng, prior_json)


# -------------------------
# Policy holders
# -------------------------
class Policy:
    def __init__(self, actor_module, actor_vars, z: jnp.ndarray, num_agents: int, action_dim: int):
        self.actor = actor_module
        self.vars = freeze({"params": unfreeze(actor_vars)["params"]})
        self.z = z
        self.num_agents = num_agents
        self.action_dim = action_dim

    def initialize_hidden_state(self):
        if isinstance(self.actor, ActorLstm):
            return self.actor.initialize_carry(self.num_agents)
        if isinstance(self.actor, ActorLstmBCAdapter):
            bound = self.actor.bind(self.vars)
            return bound.net.initialize_carry(self.num_agents)
        if isinstance(self.actor, IPPOAdapter):
            return None
        return None

    def forward_logits(self, carry, obs_dict, done_scalar, legal_moves_dict):
        agents = list(obs_dict.keys())
        B = len(agents)
        obs_batch = jnp.stack([obs_dict[a] for a in agents])[None, ...]
        resets = jnp.repeat(jnp.array(done_scalar)[None, None], B, axis=1)
        avail = jnp.stack([legal_moves_dict[a] for a in agents])[None, ...]
        z_batch = jnp.repeat(self.z[None, :], B, axis=0) if self.z.size > 0 else None

        if isinstance(self.actor, IPPOAdapter):
            carry, logits, _ = self.actor.apply(
                self.vars, carry, (obs_batch, resets, avail), z=None, training=False
            )
        else:
            carry, logits, _ = self.actor.apply(
                self.vars, carry, (obs_batch, resets), z=z_batch, training=False
            )

        masked_logits = logits - (1.0 - avail) * 1e10
        return carry, masked_logits


class OBLPolicy:
    def __init__(self, model, params, num_agents: int, action_dim: int, publ_offset: int = 125):
        self.model = model
        if (
            "params" in params
            and isinstance(params["params"], dict)
            and "params" in params["params"]
        ):
            params = freeze(params["params"])
        self.params = params
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.publ_offset = publ_offset

    def initialize_hidden_state(self):
        return self.model.initialize_carry(jax.random.PRNGKey(0), (self.num_agents,))

    def forward_logits(self, carry, obs_dict, done_scalar, legal_moves_dict):
        agents = list(obs_dict.keys())
        obs_b = jnp.stack([obs_dict[a] for a in agents], 0)  # (B, D)
        priv_s = obs_b
        publ_s = obs_b[:, self.publ_offset :]

        # OBL forward → "adv"-like scores
        carry, adv = self.model.apply(self.params, carry, (priv_s, publ_s))

        legal = jnp.stack([legal_moves_dict[a] for a in agents], 0)  # (B, A)

        # match rollout_obl_to_safetensors.py
        adv_min = jnp.min(adv, axis=-1, keepdims=True)
        scores = (1.0 + adv - adv_min) * legal  # (B, A)

        # KL code wants logits:
        logits = jnp.log(scores + 1e-8) - (1.0 - legal) * 1e10

        return carry, logits[None, ...]


class NewOBLPolicy:
    """
    Wraps the new_obl ActorCritic so it fits the Policy interface used in KL eval.
    """

    def __init__(self, model, vars, num_agents: int, action_dim: int, config: Dict[str, Any]):
        self.model = model
        self.vars = vars
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.config = config

    def initialize_hidden_state(self):
        return ActorCritic.initialize_hidden_state(
            self.config["RNN"],
            self.config["NUM_RNN_LAYERS"],
            self.num_agents,
            self.config["LAYER_WIDTH"],
        )

    def forward_logits(self, carry, obs_dict, done_scalar, legal_moves_dict):
        agents = list(obs_dict.keys())
        B = len(agents)

        obs_B0 = jnp.stack([obs_dict[a] for a in agents], 0)  # (B, O)
        obs_TBO = obs_B0[None, ...]  # (1, B, O)

        hand_TBY = jnp.zeros((1, B, 1), dtype=jnp.float32)  # unused when CRITIC_CENTRALIZED=False
        dones_TB = jnp.broadcast_to(done_scalar, (1, B))  # simple scalar reset flag
        legals_BA = jnp.stack([legal_moves_dict[a] for a in agents], 0)
        avail_TBA = legals_BA[None, ...]  # (1, B, A)

        carry_out, logits_TBA, _ = self.model.apply(
            self.vars,
            carry,
            (obs_TBO, hand_TBY, dones_TB, avail_TBA),
        )
        # logits already masked inside the model
        return carry_out, logits_TBA


# -------------------------
# Builders for gen/eval
# -------------------------
def build_policy(args, role: str, num_agents: int, action_dim: int):
    typ = getattr(args, f"{role}_type")
    act_fn_name = getattr(args, f"{role}_act_fn") or args.act_fn

    if typ == "bc":
        actor_mod = build_bc_actor_module(args, action_dim, act_fn_name)
        params_path = getattr(args, f"{role}_bc_params")
        if params_path is None:
            raise ValueError(f"--{role}_bc_params required for BC policy")
        actor_vars = load_actor_vars_for_bc(params_path)
        z = jnp.zeros((0,), dtype=jnp.float32)
        return Policy(actor_mod, actor_vars, z, num_agents, action_dim)

    if typ == "ippo":
        actor_mod = IPPOAdapter(action_dim=action_dim)
        ippo_ckpt = getattr(args, f"{role}_ippo_ckpt")
        if ippo_ckpt is None:
            raise ValueError(f"--{role}_ippo_ckpt required for IPPO policy")
        actor_vars = load_actor_vars_for_ippo(ippo_ckpt)
        z = jnp.zeros((0,), dtype=jnp.float32)
        return Policy(actor_mod, actor_vars, z, num_agents, action_dim)

    if typ == "rl_bc":
        actor_mod = build_actor_module(args, action_dim, act_fn_name)
        params_path = getattr(args, f"{role}_params")
        if params_path is None:
            raise ValueError(f"--{role}_params required for RL_BC policy")
        actor_vars = load_actor_vars_for_rl(params_path)
        z = jnp.zeros((0,), dtype=jnp.float32)
        return Policy(actor_mod, actor_vars, z, num_agents, action_dim)

    if typ == "rl":
        actor_mod = build_actor_module(args, action_dim, act_fn_name)
        params_path = getattr(args, f"{role}_params")
        if params_path is None:
            raise ValueError(f"--{role}_params required for RL policy")
        actor_vars = load_actor_vars_for_rl(params_path)
        z = infer_z_for_rl(
            params_path=params_path,
            z_npy=getattr(args, f"{role}_z_npy"),
            yaml_with_z=getattr(args, f"{role}_yaml_with_z"),
            prior_json=args.prior_json,
            z_seed=getattr(args, f"{role}_z_seed"),
            use_prior_mean=getattr(args, f"{role}_use_prior_mean"),
        )
        return Policy(actor_mod, actor_vars, z, num_agents, action_dim)

    if typ == "vae":
        actor_mod = build_actor_module(args, action_dim, act_fn_name)
        vae_ckpt = getattr(args, f"{role}_vae_ckpt")
        if vae_ckpt is None:
            raise ValueError(f"--{role}_vae_ckpt required for VAE policy")
        actor_vars = load_actor_vars_for_vae(vae_ckpt)
        z = infer_z_for_vae(
            prior_json=args.prior_json,
            z_npy=getattr(args, f"{role}_z_npy"),
            z_seed=getattr(args, f"{role}_z_seed"),
            use_prior_mean=getattr(args, f"{role}_use_prior_mean"),
        )
        return Policy(actor_mod, actor_vars, z, num_agents, action_dim)

    if typ == "obl":
        params_path = getattr(args, f"{role}_obl_params")
        if params_path is None:
            raise ValueError(f"--{role}_obl_params required for OBL policy")
        actor_vars = load_actor_vars_for_obl(params_path)
        model = OBLR2D2(out_dim=action_dim)
        # we bind params directly
        return OBLPolicy(model, actor_vars, num_agents, action_dim, publ_offset=125)

    if typ == "new_obl":
        params_path = getattr(args, f"{role}_new_obl_params")
        if params_path is None:
            raise ValueError(f"--{role}_new_obl_params required for new_obl policy")
        actor_vars = load_actor_vars_for_new_obl(params_path)
        model = ActorCritic(action_dim=action_dim, config=NEW_OBL_CONFIG)
        return NewOBLPolicy(model, actor_vars, num_agents, action_dim, config=NEW_OBL_CONFIG)

    raise ValueError(f"Unknown {role}_type: {typ}")


# -------------------------
# KL helper
# -------------------------
def kl_logits(eval_logits: jnp.ndarray, gen_logits: jnp.ndarray) -> jnp.ndarray:
    logp = jax.nn.log_softmax(eval_logits, axis=-1)
    logq = jax.nn.log_softmax(gen_logits, axis=-1)
    p = jnp.exp(logp)
    return jnp.sum(p * (logp - logq), axis=-1)


# -------------------------
# rollout_kl
# -------------------------
def rollout_kl(rng, args, env):
    first_agent = list(env.agents)[0]
    legal0 = env.get_legal_moves(unwrap_state(env.reset(jax.random.PRNGKey(0))[1]))
    action_dim = int(legal0[first_agent].shape[-1])

    gen_pol = build_policy(args, "gen", num_agents=len(env.agents), action_dim=action_dim)
    eval_pol = build_policy(args, "eval", num_agents=len(env.agents), action_dim=action_dim)

    gen_carry = gen_pol.initialize_hidden_state()
    eval_carry = eval_pol.initialize_hidden_state()

    # NEW: are we in the "OBL generator vs BC/RL_BC" regime?
    is_obl_action_match = getattr(args, "gen_type", None) == "obl" and getattr(
        args, "eval_type", None
    ) in ("bc", "rl_bc")

    use_jit = args.use_jit if args.use_jit is not None else True
    with jax.disable_jit(not use_jit):
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng)

        def cond_fn(val):
            done = val[2]
            return jnp.logical_not(done)

        @jax.jit
        def _step_env(rng, env_state, actions):
            rng, _rng = jax.random.split(rng)
            new_obs, new_env_state, reward, dones, infos = env.step(_rng, env_state, actions)
            new_legal_moves = env.get_legal_moves(unwrap_state(new_env_state))
            return rng, new_env_state, new_obs, reward, dones, new_legal_moves

        # init accumulators
        if is_obl_action_match:
            acc_val = 0.0  # will store "number of mismatches"
        else:
            acc_val = 0.0  # will store KL sum
        _done = False
        legal_moves = env.get_legal_moves(unwrap_state(env_state))

        def body_fn(val):
            acc_val, t, done, rng, env_state, obs, legal_moves, gen_carry, eval_carry = val

            curr_player = jnp.argmax(unwrap_state(env_state).cur_player_idx)

            gen_carry, gen_logits = gen_pol.forward_logits(gen_carry, obs, done, legal_moves)
            eval_carry, eval_logits = eval_pol.forward_logits(eval_carry, obs, done, legal_moves)

            if is_obl_action_match:
                # compare *actions* of acting player
                gen_act = jnp.argmax(gen_logits[0, curr_player, :])
                eval_act = jnp.argmax(eval_logits[0, curr_player, :])
                mismatch = jnp.not_equal(gen_act, eval_act).astype(jnp.float32)
                acc_val = acc_val + mismatch
            else:
                # original KL
                gl = gen_logits[0, curr_player, :]
                el = eval_logits[0, curr_player, :]
                step_kl = kl_logits(el, gl)
                acc_val = acc_val + step_kl

            # env step uses GENERATOR actions
            rng, act_rng = jax.random.split(rng)
            # always act from generator, not eval
            gen_actions_all = jnp.argmax(gen_logits.squeeze(0), axis=-1)
            actions = {
                agent_name: jnp.array(gen_actions_all[k])
                for k, agent_name in enumerate(env.agents)
            }

            rng, env_state, obs, reward, dones, legal_moves = _step_env(rng, env_state, actions)

            done = dones["__all__"]
            t = t + 1
            return (acc_val, t, done, rng, env_state, obs, legal_moves, gen_carry, eval_carry)

        init_val = (acc_val, 0, False, rng, env_state, obs, legal_moves, gen_carry, eval_carry)
        acc_val, t, _, _, _, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)
        return acc_val, t


def run_many(rng_keys, args, env):
    f = jax.vmap(lambda r: rollout_kl(r, args, env), in_axes=0, out_axes=0)
    vals, steps = f(rng_keys)

    is_obl_action_match = getattr(args, "gen_type", None) == "obl" and getattr(
        args, "eval_type", None
    ) in ("bc", "rl_bc")

    if is_obl_action_match:
        # vals = #mismatches per episode
        mismatch_rates = vals / steps
        mean_total = jnp.mean(mismatch_rates)
        sem_total = jnp.std(mismatch_rates) / jnp.sqrt(mismatch_rates.shape[0])
        # for API compatibility, also return "per-step" as the same thing
        return vals, steps, mean_total, sem_total, mean_total, sem_total
    else:
        # original KL path
        mean_total = jnp.mean(vals)
        sem_total = jnp.std(vals) / jnp.sqrt(vals.shape[0])

        per_step_vals = vals / steps
        mean_per_step = jnp.mean(per_step_vals)
        sem_per_step = jnp.std(per_step_vals) / jnp.sqrt(per_step_vals.shape[0])

        return vals, steps, mean_total, sem_total, mean_per_step, sem_per_step


def main():
    parser = argparse.ArgumentParser(description="KL_{tau~gen}( eval || gen ) evaluator.")

    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_jit", type=int, default=1)

    parser.add_argument("--preprocessing_features", type=int, nargs="+", default=[1024])
    parser.add_argument("--lstm_features", type=int, nargs="+", default=[512])
    parser.add_argument("--postprocessing_features", type=int, nargs="+", default=[256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--act_fn", type=str, choices=["relu", "gelu"], default="relu")
    parser.add_argument("--gen_act_fn", type=str, choices=["relu", "gelu"], default=None)
    parser.add_argument("--eval_act_fn", type=str, choices=["relu", "gelu"], default=None)

    parser.add_argument("--prior_json", type=str, default=None)

    parser.add_argument(
        "--gen_type",
        type=str,
        choices=["rl", "vae", "bc", "ippo", "rl_bc", "obl", "new_obl"],
        required=True,
    )
    parser.add_argument("--gen_params", type=str, default=None)
    parser.add_argument("--gen_vae_ckpt", type=str, default=None)
    parser.add_argument("--gen_bc_params", type=str, default=None)
    parser.add_argument("--gen_ippo_ckpt", type=str, default=None)
    parser.add_argument("--gen_yaml_with_z", type=str, default=None)
    parser.add_argument("--gen_z_npy", type=str, default=None)
    parser.add_argument("--gen_z_seed", type=int, default=None)
    parser.add_argument("--gen_use_prior_mean", action="store_true", default=False)
    parser.add_argument("--gen_obl_params", type=str, default=None)  # <--- new
    parser.add_argument("--gen_mode", type=str, choices=["greedy", "sample"], default="sample")

    parser.add_argument(
        "--eval_type",
        type=str,
        choices=["rl", "vae", "bc", "ippo", "rl_bc", "obl", "new_obl"],
        required=True,
    )
    parser.add_argument("--eval_params", type=str, default=None)
    parser.add_argument("--eval_vae_ckpt", type=str, default=None)
    parser.add_argument("--eval_bc_params", type=str, default=None)
    parser.add_argument("--eval_ippo_ckpt", type=str, default=None)
    parser.add_argument("--eval_yaml_with_z", type=str, default=None)
    parser.add_argument("--eval_z_npy", type=str, default=None)
    parser.add_argument("--eval_z_seed", type=int, default=None)
    parser.add_argument("--eval_use_prior_mean", action="store_true", default=False)
    parser.add_argument("--eval_obl_params", type=str, default=None)  # <--- new

    args = parser.parse_args()

    hand_size = 5 if args.num_players <= 3 else 4
    use_raw_obs = any(t in ("bc", "obl", "new_obl") for t in (args.gen_type, args.eval_type))
    base_env = make("hanabi", hand_size=hand_size, num_agents=args.num_players)
    env = base_env if use_raw_obs else LogWrapper(base_env)

    rng = jax.random.PRNGKey(args.seed)
    rng_keys = jax.random.split(rng, args.num_rollouts)
    kl_vals, steps, mean_total, sem_total, mean_per_step, sem_per_step = run_many(
        rng_keys, args, env
    )

    print(f"KL_{{tau~gen}}(eval||gen) over {args.num_rollouts} rollouts:")
    print(f"  total KL per episode (nats): {float(mean_total):.3f} ± {float(sem_total):.3f} (SEM)")
    print(f"  steps per episode: mean={float(jnp.mean(steps)):.1f}")
    print(
        f"  per-step KL (nats/step):     {float(mean_per_step):.4f} ± {float(sem_per_step):.4f} (SEM)"
    )
    print("  units are nats; divide by ln(2) for bits.")


if __name__ == "__main__":
    main()
