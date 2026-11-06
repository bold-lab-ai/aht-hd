# crossplay_eval.py
import argparse
import json
import os
from typing import Any, Dict, List

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze
from flax.traverse_util import unflatten_dict
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper
from safetensors.numpy import load_file as st_load

# ---- your project LSTM cells ----
from ah2ac2.nn.multi_layer_lstm import LstmCellWithHiddenStateReset, MultiLayerLstm
from ah2ac2.tools.kl_eval import OBLR2D2
from ah2ac2.training.br_bc_lstm import ActorCriticRNN

"""
e.g.
python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 1024 \
  --seed 0 \
  --pol1_type ippo \
  --pol1_ippo_ckpt ah2ac2/models/proud-music/op_ippo_ff/checkpoint_1.pkl \
  --pol1_act_fn relu \
  --pol2_type bc \
  --pol2_bc_params checkpoints/bc/op_bc_proudmusic_chpt1/2k/epoch46_seed0_score19.158_2p.safetensors \
  --pol2_act_fn gelu \
  --eval_mode greedy

  python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 1024 \
  --seed 0 \
  --pol1_type ippo \
  --pol1_ippo_ckpt ah2ac2/models/proud-music/op_ippo_ff/checkpoint_10.pkl \
  --pol1_act_fn relu \
  --pol2_type ippo \
  --pol2_ippo_ckpt ah2ac2/models/proud-music/op_ippo_ff/checkpoint_10.pkl \
  --pol2_act_fn relu \
  --eval_mode greedy

python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 1024 \
  --seed 0 \
  --pol1_type ippo \
  --pol1_ippo_ckpt ah2ac2/models/ippo_ff/ippo_ff_10/checkpoint_1.pkl \
  --pol1_act_fn relu \
  --pol2_type ippo \
  --pol2_ippo_ckpt ah2ac2/models/ippo_ff/ippo_ff_10/checkpoint_1.pkl \
  --pol2_act_fn relu \
  --eval_mode greedy

  python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 1024 \
  --seed 0 \
  --pol1_type ippo \
  --pol1_ippo_ckpt ah2ac2/models/proud-music/op_ippo_ff/checkpoint_1.pkl \
  --pol1_act_fn relu \
  --pol2_type rl_bc \
  --pol2_params checkpoints/kl_anchored/op_bc_proudmusic_chpt1_2k_19.158/kl_target0.075/seed1_score23.889_klt0.075_2p.safetensors \
  --pol2_act_fn gelu \
  --eval_mode greedy


python ah2ac2/tools/crossplay_eval.py \
--num_players 2 \
--num_rollouts 16384 \
--seed 0 \
--pol1_type bc \
--pol1_bc_params checkpoints/bc/human_bc/100k/epoch34_seed0_score19.918_2p.safetensors \
--pol2_type bc \
--pol2_bc_params checkpoints/bc/human_bc/100k/epoch34_seed0_score19.918_2p.safetensors \
--pol1_act_fn gelu \
--pol2_act_fn gelu \
--eval_mode greedy \
--use_jit 1

python ah2ac2/tools/crossplay_eval.py \
--num_players 2 \
--num_rollouts 1024 \
--seed 0 \
--pol1_type vae \
--pol1_vae_ckpt checkpoints/vae_masked/human_vae/latent_dim32/100k/epoch98_seed0_valacc0.813.safetensors \
--pol1_z_npy "outputs/es_refine_test/es_final_Z_i002_SP=17.266_XPavg=2.453.npy" \
--pol1_act_fn gelu \
--pol2_type br \
--pol2_br_params checkpoints/br_latent_lstm/human_vae/100k/seed0_step76292_2p.safetensors \
--br_fc_dim_size 128 \
--br_gru_hidden_dim 128 \
--eval_mode greedy \
--use_jit 1


python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 16384 \
  --seed 0 \
  --pol1_type vae \
  --pol1_vae_ckpt checkpoints/vae_masked/human_vae/latent_dim32/100k/epoch98_seed0_valacc0.813.safetensors \
  --pol1_z_npy outputs/latent_search_no_xp/es_final_Z_i000_SP=19.938_XPavg=0.000.npy \
  --pol2_type vae \
  --pol2_vae_ckpt checkpoints/vae_masked/human_vae/latent_dim32/100k/epoch98_seed0_valacc0.813.safetensors \
  --pol2_z_npy outputs/latent_search_no_xp/es_final_Z_i000_SP=19.938_XPavg=0.000.npy \
  --pol1_act_fn gelu \
  --pol2_act_fn gelu \
  --eval_mode greedy \
  --use_jit 1

python ah2ac2/tools/crossplay_eval.py \
--num_players 2 \
--num_rollouts 1024 \
--seed 0 \
--pol1_type vae \
--pol1_vae_ckpt checkpoints/vae_masked/human_vae/latent_dim32/100k/epoch98_seed0_valacc0.813.safetensors \
--pol1_z_npy outputs/es_refine_test/es_final_Z_i002_SP=17.266_XPavg=2.453.npy \
--pol1_act_fn gelu \
--pol2_type vae \
--pol2_vae_ckpt checkpoints/vae_lowdim_masked/human_vae/latent_dim32/100k/epoch99_seed0_valacc0.803.safetensors \
--pol2_z_npy checkpoints/vae_lowdim_masked/human_vae/latent_dim32/100k/es_refine/es_final_Z_i001_SP=17.797_XPavg=2.914.npy \
--pol2_act_fn gelu \
--pol2_lstm_features 256 \
--pol2_postprocessing_features 128 \
--eval_mode greedy \
--use_jit 1


python ah2ac2/tools/crossplay_eval.py \
--num_players 2 \
--num_rollouts 1024 \
--seed 0 \
--pol1_type vae \
--pol1_vae_ckpt checkpoints/vae_lowdim_masked/human_vae/latent_dim32/2k/epoch96_seed0_valacc0.566.safetensors \
--pol1_z_npy checkpoints/vae_lowdim_masked/human_vae/latent_dim32/2k/es_refine_4/es_final_Z_i000_SP=4.148_XPavg=1.349.npy \
--pol1_act_fn gelu \
--pol1_lstm_features 256 \
--pol1_postprocessing_features 128 \
--pol2_type vae \
--pol2_vae_ckpt checkpoints/vae_lowdim_masked/human_vae/latent_dim32/2k/epoch96_seed0_valacc0.566.safetensors \
--pol2_z_npy checkpoints/vae_lowdim_masked/human_vae/latent_dim32/2k/es_refine_4/es_final_Z_i000_SP=4.148_XPavg=1.349.npy \
--pol2_act_fn gelu \
--pol2_lstm_features 256 \
--pol2_postprocessing_features 128 \
--eval_mode greedy \
--use_jit 1

python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 1024 \
  --seed 0 \
  --pol1_type rl_vae \
  --pol1_act_fn gelu \
  --pol1_params checkpoints/test_proxies/es_final_Z_i002_SP=17.266_XPavg=2.453/seed1_score21.877_klt0.064_2p.safetensors \
  --pol1_z_npy outputs/es_refine_test/es_final_Z_i002_SP=17.266_XPavg=2.453.npy \
  --pol2_type ippo \
  --pol2_ippo_ckpt ah2ac2/models/ippo_ff/ippo_ff_10/checkpoint_1.pkl \
  --pol2_act_fn relu \
  --eval_mode greedy \
  --use_jit 1


python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 1024 \
  --seed 0 \
  --pol1_type rl_vae \
  --pol1_act_fn gelu \
  --pol1_params checkpoints/test_proxies/es_final_Z_i000_SP=17.156_XPavg=2.203/seed1_score20.117_klt0.064_2p.safetensors \
  --pol1_z_npy outputs/es_refine_test/es_final_Z_i000_SP=17.156_XPavg=2.203.npy \
  --pol2_type ippo \
  --pol2_ippo_ckpt ah2ac2/models/ippo_ff/ippo_ff_10/checkpoint_1.pkl \
  --pol2_act_fn relu \
  --eval_mode greedy \
  --use_jit 1


python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 1024 \
  --seed 0 \
  --pol1_type rl_bc \
  --pol1_act_fn gelu \
  --pol1_params checkpoints/bc_test_proxies/bc_proxy_1/seed1_score23.723_klt0.05_2p.safetensors \
  --pol2_type ippo \
  --pol2_ippo_ckpt ah2ac2/models/proud-music/op_ippo_ff/checkpoint_10.pkl \
  --pol2_act_fn relu \
  --eval_mode greedy \
  --use_jit 1

python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 1024 \
  --seed 0 \
  --pol1_type rl_bc \
  --pol1_act_fn gelu \
  --pol1_params checkpoints/bc_test_proxies/bc_proxy_2/seed1_score24.406_klt0.15_2p.safetensors \
  --pol2_type ippo \
  --pol2_ippo_ckpt ah2ac2/models/ippo_ff/ippo_ff_10/checkpoint_1.pkl \
  --pol2_act_fn relu \
  --eval_mode greedy \
  --use_jit 1

python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 1024 \
  --seed 0 \
  --pol1_type rl_vae \
  --pol1_act_fn gelu \
  --pol1_params checkpoints/test_proxies/es_final_Z_i000_SP=17.156_XPavg=2.203/seed1_score20.117_klt0.064_2p.safetensors \
  --pol1_z_npy outputs/es_refine_test/es_final_Z_i000_SP=17.156_XPavg=2.203.npy \
  --pol2_type rl_bc \
  --pol2_params checkpoints/bc_test_proxies/bc_proxy_2/seed1_score24.406_klt0.15_2p.safetensors \
  --pol2_act_fn gelu \
  --eval_mode greedy \
  --use_jit 1

python ah2ac2/tools/crossplay_eval.py \
  --num_players 2 \
  --num_rollouts 1024 \
  --seed 0 \
  --pol1_type rl_bc \
  --pol1_act_fn gelu \
  --pol1_params checkpoints/bc_test_proxies/bc_proxy_2/seed1_score24.406_klt0.15_2p.safetensors \
  --pol2_type obl \
  --pol2_obl_params ah2ac2/models/icml_OBL4/OFF_BELIEF1_SHUFFLE_COLOR0_LOAD1_BZA0_BELIEF_b.safetensors \
  --eval_mode greedy \
  --use_jit 1
"""

# --- DIAGNOSTICS: ---


def bc_forward_equivalence_check(env, args, ckpt_path):
    print("\n[BC ONE-STEP FORWARD CHECK]")
    # Build env slice
    key = jax.random.PRNGKey(0)
    obs, es = env.reset(key)
    legal = env.get_legal_moves(unwrap_state(es))
    agents = list(env.agents)
    B = len(agents)
    first_agent = agents[0]
    _D = int(obs[first_agent].shape[-1])

    # Build model and load params (direct MultiLayerLstm)
    model = MultiLayerLstm(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=args.action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=args.act_fn,
    )
    carr_direct = model.initialize_carry(B)
    loaded = load_flax_safetensors(ckpt_path)["params"]
    if "MultiLayerLstm_0" in loaded:  # optional unwrapping
        loaded = loaded["MultiLayerLstm_0"]
    # inputs (B,1,D)
    x_bt = jnp.stack([obs[a] for a in agents], axis=0)[:, None, :]

    # Direct path
    carr_d2, logits_bt = model.apply({"params": loaded}, x=x_bt, carry=carr_direct, training=False)
    logits_direct = logits_bt  # (B,1,A)

    # Adapter path (your current code path)
    adapter = ActorLstmBCAdapter(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=args.action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=args.act_fn,
    )
    # variables matching your loader: {"params":{"net": loaded}}
    adapter_vars = freeze({"params": {"net": loaded}})
    carr_adapt = adapter.bind(adapter_vars).net.initialize_carry(B)

    # adapter expects (T,B,D) with T=1, so swap:
    x_tb = jnp.swapaxes(x_bt, 0, 1)  # (1,B,D)
    carr_a2, logits_tb, _ = adapter.apply(
        adapter_vars, carr_adapt, (x_tb, jnp.zeros((1, B), bool)), training=False
    )
    logits_adapter_bt = jnp.swapaxes(logits_tb, 0, 1)  # (B,1,A)

    # Compare a few numbers
    diff = jnp.max(jnp.abs(logits_direct - logits_adapter_bt))
    print(" max|logit_direct - logit_adapter| =", float(diff))
    # After masking, argmax should agree
    avail = jnp.stack([legal[a] for a in agents], axis=0)[:, None, :]
    ld = logits_direct - (1.0 - avail) * 1e10
    la = logits_adapter_bt - (1.0 - avail) * 1e10
    a1 = jnp.argmax(ld, axis=-1)
    a2 = jnp.argmax(la, axis=-1)
    agree = jnp.all(a1 == a2)
    print(" masked-argmax agree? ->", bool(agree))
    if not agree:
        # dump a few entries for inspection
        print(" first agent logits (direct vs adapter):")
        print(np.array(ld[0, 0, :5])), print(np.array(la[0, 0, :5]))


def tree_shapes(tree):
    return jax.tree.map(lambda x: tuple(x.shape), tree)


def compare_param_trees(expected, loaded, prefix=""):
    ok = True
    # keys present in either
    exp_keys = set(expected.keys())
    ld_keys = set(loaded.keys())
    for k in sorted(exp_keys - ld_keys):
        print(f"[PARAM-MISS] in loaded: {prefix}{k}")
        ok = False
    for k in sorted(ld_keys - exp_keys):
        print(f"[PARAM-EXTRA] in loaded: {prefix}{k}")
        ok = False
    for k in sorted(exp_keys & ld_keys):
        if isinstance(expected[k], dict) and isinstance(loaded[k], dict):
            ok &= compare_param_trees(expected[k], loaded[k], prefix + k + "/")
        else:
            if getattr(expected[k], "shape", None) != getattr(loaded[k], "shape", None):
                print(
                    f"[SHAPE-MISMATCH] {prefix}{k}: expected {getattr(expected[k], 'shape', None)} vs loaded {getattr(loaded[k], 'shape', None)}"
                )
                ok = False
    return ok


def ck_bc_params_match(env, args, ckpt_path):
    # env obs & dims
    _rng = jax.random.PRNGKey(0)
    obs0, es0 = env.reset(_rng)
    first_agent = list(env.agents)[0]
    obs_dim = int(obs0[first_agent].shape[-1])

    _hand_size = 5 if args.num_players <= 3 else 4
    # Build a *fresh* MultiLayerLstm to get expected param structure
    model = MultiLayerLstm(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=args.action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=args.act_fn,
    )
    carr = model.initialize_carry(batch_size=len(env.agents))
    vars_init = model.init(
        jax.random.PRNGKey(123),
        x=jnp.ones((len(env.agents), 1, obs_dim), jnp.float32),
        carry=carr,
        training=False,
    )
    expected = vars_init["params"]

    # Load the checkpoint as you do
    raw = load_flax_safetensors(ckpt_path)["params"]
    # Some runs save top-level under a module name; unwrap if present
    if "MultiLayerLstm_0" in raw:  # optional
        raw = raw["MultiLayerLstm_0"]

    print("\n[BC PARAM TREE CHECK]")
    ok = compare_param_trees(unfreeze(expected), unfreeze(raw))
    if ok:
        print("✓ Param tree matches (keys & shapes).")
    else:
        print("✗ Param tree mismatch detected.")
    return raw, expected


# -------------------------
# Utilities
# -------------------------
# add near the top (after imports)
def unwrap_state(s):
    # Works for both raw EnvState and LogEnvState
    return getattr(s, "env_state", s)


def load_flax_safetensors(path: str):
    """Load a Flax params tree from a .safetensors file.
    Supports '/' or ',' separators. Wraps under 'params' if missing.
    """
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
    tree = unflatten_dict(pairs)  # nested dict
    if "params" not in tree:
        tree = {"params": tree}
    return freeze(tree)


def actor_uses_policy_out(actor_params_tree) -> bool:
    """
    Detect whether the actor head is named 'policy_out' (newer rl_bc / rl_vae style)
    vs unnamed Dense_2 (older style).
    """
    p = unfreeze(actor_params_tree)

    # common layouts we might see:
    #  - {"policy_out": {...}, "Dense_0":..., ...}
    #  - {"ActorLstm_0": {"policy_out": {...}, ...}}
    #  - {"actor": {"policy_out": {...}}} (if passed wrong slice)
    def has_policy_out(d):
        if not isinstance(d, dict):
            return False
        if "policy_out" in d:
            return True
        return any(has_policy_out(v) for v in d.values() if isinstance(v, dict))

    return has_policy_out(p)


def _unflatten_dict_obl(flat: Dict[str, Any], sep: str = ",") -> Dict[str, Any]:
    # flat: Dict[str, np.ndarray] from safetensors
    pairs = {}
    for k, v in flat.items():
        key = tuple(k.split(sep)) if sep in k else (k,)
        pairs[key] = jnp.array(v)
    tree = unflatten_dict(pairs)
    # OBL checkpoints might already have "params" at top-level or might not.
    if "params" not in tree:
        tree = {"params": tree}
    return tree


def load_actor_vars_for_obl(params_path: str):
    flat = st_load(params_path)  # Dict[str, np.ndarray]
    nested = _unflatten_dict_obl(flat, sep=",")  # {"params": ...} (maybe)
    p = unfreeze(nested["params"])

    # If checkpoint did not include the Flax top-level module name, add it.
    # OBLAdapter creates OBLR2D2 as a submodule, so params must live under OBLR2D2_0.
    if "OBLR2D2_0" not in p:
        p = {"OBLR2D2_0": p}

    return freeze({"params": p})


# Add this small helper near the other builders:
def build_bc_core_and_params(params_path, args):
    # direct MultiLayerLstm + its raw params (no adapter)
    net = MultiLayerLstm(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=args.action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=args.act_fn,
    )
    raw = load_flax_safetensors(params_path)["params"]
    if "MultiLayerLstm_0" in raw:
        raw = raw["MultiLayerLstm_0"]
    params = freeze(raw)
    return net, params


def extract_decoder_params(vae_vars):
    """Return decoder params (frozen) and a variables tree for apply."""
    p = unfreeze(vae_vars)
    assert "params" in p, "expected 'params' at top-level after loading"
    params = p["params"]
    if "decoder" in params:
        dec = params["decoder"]
    elif "MultiLayerLstm_0" in params:
        dec = params["MultiLayerLstm_0"]
    else:
        # If file contained only decoder
        dec = params
    return freeze(dec), freeze({"params": {"decoder": dec}})


def zeros_lstm_carry(lstm_features: List[int], batch_size: int):
    cs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in lstm_features]
    hs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in lstm_features]
    return jnp.stack(cs), jnp.stack(hs)


def z_from_prior_mean(prior_json_path: str):
    """Return mixture mean of a GMM prior: sum_k w_k * mu_k."""
    with open(prior_json_path, "r") as f:
        prior = json.load(f)
    weights = jnp.array(prior["weights"], dtype=jnp.float32)
    weights = weights / weights.sum()
    means = jnp.array(prior["means"], dtype=jnp.float32)  # (K, L)
    # tensordot over K -> (L,)
    return jnp.tensordot(weights, means, axes=1)


def sample_z_from_prior(rng, prior_json_path: str):
    """Matches training sampling: categorical over K, then z = mu + L @ eps."""
    with open(prior_json_path, "r") as f:
        prior = json.load(f)
    ptype = prior.get("type", "gmm-fullcov")
    if ptype != "gmm-fullcov":
        raise ValueError(f"Unsupported prior type: {ptype}")
    weights = jnp.array(prior["weights"], dtype=jnp.float32)
    weights = weights / weights.sum()
    means = jnp.array(prior["means"], dtype=jnp.float32)  # (K,L)
    Ls = jnp.array(prior["cov_cholesky"], dtype=jnp.float32)  # (K,L,L)
    k = jax.random.categorical(rng, jnp.log(weights + 1e-8))
    mu = means[k]
    L = Ls[k]
    eps = jax.random.normal(rng, (mu.shape[0],))
    z = mu + L @ eps
    return z


# -------------------------
# Actor network (same as training, actor-only)
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
        # x: (obs, resets) with shapes obs:(T,B,D), resets:(T,B)
        x, resets = x
        # Concatenate fixed z to obs along features.
        if z is not None:
            T = x.shape[0]
            if z.ndim == 2:  # (B,L) -> (T,B,L)
                z = jnp.broadcast_to(z[None, ...], (T,) + z.shape)
            elif z.ndim == 3 and z.shape[0] == 1 and T != 1:  # (1,B,L) -> (T,B,L)
                z = jnp.broadcast_to(z, (T,) + z.shape[1:])
            x = jnp.concatenate([x, z], axis=-1)

        # Preprocess:
        for feature_size in self.preprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = nn.LayerNorm()(x)
            x = self.act_fn(x)

        # LSTM stack (time-major T):
        new_cs, new_hs = [], []
        for i, feature_size in enumerate(self.lstm_features):
            layer = self.lstm(feature_size)
            layer_carry = jax.tree.map(lambda c: c[i], carry)
            rnn_in = (x, resets)
            layer_carry, x = layer(layer_carry, rnn_in)
            new_c, new_h = layer_carry
            new_cs.append(new_c)
            new_hs.append(new_h)

        # Postprocess:
        embedding = x
        for feature_size in self.postprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = self.act_fn(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        logits = nn.Dense(self.action_dim)(x)  # (T,B,A)
        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_carry, logits, embedding

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
        x, resets = x
        if z is not None:
            T = x.shape[0]
            if z.ndim == 2:
                z = jnp.broadcast_to(z[None, ...], (T,) + z.shape)
            elif z.ndim == 3 and z.shape[0] == 1 and T != 1:
                z = jnp.broadcast_to(z, (T,) + z.shape[1:])
            x = jnp.concatenate([x, z], axis=-1)

        for feature_size in self.preprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = nn.LayerNorm()(x)
            x = self.act_fn(x)

        new_cs, new_hs = [], []
        for i, feature_size in enumerate(self.lstm_features):
            layer = self.lstm(feature_size)
            layer_carry = jax.tree.map(lambda c: c[i], carry)
            layer_carry, x = layer(layer_carry, (x, resets))
            new_c, new_h = layer_carry
            new_cs.append(new_c)
            new_hs.append(new_h)

        embedding = x
        for feature_size in self.postprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = self.act_fn(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # IMPORTANT: match rl_vae training
        logits = nn.Dense(self.action_dim, name="policy_out")(x)

        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_carry, logits, embedding

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return zeros_lstm_carry(self.lstm_features, batch_size)


# -------------------------
# BC adapter
# -------------------------
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
        # MultiLayerLstm expects (B,T,D). You currently send (T,B,D) and swap to (B,1,D) — that’s fine:
        x_bt = jnp.swapaxes(obs, 0, 1)  # (B,1,D)
        carry, logits_bt = self.net(x=x_bt, carry=carry, training=training)  # (B,1,A)
        logits = jnp.swapaxes(logits_bt, 0, 1)  # (1,B,A) to match your caller
        return carry, logits, logits

    @nn.nowrap
    def initialize_carry(self, batch_size):
        # ✅ Use the real carry from MultiLayerLstm
        return self.net.initialize_carry(batch_size)


class IPPOAdapter(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, carry, x, z: jnp.ndarray = None, training: bool = False):
        # x may be (obs, resets) or (obs, resets, avail); we only need obs (+ optional avail)
        if len(x) == 2:
            obs, _resets = x
            avail = None
        else:
            obs, _resets, avail = x  # obs: (1,B,D), avail: (1,B,A)

        # Flatten time dimension (T=1) -> (B,D)
        obs_b = obs.squeeze(0)

        # 2-layer MLP + logits (match IPPO rollout architecture)
        h = nn.Dense(512)(obs_b)
        h = nn.relu(h)
        h = nn.Dense(512)(h)
        h = nn.relu(h)
        logits = nn.Dense(self.action_dim)(h)  # (B,A)

        # If avail passed, mask here to keep behavior identical to rollout
        if avail is not None:
            logits = logits - (1.0 - avail.squeeze(0)) * 1e10

        # Restore (T=1,B,A). carry is stateless -> passthrough
        logits = logits[None, ...]
        return carry, logits, logits  # embedding stub = logits

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return None


class OBLAdapter(nn.Module):
    action_dim: int
    publ_offset: int = 125  # public features are obs[..., :publ_offset]

    @nn.compact
    def __call__(self, carry, x, z: jnp.ndarray = None, training: bool = False):
        # x: (obs, resets) or (obs, resets, avail)
        if len(x) == 2:
            obs, _resets = x
            avail = None
        else:
            obs, _resets, avail = x

        # Expect time-major (T,B,D); your harness uses T=1
        obs = jnp.asarray(obs)
        if obs.ndim == 2:  # (B,D) -> (1,B,D)
            obs = obs[None, ...]

        # T=1 path
        obs_b = obs.squeeze(0)  # (B,D)

        priv_s = obs_b
        publ_s = obs_b[..., self.publ_offset :]

        core = OBLR2D2(out_dim=self.action_dim)
        carry, logits_ba = core(carry, (priv_s, publ_s))  # logits: (B,A)

        # Optional in-network masking (you also mask outside, so this is optional)
        if avail is not None:
            logits_ba = logits_ba - (1.0 - avail.squeeze(0)) * 1e10

        logits = logits_ba[None, ...]  # (1,B,A) to match rest of harness
        return carry, logits, logits

    @nn.nowrap
    def initialize_carry(self, batch_size: int):
        # initialize_carry signature expects rng + batch_dims
        core = OBLR2D2(out_dim=self.action_dim)
        return core.initialize_carry(jax.random.PRNGKey(0), (batch_size,))


class BRAdapter(nn.Module):
    action_dim: int
    fc_dim_size: int
    gru_hidden_dim: int

    @nn.compact
    def __call__(self, hidden, x, z: jnp.ndarray = None, training: bool = False):
        # x: (obs, dones) or (obs, dones, avail)
        if len(x) == 2:
            obs, dones = x
            avail = None
        else:
            obs, dones, avail = x  # obs:(T,B,D), dones:(T,B), avail:(T,B,A) with T=1 in harness

        # Ensure time axis exists (your harness uses T=1, but be robust)
        obs = jnp.asarray(obs)
        dones = jnp.asarray(dones)

        if obs.ndim == 2:  # (B,D) -> (1,B,D)
            obs = obs[None, ...]
        if dones.ndim == 1:  # (B,)  -> (1,B)
            dones = dones[None, ...]

        T, B = obs.shape[0], obs.shape[1]

        if avail is None:
            avail = jnp.ones((T, B, self.action_dim), dtype=jnp.float32)
        else:
            avail = jnp.asarray(avail)
            if avail.ndim == 2:  # (B,A) -> (1,B,A)
                avail = avail[None, ...]

        # Cast dones to bool for reset semantics
        dones = dones.astype(jnp.bool_)

        # Minimal config for ActorCriticRNN
        cfg = type(
            "Cfg", (), {"fc_dim_size": self.fc_dim_size, "gru_hidden_dim": self.gru_hidden_dim}
        )()
        core = ActorCriticRNN(action_dim=self.action_dim, config=cfg)

        # IMPORTANT: pass time-major tensors to match training / ScannedRNN expectations
        new_hidden, pi, _v = core(hidden, (obs, dones, avail))

        # pi logits are (T,B,A); your harness expects (1,B,A) already
        logits = pi.logits_parameter()  # (T,B,A)
        return new_hidden, logits, logits

    @nn.nowrap
    def initialize_carry(self, batch_size: int):
        return jnp.zeros((batch_size, self.gru_hidden_dim), dtype=jnp.float32)


# -------------------------
# Agent wrapper
# -------------------------
class Agent:
    def __init__(
        self,
        actor_module: ActorLstm,
        actor_vars,  # freeze({"params": <actor>})
        z: jnp.ndarray,  # (L,)
        num_agents: int,
        action_dim: int,
    ):
        self.actor = actor_module
        self.vars = freeze({"params": unfreeze(actor_vars)["params"]})  # normalize
        self.z = z
        self.num_agents = num_agents
        self.action_dim = action_dim

    def initialize_hidden_state(self):
        if isinstance(self.actor, (ActorLstm, ActorLstmPolicyOut)):
            return self.actor.initialize_carry(self.num_agents)
        if isinstance(self.actor, ActorLstmBCAdapter):
            bound = self.actor.bind(self.vars)
            return bound.net.initialize_carry(self.num_agents)
        if isinstance(self.actor, IPPOAdapter):
            return None
        if isinstance(self.actor, BRAdapter):
            return self.actor.initialize_carry(self.num_agents)
        if isinstance(self.actor, OBLAdapter):
            return self.actor.initialize_carry(self.num_agents)
        return None

    def act(
        self,
        obs_dict,
        done_scalar,
        legal_moves_dict,
        curr_player,
        rng,
        hidden_state,
        eval_mode: str = "greedy",
    ):
        # Prepare batched inputs over agents
        agents = list(obs_dict.keys())
        B = len(agents)
        obs_batch = jnp.stack([obs_dict[a] for a in agents])[None, ...]  # (1,B,D)
        resets = jnp.repeat(jnp.array(done_scalar)[None, None], B, axis=1)  # (1,B)
        avail = jnp.stack([legal_moves_dict[a] for a in agents])[None, ...]  # (1,B,A)

        # z batch
        z_batch = jnp.repeat(self.z[None, :], B, axis=0)  # (B,L)

        # Forward
        if isinstance(self.actor, (IPPOAdapter, BRAdapter)):
            # For IPPO we also pass avail actions so masking happens in-network
            hidden_state, logits, _ = self.actor.apply(
                self.vars, hidden_state, (obs_batch, resets, avail), z=None, training=False
            )
        else:
            z_arg = None if self.z.size == 0 else z_batch
            hidden_state, logits, _ = self.actor.apply(
                self.vars, hidden_state, (obs_batch, resets), z=z_arg, training=False
            )

        logits = logits  # (1,B,A)

        # Mask illegal actions and pick greedy
        action_logits = logits - (1.0 - avail) * 1e10

        if eval_mode == "greedy":
            action = jnp.argmax(action_logits, axis=-1)  # (1,B)
        else:  # sample
            action = distrax.Categorical(logits=action_logits).sample(seed=rng)
        action = action.squeeze(0)  # (B,)

        # Return per-agent actions as a JAX array (B,), and new hidden state
        return action.astype(jnp.int32), hidden_state


# -------------------------
# Loading helpers for the two policy types
# -------------------------
def build_actor_module(args, action_dim: int, act_fn_name: str = None):
    return ActorLstm(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=act_fn_name or args.act_fn,
    )


def build_actor_module_policy_out(args, action_dim: int, act_fn_name: str = None):
    return ActorLstmPolicyOut(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=act_fn_name or args.act_fn,
    )


def build_br_actor_module(action_dim: int, fc_dim_size: int, gru_hidden_dim: int):
    return BRAdapter(action_dim=action_dim, fc_dim_size=fc_dim_size, gru_hidden_dim=gru_hidden_dim)


def build_obl_actor_module(action_dim: int, publ_offset: int = 125):
    return OBLAdapter(action_dim=action_dim, publ_offset=publ_offset)


def build_bc_actor_module(args, action_dim: int, act_fn_name: str = None):
    return ActorLstmBCAdapter(
        preprocessing_features=args.preprocessing_features,
        lstm_features=args.lstm_features,
        postprocessing_features=args.postprocessing_features,
        action_dim=action_dim,
        dropout_rate=args.dropout,
        activation_fn_name=act_fn_name or args.act_fn,
    )


def load_actor_vars_for_bc(params_path: str):
    """BC checkpoints save the MultiLayerLstm directly under 'params'."""
    full_vars = load_flax_safetensors(params_path)
    p = unfreeze(full_vars)
    assert "params" in p, f"'params' not found in {params_path}"
    return freeze({"params": {"net": p["params"]}})


def load_actor_vars_for_rl(params_path: str):
    """Load the RL checkpoint and return the actor subtree as Flax variables."""
    full_vars = load_flax_safetensors(params_path)  # {'params': ...}
    p = unfreeze(full_vars)
    assert "params" in p and "actor" in p["params"], (
        f"'actor' not found in params of {params_path}"
    )
    actor_params = freeze(p["params"]["actor"])
    return freeze({"params": actor_params})


def load_actor_vars_for_br(params_path: str):
    """
    BR params were saved as a raw param tree (train_state.params) flattened with sep=','.
    When used as a submodule inside BRAdapter, Flax expects them under ActorCriticRNN_0.
    """
    full_vars = load_flax_safetensors(params_path)  # returns {'params': ...}
    params = unfreeze(full_vars["params"])

    # If it already has ActorCriticRNN_0, leave it
    if "ActorCriticRNN_0" not in params:
        params = {"ActorCriticRNN_0": params}

    return freeze({"params": params})


def ippo_load_checkpoint(checkpoint_path: str):
    import pickle

    with open(checkpoint_path, "rb") as f:
        params = pickle.load(f)
    # Optional legacy key remap
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


def build_ippo_actor_module(action_dim: int):
    return IPPOAdapter(action_dim=action_dim)


def load_actor_vars_for_ippo(pickle_path: str):
    """IPPO checkpoints are plain Flax param dicts pickled during training."""
    return ippo_load_checkpoint(pickle_path)


def infer_z_for_rl(
    params_path: str,
    z_npy: str = None,
    yaml_with_z: str = None,
    prior_json: str = None,
    z_seed: int = None,
    use_prior_mean: bool = False,
):
    """Order: npy -> yaml 'sampled_z' -> prior mean -> sample(prior, seed)."""
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
        "Could not determine z for RL policy. Provide one of: "
        "--pol*_z_npy | --pol*_yaml_with_z | --pol*_use_prior_mean | (--prior_json & --pol*_z_seed)."
    )


def infer_z_for_vae(
    prior_json: str, z_npy: str = None, z_seed: int = None, use_prior_mean: bool = False
):
    """Order: npy -> prior mean -> sample(prior, seed)."""
    if z_npy and os.path.exists(z_npy):
        return jnp.array(np.load(z_npy))
    if use_prior_mean:
        if prior_json is None:
            raise ValueError("use_prior_mean=True requires --prior_json.")
        return z_from_prior_mean(prior_json)
    if z_seed is None:
        raise ValueError(
            "For VAE policy, provide one of: --pol*_z_npy | --pol*_use_prior_mean | --pol*_z_seed (with --prior_json)."
        )
    if prior_json is None:
        raise ValueError("Sampling z requires --prior_json for the GMM prior.")
    rng = jax.random.PRNGKey(int(z_seed))
    return sample_z_from_prior(rng, prior_json)


def load_actor_vars_for_vae(vae_checkpoint_path: str):
    """Load VAE checkpoint and take its decoder subtree as actor params."""
    vae_vars = load_flax_safetensors(vae_checkpoint_path)
    decoder_params, _ = extract_decoder_params(vae_vars)
    return freeze({"params": decoder_params})


# -------------------------
# Cross-play harness (vectorized over rollouts)
# -------------------------
def get_agents(args, env, action_dim: int):
    agents = []

    # Prepare the two policies
    for idx in (1, 2):
        pre = (
            getattr(args, f"pol{idx}_preprocessing_features", None) or args.preprocessing_features
        )
        lstm = getattr(args, f"pol{idx}_lstm_features", None) or args.lstm_features
        post = (
            getattr(args, f"pol{idx}_postprocessing_features", None)
            or args.postprocessing_features
        )

        act_fn_name = getattr(args, f"pol{idx}_act_fn") or args.act_fn
        ptype = getattr(args, f"pol{idx}_type")

        actor_mod = ActorLstm(
            preprocessing_features=pre,
            lstm_features=lstm,
            postprocessing_features=post,
            action_dim=action_dim,
            dropout_rate=args.dropout,
            activation_fn_name=act_fn_name,
        )

        if ptype in ("rl", "rl_bc", "rl_vae"):
            params_path = getattr(args, f"pol{idx}_params")
            if params_path is None:
                raise ValueError(f"--pol{idx}_params required for {ptype} policy")

            actor_vars = load_actor_vars_for_rl(params_path)
            actor_params = actor_vars["params"]  # frozen

            # Pick correct module head naming based on checkpoint
            if actor_uses_policy_out(actor_params):
                actor_mod = ActorLstmPolicyOut(
                    preprocessing_features=pre,
                    lstm_features=lstm,
                    postprocessing_features=post,
                    action_dim=action_dim,
                    dropout_rate=args.dropout,
                    activation_fn_name=act_fn_name,
                )
            else:
                actor_mod = ActorLstm(
                    preprocessing_features=pre,
                    lstm_features=lstm,
                    postprocessing_features=post,
                    action_dim=action_dim,
                    dropout_rate=args.dropout,
                    activation_fn_name=act_fn_name,
                )

            if ptype == "rl":
                z = infer_z_for_rl(
                    params_path=params_path,
                    z_npy=getattr(args, f"pol{idx}_z_npy"),
                    yaml_with_z=getattr(args, f"pol{idx}_yaml_with_z"),
                    prior_json=args.prior_json,
                    z_seed=getattr(args, f"pol{idx}_z_seed"),
                    use_prior_mean=getattr(args, f"pol{idx}_use_prior_mean"),
                )
            elif ptype == "rl_vae":
                # same as rl, but z required in your setup
                z = infer_z_for_rl(
                    params_path=params_path,
                    z_npy=getattr(args, f"pol{idx}_z_npy"),
                    yaml_with_z=getattr(args, f"pol{idx}_yaml_with_z"),
                    prior_json=args.prior_json,
                    z_seed=getattr(args, f"pol{idx}_z_seed"),
                    use_prior_mean=getattr(args, f"pol{idx}_use_prior_mean"),
                )
            else:  # rl_bc
                z = jnp.zeros((0,), dtype=jnp.float32)

        elif ptype == "vae":
            vae_ckpt = getattr(args, f"pol{idx}_vae_ckpt")
            if vae_ckpt is None:
                raise ValueError(f"--pol{idx}_vae_ckpt required for VAE policy")
            actor_vars = load_actor_vars_for_vae(vae_ckpt)

            actor_mod = ActorLstm(
                preprocessing_features=pre,
                lstm_features=lstm,
                postprocessing_features=post,
                action_dim=action_dim,
                dropout_rate=args.dropout,
                activation_fn_name=act_fn_name,
            )

            z = infer_z_for_vae(
                prior_json=args.prior_json,
                z_npy=getattr(args, f"pol{idx}_z_npy"),
                z_seed=getattr(args, f"pol{idx}_z_seed"),
                use_prior_mean=getattr(args, f"pol{idx}_use_prior_mean"),  # <-- and this
            )
        elif ptype == "bc":
            actor_mod = build_bc_actor_module(args, action_dim, act_fn_name)
            params_path = getattr(args, f"pol{idx}_bc_params")
            if params_path is None:
                raise ValueError(f"--pol{idx}_bc_params required for BC policy")
            actor_vars = load_actor_vars_for_bc(params_path)
            # BC doesn't use z; give a zero-length placeholder to satisfy Agent API.
            z = jnp.zeros((0,), dtype=jnp.float32)
        elif ptype == "ippo":
            ippo_ckpt = getattr(args, f"pol{idx}_ippo_ckpt")
            if ippo_ckpt is None:
                raise ValueError(f"--pol{idx}_ippo_ckpt required for IPPO policy")
            actor_mod = build_ippo_actor_module(action_dim)
            actor_vars = load_actor_vars_for_ippo(ippo_ckpt)
            # IPPO doesn't use z
            z = jnp.zeros((0,), dtype=jnp.float32)
        elif ptype == "br":
            params_path = getattr(args, f"pol{idx}_br_params")
            if params_path is None:
                raise ValueError(f"--pol{idx}_br_params required for BR policy")

            actor_mod = build_br_actor_module(
                action_dim=action_dim,
                fc_dim_size=args.br_fc_dim_size,
                gru_hidden_dim=args.br_gru_hidden_dim,
            )
            actor_vars = load_actor_vars_for_br(params_path)
            z = jnp.zeros((0,), dtype=jnp.float32)
        elif ptype == "obl":
            params_path = getattr(args, f"pol{idx}_obl_params")
            if params_path is None:
                raise ValueError(f"--pol{idx}_obl_params required for OBL policy")

            actor_mod = build_obl_actor_module(action_dim, publ_offset=args.obl_publ_offset)
            actor_vars = load_actor_vars_for_obl(params_path)
            z = jnp.zeros((0,), dtype=jnp.float32)
        else:
            raise ValueError(f"Unknown policy type for pol{idx}: {ptype}")

        agents.append(
            Agent(actor_mod, actor_vars, z, num_agents=len(env.agents), action_dim=action_dim)
        )

    return agents


def play_game(rng, args, env):
    agents = get_agents(args, env, action_dim=args.action_dim)

    hidden_states = []
    for agent in agents:
        if hasattr(agent, "initialize_hidden_state"):
            hidden_state = agent.initialize_hidden_state()
        else:
            hidden_state = None
        hidden_states.append(hidden_state)

    use_jit = args.use_jit if args.use_jit is not None else True
    with jax.disable_jit(not use_jit):
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng)
        legal_moves = env.get_legal_moves(unwrap_state(env_state))

        @jax.jit
        def _step_env(rng, env_state, actions):
            rng, _rng = jax.random.split(rng)
            new_obs, new_env_state, reward, dones, infos = env.step(_rng, env_state, actions)
            new_legal_moves = env.get_legal_moves(unwrap_state(new_env_state))
            return rng, new_env_state, new_obs, reward, dones, new_legal_moves

        _done = False
        cum_rew = 0

        def cond_fn(val):
            _, _, done, _, _, _, _, _ = val
            return jnp.logical_not(done)

        def body_fn(val):

            cum_rew, t, done, rng, env_state, obs, legal_moves, hidden_states = val
            rng, _rng = jax.random.split(rng)
            curr_player = jnp.argmax(unwrap_state(env_state).cur_player_idx)  # 0..B-1

            # One RNG per agent for sampling mode
            agent_rngs = jax.random.split(_rng, len(env.agents))

            actions_all = []
            updated_hidden_states = list(hidden_states)

            _B = len(env.agents)

            for i in range(len(env.agents)):
                agent = agents[i]
                old_carry = hidden_states[i]

                # forward
                action_list, provisional_new_carry = agent.act(
                    obs,
                    done,
                    legal_moves,
                    curr_player,
                    agent_rngs[i],
                    old_carry,
                    eval_mode=args.eval_mode,
                )
                actions_all.append(action_list)

                updated_hidden_states[i] = provisional_new_carry

            def take_from_pol(pidx):
                return {
                    agent_name: jnp.array(actions_all[pidx][k])
                    for k, agent_name in enumerate(env.agents)
                }

            actions = jax.lax.cond(
                curr_player == 1,
                lambda _: take_from_pol(1),
                lambda _: take_from_pol(0),
                operand=None,
            )

            rng, env_state, obs, reward, dones, legal_moves = _step_env(rng, env_state, actions)

            done = dones["__all__"]
            cum_rew = cum_rew + reward["__all__"]
            t = t + 1
            return (
                cum_rew,
                t,
                done,
                rng,
                env_state,
                obs,
                legal_moves,
                tuple(updated_hidden_states),
            )

        init_val = (0.0, 0, False, rng, env_state, obs, legal_moves, tuple(hidden_states))

        cum_rew, _, _, _, final_env_state, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)

        return cum_rew


def play_games(rng_keys, args, env):
    play_game_vec = jax.vmap(lambda r: play_game(r, args, env), in_axes=(0,))
    results = play_game_vec(rng_keys)
    mean = jnp.mean(results)
    sem = jnp.std(results) / jnp.sqrt(results.shape[0])
    return mean, sem


def main():
    parser = argparse.ArgumentParser(description="Cross-play evaluation (RL / VAE / BC).")

    # Env
    parser.add_argument("--num_players", type=int, default=2)
    parser.add_argument("--num_rollouts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_jit", type=int, default=1)

    # Network architecture (must match checkpoints)
    parser.add_argument("--preprocessing_features", type=int, nargs="+", default=[1024])
    parser.add_argument("--lstm_features", type=int, nargs="+", default=[512])
    parser.add_argument("--postprocessing_features", type=int, nargs="+", default=[256])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--act_fn",
        type=str,
        choices=["relu", "gelu"],
        default="relu",
        help="Global fallback activation; overridden by --pol{1,2}_act_fn.",
    )
    parser.add_argument("--pol1_act_fn", type=str, choices=["relu", "gelu"], default=None)
    parser.add_argument("--pol2_act_fn", type=str, choices=["relu", "gelu"], default=None)

    # Optional per-policy overrides (if not provided, fall back to global values)
    parser.add_argument("--pol1_preprocessing_features", type=int, nargs="+", default=None)
    parser.add_argument("--pol1_lstm_features", type=int, nargs="+", default=None)
    parser.add_argument("--pol1_postprocessing_features", type=int, nargs="+", default=None)

    parser.add_argument("--pol2_preprocessing_features", type=int, nargs="+", default=None)
    parser.add_argument("--pol2_lstm_features", type=int, nargs="+", default=None)
    parser.add_argument("--pol2_postprocessing_features", type=int, nargs="+", default=None)

    # BR specific
    parser.add_argument("--br_fc_dim_size", type=int, default=128)
    parser.add_argument("--br_gru_hidden_dim", type=int, default=128)

    # Prior (needed for sampling z for RL/VAE)
    parser.add_argument("--prior_json", type=str, default=None)

    # Policy 1 spec
    parser.add_argument(
        "--pol1_type",
        type=str,
        choices=["rl", "rl_vae", "vae", "bc", "ippo", "rl_bc", "br", "obl"],
        required=True,
    )

    # RL or VAE
    parser.add_argument("--pol1_use_prior_mean", action="store_true", default=False)
    parser.add_argument("--pol2_use_prior_mean", action="store_true", default=False)

    # RL or VAE
    parser.add_argument("--pol1_yaml_with_z", type=str, default=None)
    parser.add_argument("--pol1_z_npy", type=str, default=None)
    parser.add_argument("--pol1_z_seed", type=int, default=None)

    # RL specific
    parser.add_argument("--pol1_params", type=str, default=None)

    # VAE specific
    parser.add_argument("--pol1_vae_ckpt", type=str, default=None)

    # BR specific
    parser.add_argument("--pol1_br_params", type=str, default=None)

    # OBL specific
    parser.add_argument("--pol1_obl_params", type=str, default=None)

    # BC specific
    parser.add_argument("--pol1_bc_params", type=str, default=None)

    # Policy 2 spec
    parser.add_argument(
        "--pol2_type",
        type=str,
        choices=["rl", "rl_vae", "vae", "bc", "ippo", "rl_bc", "br", "obl"],
        required=True,
    )

    # RL specific
    parser.add_argument("--pol2_params", type=str, default=None)

    # VAE specific
    parser.add_argument("--pol2_vae_ckpt", type=str, default=None)

    # BR specific
    parser.add_argument("--pol2_br_params", type=str, default=None)

    # OBL specific
    parser.add_argument("--pol2_obl_params", type=str, default=None)
    parser.add_argument("--obl_publ_offset", type=int, default=125)

    # IPPO specific
    parser.add_argument("--pol1_ippo_ckpt", type=str, default=None)
    parser.add_argument("--pol2_ippo_ckpt", type=str, default=None)

    # RL or VAE
    parser.add_argument("--pol2_yaml_with_z", type=str, default=None)
    parser.add_argument("--pol2_z_npy", type=str, default=None)
    parser.add_argument("--pol2_z_seed", type=int, default=None)

    # BC
    parser.add_argument("--pol2_bc_params", type=str, default=None)

    # Eval mode
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["greedy", "sample"],
        default="greedy",
        help="greedy: argmax over masked logits; sample: draw from Categorical(logits).",
    )

    args = parser.parse_args()

    # Build environment & infer dims
    hand_size = 5 if args.num_players <= 3 else 4
    use_raw_obs = any(t in ("bc", "obl") for t in (args.pol1_type, args.pol2_type))

    base_env = make("hanabi", hand_size=hand_size, num_agents=args.num_players)
    env = base_env if use_raw_obs else LogWrapper(base_env)

    rng0 = jax.random.PRNGKey(args.seed)
    _rng = jax.random.split(rng0, 1)[0]
    obs0, env_state0 = env.reset(_rng)
    legal0 = env.get_legal_moves(unwrap_state(env_state0))
    first_agent = list(env.agents)[0]
    args.action_dim = int(legal0[first_agent].shape[-1])

    # inside main(), right after building env and inferring args.action_dim:
    if args.pol1_type == "bc" and args.pol1_bc_params:
        _raw_loaded, _expected = ck_bc_params_match(env, args, args.pol1_bc_params)
        bc_forward_equivalence_check(env, args, args.pol1_bc_params)

    # Rollouts
    rng = jax.random.PRNGKey(args.seed)
    rng_keys = jax.random.split(rng, args.num_rollouts)
    mean, sem = play_games(rng_keys, args, env)
    print(
        f"Cross-play result over {args.num_rollouts} rollouts: {float(mean):.3f} ± {float(sem):.3f}"
    )


if __name__ == "__main__":
    main()
