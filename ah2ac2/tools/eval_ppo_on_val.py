#!/usr/bin/env python3
"""
Evaluate a trained PPO model on the validation set.

Usage
-----
python ah2ac2/tools/eval_ppo_on_val.py \
  --ppo_params_path checkpoints/ppo/human_ppo/2k/epoch45_seed0_score0.404_2p.safetensors \
  --val_data_path ah2ac2/datasets/data/2_player_games_val.safetensors

# Optional:
#   --config_path path/to/config.yaml
#   --seed 0
"""

import argparse
import glob
import os
from typing import List, NamedTuple, TypedDict

import chex
import distrax
import flax.linen as nn
import flax.training.train_state
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import struct
from flax.core import freeze, unfreeze
from flax.traverse_util import unflatten_dict
from jaxmarl import make
from jaxmarl.environments.hanabi import hanabi_game
from jaxmarl.environments.hanabi.hanabi import HanabiEnv
from omegaconf import OmegaConf

from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset
from ah2ac2.nn.multi_layer_lstm import LstmCellWithHiddenStateReset

"""
e.g.

python ah2ac2/tools/eval_ppo_on_val.py \
--ppo_params_path checkpoints/kl_anchored/human_1.5k_valacc0.480/kl_target0.25/seed1_score21.344_klt0.25_2p.safetensors \
--val_data_path ah2ac2/datasets/data/2_player_games_train_val1000.safetensors
"""


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def batchify(x, agent_list):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((len(agent_list), -1))


def unbatchify(x, agent_list):
    return {a: x[i] for i, a in enumerate(agent_list)}


def zeros_lstm_carry(lstm_features: List[int], batch_size: int):
    """Create zero carry for a stacked LSTM: (Cs, Hs) with shape (num_layers, B, hidden)."""
    cs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in lstm_features]
    hs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in lstm_features]
    return jnp.stack(cs), jnp.stack(hs)


def actor_has_adv_out(actor_params_tree) -> bool:
    p = unfreeze(actor_params_tree)

    def has_key(d):
        if not isinstance(d, dict):
            return False
        if "adv_out" in d:
            return True
        return any(has_key(v) for v in d.values() if isinstance(v, dict))

    return has_key(p)


# --------------------------------------------------------------------------- #
# Trajectory factory (same as BC eval script)
# --------------------------------------------------------------------------- #
class StepCarry(NamedTuple):
    rng: chex.PRNGKey  # Random number generator.
    prev_obs: dict  # Observation at previous timestep.
    prev_env_state: hanabi_game.State  # Current state of the environment.


class Transition(NamedTuple):
    observations: chex.Array
    actions: chex.Array
    legal_actions: chex.Array


def make_trajectory_batcher(num_players: int):
    hand_size = 5 if num_players <= 3 else 4
    env: HanabiEnv = make("hanabi", hand_size=hand_size, num_agents=num_players)

    def get_trajectory_batch(
        rng: chex.PRNGKey,
        deck: chex.Array,
        actions: chex.Array,
    ):
        def _step(carry: StepCarry, step_actions: chex.Array):
            step_rng, _step_rng = jax.random.split(carry.rng)

            # Step env with selected actions.
            env_act = unbatchify(step_actions, env.agents)
            obs, env_state, reward, dones, infos = env.step_env(
                _step_rng, carry.prev_env_state, env_act
            )

            # Stay in terminal state if one is reached.
            reached_terminal = carry.prev_env_state.terminal
            env_state = jax.lax.cond(
                reached_terminal, lambda: carry.prev_env_state, lambda: env_state
            )
            obs = jax.lax.cond(reached_terminal, lambda: carry.prev_obs, lambda: obs)

            # Batchify observations, actions and legal actions.
            legal_moves = env.get_legal_moves(carry.prev_env_state)
            legal_moves_batched = batchify(legal_moves, env.agents)
            prev_obs_batched = batchify(carry.prev_obs, env.agents)
            act_batched = batchify(env_act, env.agents)

            # Create the carry/transition.
            new_carry = StepCarry(prev_obs=obs, prev_env_state=env_state, rng=step_rng)
            new_transition = Transition(
                observations=prev_obs_batched,  # (num_agents, obs_dim)
                actions=act_batched,  # (num_agents, 1)
                legal_actions=legal_moves_batched,  # (num_agents, num_moves)
            )
            return new_carry, new_transition

        rng, _rng = jax.random.split(rng)
        init_obs, init_env_state = env.reset_from_deck_of_pairs(deck)
        init_carry = StepCarry(prev_obs=init_obs, prev_env_state=init_env_state, rng=_rng)
        _, trajectory_batch = jax.lax.scan(_step, init_carry, actions)
        return trajectory_batch  # (num_steps, num_players, -1)

    return get_trajectory_batch


@jax.jit
def transform_trajectory_batch(trajectory_batch, game_len_mask):
    """
    Trajectory batch is expected to be a PyTree of arrays with the following structure:
    Objects have shape: (batch_size, max_game_len, num_players, features)
    We transform to:    (batch_size * num_players, max_game_len, features)

    Game len mask shape:
        in:  (batch_size, max_game_len)
        out: (batch_size * num_players, max_game_len)
    """
    # Transform the trajectory batch data.
    obs_batch, act_batch, legal_moves_batch = jax.tree.map(
        lambda x: jnp.swapaxes(
            x,
            1,
            2,  # Swap max_game_len and num_players axes.
        ).reshape(-1, x.shape[1], x.shape[-1]),
        trajectory_batch,
    )
    # Repeat game len mask for each player.
    game_len_mask_batch = jnp.repeat(game_len_mask, trajectory_batch[0].shape[2], axis=0)
    return obs_batch, act_batch, legal_moves_batch, game_len_mask_batch


# --------------------------------------------------------------------------- #
# PPO network definition (as provided)
# --------------------------------------------------------------------------- #
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
        if self.activation_fn_name == "relu":
            self.act_fn = nn.relu
        elif self.activation_fn_name == "gelu":
            self.act_fn = nn.gelu
        else:
            self.act_fn = nn.relu

    @nn.compact
    def __call__(self, carry, x, z: jnp.ndarray = None, training: bool = False):
        # x: (obs, resets), obs: (T,B,D), resets: (T,B) with T=1 in our usage
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

        # LSTM stack over time dimension (T):
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
        embedding = x  # (T,B,H)
        for feature_size in self.postprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = self.act_fn(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        policy_logits = nn.Dense(self.action_dim, name="policy_out")(x)  # logits (T,B,A)
        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))

        # --- Advantage head (for Q-hat): produce A(s,a) with zero-mean across actions ---
        adv = nn.Dense(self.action_dim, name="adv_out")(embedding)  # (T,B,A)
        adv = adv - jnp.mean(adv, axis=-1, keepdims=True)  # zero-mean over actions

        return new_carry, policy_logits, embedding, adv

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return zeros_lstm_carry(self.lstm_features, batch_size)


class ActorLstmNoAdv(nn.Module):
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
        if self.activation_fn_name == "relu":
            self.act_fn = nn.relu
        elif self.activation_fn_name == "gelu":
            self.act_fn = nn.gelu
        else:
            self.act_fn = nn.relu

    @nn.compact
    def __call__(self, carry, x, z: jnp.ndarray = None, training: bool = False):
        # x: (obs, resets), obs: (T,B,D), resets: (T,B) with T=1 in our usage
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

        # LSTM stack over time dimension (T):
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
        embedding = x  # (T,B,H)
        for feature_size in self.postprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = self.act_fn(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        policy_logits = nn.Dense(self.action_dim, name="policy_out")(x)  # (T,B,A)
        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))

        # No adv head in older checkpoints; return zeros with correct shape.
        adv = jnp.zeros_like(policy_logits)  # (T,B,A)

        return new_carry, policy_logits, embedding, adv

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return zeros_lstm_carry(self.lstm_features, batch_size)


class ActorCriticRNN(nn.Module):
    actor: ActorLstm
    critic_features: int

    @nn.compact
    def __call__(self, hidden, x, z_batch: jnp.ndarray = None):
        obs, dones, avail_actions = x  # obs: (T,B,D), dones: (T,B), avail_actions: (T,B,A)

        # Actor:
        hidden, actor_logits, embedding, adv = self.actor(hidden, (obs, dones), z=z_batch)
        action_logits = actor_logits - (1.0 - avail_actions) * 1e10
        pi = distrax.Categorical(logits=action_logits)

        # Critic (on actor embedding):
        critic = nn.Dense(self.critic_features)(embedding)
        critic = nn.relu(critic)
        value = nn.Dense(1)(critic)
        value = jnp.squeeze(value, axis=-1)  # (T,B)

        return hidden, pi, value, adv  # adv gives per-action A_hat

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return self.actor.initialize_carry(batch_size)


# --------------------------------------------------------------------------- #
# Training utilities needed for evaluation (model + state + metrics)
# --------------------------------------------------------------------------- #
class TrainState(flax.training.train_state.TrainState):
    key: jax.Array = struct.field(pytree_node=False)


class EvaluationMetrics(TypedDict):
    loss: float
    cross_entropy: float
    accuracy: float
    non_noop_accuracy: float


def masked_cross_entropy_loss(logits, act, mask):
    """
    logits: (B,T,A), act: (B,T,1), mask: (B,T)
    """
    act = act.squeeze(axis=-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, act)
    masked_loss = loss * mask
    return masked_loss.sum() / mask.sum()


def compute_metrics(logits, actions, mask) -> EvaluationMetrics:
    actions_squeezed = actions.squeeze(axis=-1)
    correct = jnp.argmax(logits, -1) == actions_squeezed

    # Full accuracy.
    correct_masked = correct * mask
    accuracy = correct_masked.sum() / mask.sum()

    # Non-noop accuracy.
    non_noop_mask = (actions_squeezed != actions.max()) * mask
    correct_non_noop_masked = correct * non_noop_mask
    non_noop_accuracy = correct_non_noop_masked.sum() / non_noop_mask.sum()

    # Cross-entropy (masked over valid timesteps)
    ce = masked_cross_entropy_loss(logits, actions, mask)

    return EvaluationMetrics(
        loss=ce,
        cross_entropy=ce,
        accuracy=accuracy,
        non_noop_accuracy=non_noop_accuracy,
    )


def make_model(config, action_dim: int, actor_cls=ActorLstm) -> ActorCriticRNN:
    net_cfg = config.network
    actor = actor_cls(
        preprocessing_features=list(net_cfg.preprocessing_features),
        lstm_features=list(net_cfg.lstm_features),
        postprocessing_features=list(net_cfg.postprocessing_features),
        action_dim=action_dim,
        dropout_rate=net_cfg.dropout,
        activation_fn_name=net_cfg.act_fn,
    )
    return ActorCriticRNN(actor=actor, critic_features=net_cfg.critic_features)


def get_train_state(
    rng, batch_size: int, obs_dim: int, action_dim: int, config, actor_cls=ActorLstm
):
    rng, _rng = jax.random.split(rng)

    model = make_model(config, action_dim=action_dim, actor_cls=actor_cls)

    # Dummy init obs with T=1 just to create parameters.
    dummy_obs = jnp.ones((1, batch_size, obs_dim), dtype=jnp.float32)
    dummy_dones = jnp.zeros((1, batch_size), dtype=jnp.float32)
    dummy_avail = jnp.ones((1, batch_size, action_dim), dtype=jnp.float32)

    init_carry = model.initialize_carry(batch_size=batch_size)
    variables = model.init(_rng, init_carry, (dummy_obs, dummy_dones, dummy_avail))
    params = variables["params"]

    # We don't actually train, so use a trivial optimizer.
    tx = optax.sgd(learning_rate=1.0)

    rng, dropout_key = jax.random.split(rng)
    state = TrainState.create(apply_fn=model.apply, params=params, key=dropout_key, tx=tx)
    return state, model


# --------------------------------------------------------------------------- #
# Safetensors loading for PPO params
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# PPO evaluation
# --------------------------------------------------------------------------- #
@jax.jit
def evaluate_ppo(network_carry, train_state, obs, act, game_len_mask, legal_moves):
    """
    obs:           (B,T,D)
    act:           (B,T,1)
    game_len_mask: (B,T)
    legal_moves:   (B,T,A)
    """
    # Convert to time-major for the RNN: (T,B,·)
    obs_TB = jnp.swapaxes(obs, 0, 1)  # (T,B,D)
    avail_TB = jnp.swapaxes(legal_moves, 0, 1)  # (T,B,A)

    T, B, _ = obs_TB.shape

    # Resets/dones: we reset at t=0 for every sequence.
    # (If your PPO training used a different convention, adapt here.)
    dones_TB = jnp.zeros((T, B), dtype=jnp.float32)
    dones_TB = dones_TB.at[0, :].set(1.0)

    variables = {"params": jax.lax.stop_gradient(train_state.params)}
    new_hidden, pi, value, adv = train_state.apply_fn(
        variables,
        network_carry,
        (obs_TB, dones_TB, avail_TB),
    )

    # pi.logits: (T,B,A) -> (B,T,A) for metrics
    logits = jnp.swapaxes(pi.logits, 0, 1)

    metrics = compute_metrics(logits, act, game_len_mask)
    return new_hidden, metrics


# --------------------------------------------------------------------------- #
# Config & path utils
# --------------------------------------------------------------------------- #
def load_config(config_path: str) -> ml_collections.ConfigDict:
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return ml_collections.ConfigDict(cfg)


def infer_config_path_from_params(params_path: str) -> str:
    """
    Given a checkpoint like:
        checkpoints/ppo/epoch10_seed0_score20.500_2p.safetensors
    infer:
        checkpoints/ppo/epoch10_seed0_score20.500_2p.yaml
    """
    base, ext = os.path.splitext(params_path)
    yaml_candidate = base + ".yaml"
    if os.path.exists(yaml_candidate):
        return yaml_candidate

    # Fallback: best-effort glob in same directory.
    pattern = base + "*.yaml"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"Could not infer config path from params path: {params_path}. "
            f"Tried {yaml_candidate} and {pattern}"
        )
    return matches[0]


# --------------------------------------------------------------------------- #
# Main evaluation logic
# --------------------------------------------------------------------------- #
def evaluate_ppo_on_val(
    ppo_params_path: str,
    val_data_path: str,
    config_path: str | None = None,
    seed: int = 0,
):
    # 1) Config
    if config_path is None:
        config_path = infer_config_path_from_params(ppo_params_path)
        print(f"Inferred config path: {config_path}")
    config = load_config(config_path)

    # 2) Dataset and dataloader
    key = jax.random.PRNGKey(seed)
    val_dataset = HanabiLiveGamesDataset(val_data_path)
    val_loader = HanabiLiveGamesDataloader(val_dataset)  # returns all data in a single batch

    val_games_batch = next(iter(val_loader))  # All validation games.

    # 3) Trajectory reconstruction (same as in BC eval)
    key, _rng = jax.random.split(key)
    get_trajectory_batch_vjit = jax.jit(
        jax.vmap(make_trajectory_batcher(val_dataset.num_players), in_axes=0)
    )

    trajectory_rngs = jax.random.split(_rng, len(val_dataset))
    val_trajectory_batch = get_trajectory_batch_vjit(
        trajectory_rngs,
        val_games_batch.decks,
        val_games_batch.actions,
    )
    val_obs, val_act, val_legal_moves, val_game_len_mask = transform_trajectory_batch(
        val_trajectory_batch, val_games_batch.game_len_masks
    )

    # infer dims for init
    batch_size = val_obs.shape[0]
    obs_dim = val_obs.shape[-1]
    action_dim = val_legal_moves.shape[-1]

    # 4) Load PPO params first (so we can pick the correct actor variant)
    print(f"Loading PPO params from: {ppo_params_path}")
    variables = load_flax_safetensors(ppo_params_path)
    params = variables["params"]

    # detect whether checkpoint has adv_out in the actor
    has_adv = actor_has_adv_out(params["actor"])
    actor_cls = ActorLstm if has_adv else ActorLstmNoAdv

    # 5) Build model & train state with the matching actor class
    key, init_key = jax.random.split(key)
    train_state, model = get_train_state(
        init_key,
        batch_size=batch_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config=config,
        actor_cls=actor_cls,
    )
    train_state = train_state.replace(params=params)

    # 6) Evaluate on validation trajectories
    val_network_carry = model.initialize_carry(batch_size=batch_size)
    _, val_metrics = evaluate_ppo(
        network_carry=val_network_carry,
        train_state=train_state,
        obs=val_obs,
        act=val_act,
        game_len_mask=val_game_len_mask,
        legal_moves=val_legal_moves,
    )

    # print("Validation metrics (PPO):")
    # print(f"  loss              = {float(val_metrics['loss']):.6f}")
    # print(f"  accuracy          = {float(val_metrics['accuracy']):.6f}")
    # print(f"  non_noop_accuracy = {float(val_metrics['non_noop_accuracy']):.6f}")

    return val_metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PPO model on validation set.")
    parser.add_argument(
        "--ppo_params_path",
        type=str,
        required=True,
        help="Path to PPO safetensors checkpoint.",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        required=True,
        help="Path to validation safetensors dataset.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="ah2ac2/training/config/rl_ppo_bc_anchor_2p.yaml",
        help=(
            "Path to config .yaml. If not provided, inferred by replacing "
            ".safetensors with .yaml next to the checkpoint."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for trajectory reconstruction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluate_ppo_on_val(
        ppo_params_path=args.ppo_params_path,
        val_data_path=args.val_data_path,
        config_path=args.config_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
