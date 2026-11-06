#!/usr/bin/env python3
# eval_bc_on_val.py
"""
Evaluate a trained BC model on the validation set.

Usage
-----
python ah2ac2/tools/eval_bc_on_val.py \
  --bc_params_path checkpoints/bc/human_bc/2k/epoch45_seed0_score0.404_2p.safetensors \
  --val_data_path ah2ac2/datasets/data/2_player_games_val.safetensors

# Optional:
#   --config_path path/to/config.yaml
#   --seed 0
"""

import argparse
import glob
import os
from typing import NamedTuple, TypedDict

import chex
import flax.training.train_state
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import struct
from jaxmarl import make
from jaxmarl.environments.hanabi import hanabi_game
from jaxmarl.environments.hanabi.hanabi import HanabiEnv
from jaxmarl.wrappers.baselines import load_params
from omegaconf import OmegaConf

from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset
from ah2ac2.nn.multi_layer_lstm import MultiLayerLstm


# --------------------------------------------------------------------------- #
# Utility: batchify helpers (same as training script)
# --------------------------------------------------------------------------- #
def batchify(x, agent_list):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((len(agent_list), -1))


def unbatchify(x, agent_list):
    return {a: x[i] for i, a in enumerate(agent_list)}


# --------------------------------------------------------------------------- #
# Trajectory factory (same as training script)
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
# Training utilities needed for evaluation (model + state + metrics)
# --------------------------------------------------------------------------- #
class TrainState(flax.training.train_state.TrainState):
    key: jax.Array = struct.field(pytree_node=False)


class EvaluationMetrics(TypedDict):
    loss: float
    cross_entropy: float
    accuracy: float
    non_noop_accuracy: float


def learning_rate_schedule_fn(config):
    # For eval we might not have num_steps_per_epoch; default to 1.
    if "num_steps_per_epoch" in config:
        num_spe = config.num_steps_per_epoch
    else:
        num_spe = 1

    config.num_steps = config.num_epochs * num_spe
    return optax.linear_schedule(
        init_value=config.lr_initial,
        end_value=config.lr_final,
        transition_steps=int(config.num_steps * 0.9),
    )


def make_model(config):
    return MultiLayerLstm(
        preprocessing_features=config.preprocessing_features,
        lstm_features=config.lstm_features,
        postprocessing_features=config.postprocessing_features,
        action_dim=config.action_dim,
        dropout_rate=config.dropout,
        activation_fn_name=config.act_fn,
    )


def get_train_state(rng, input_shape, config):
    rng, _rng = jax.random.split(rng)

    model = make_model(config)
    init_carry = model.initialize_carry(batch_size=input_shape[0])
    variables = model.init(_rng, x=jnp.ones(input_shape), carry=init_carry, training=False)
    params = variables["params"]

    schedule = learning_rate_schedule_fn(config)
    tx = optax.adamw(learning_rate=schedule)

    rng, dropout_key = jax.random.split(rng)
    state = TrainState.create(apply_fn=model.apply, params=params, key=dropout_key, tx=tx)
    return state, model


def masked_cross_entropy_loss(logits, act, mask):
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


@jax.jit
def evaluate(network_carry, train_state, obs, act, game_len_mask, legal_moves):
    _, logits = train_state.apply_fn(
        {"params": jax.lax.stop_gradient(train_state.params)},
        x=obs,
        carry=network_carry,
        training=False,
    )
    logits = logits - (1 - legal_moves) * 1e10
    return compute_metrics(logits, act, game_len_mask)


# --------------------------------------------------------------------------- #
# Main evaluation logic
# --------------------------------------------------------------------------- #
def load_config(config_path: str) -> ml_collections.ConfigDict:
    cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    return ml_collections.ConfigDict(cfg)


def infer_config_path_from_params(params_path: str) -> str:
    """
    Given a checkpoint like:
        checkpoints/bc/epoch10_seed0_score20.500_2p.safetensors
    infer:
        checkpoints/bc/epoch10_seed0_score20.500_2p.yaml
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


def evaluate_bc_on_val(
    bc_params_path: str,
    val_data_path: str,
    config_path: str | None = None,
    seed: int = 0,
):
    # 1) Config
    if config_path is None:
        config_path = infer_config_path_from_params(bc_params_path)
        print(f"Inferred config path: {config_path}")
    config = load_config(config_path)

    # 2) Dataset and dataloader
    key = jax.random.PRNGKey(seed)
    val_dataset = HanabiLiveGamesDataset(val_data_path)
    val_loader = HanabiLiveGamesDataloader(val_dataset)  # returns all data in a single batch

    val_games_batch = next(iter(val_loader))  # All validation games.

    # 3) Trajectory reconstruction (same as in training)
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

    # 4) Build model & train state, then load params
    key, init_key = jax.random.split(key)
    train_state, model = get_train_state(init_key, val_obs.shape, config)

    print(f"Loading params from: {bc_params_path}")
    params = load_params(bc_params_path)
    train_state = train_state.replace(params=params)

    # 5) Evaluate on validation trajectories
    val_network_carry = model.initialize_carry(batch_size=val_obs.shape[0])
    val_metrics = evaluate(
        network_carry=val_network_carry,
        train_state=train_state,
        obs=val_obs,
        act=val_act,
        game_len_mask=val_game_len_mask,
        legal_moves=val_legal_moves,
    )

    return val_metrics


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BC model on validation set.")
    parser.add_argument(
        "--bc_params_path",
        type=str,
        required=True,
        help="Path to BC safetensors checkpoint.",
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
        default="ah2ac2/training/config/bc_2p.yaml",
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
    evaluate_bc_on_val(
        bc_params_path=args.bc_params_path,
        val_data_path=args.val_data_path,
        config_path=args.config_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
