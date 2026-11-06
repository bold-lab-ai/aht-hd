# vae_2p_masked_mixture.py
# ----------------------------------------------------------
# Drop-in variant of vae_2p_masked.py that supports training
# from (human + OBL) datasets with controlled human oversampling.
#
# Key design goals:
#   - Keep the original structure intact.
#   - Make *surgical* changes: only data-loading / batch sampling changes.
#   - Preserve the same downstream batch interface: .decks, .actions, .game_len_masks
# ----------------------------------------------------------

from __future__ import annotations

import glob
import math
import os
from typing import Literal, NamedTuple, Optional, Sequence, Tuple, TypedDict

import chex
import flax.linen as nn
import flax.training.train_state as train_state_lib
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import struct

# Local deps ------------------------------------------------
from jaxmarl import make
from jaxmarl.environments.hanabi import hanabi_game
from jaxmarl.environments.hanabi.hanabi import HanabiEnv
from omegaconf import OmegaConf
from tqdm import tqdm

from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset
from ah2ac2.nn.multi_layer_lstm import MultiLayerLstm

# ==========================================================
# Obs layout (HanabiLiveGamesDataset / jaxmarl style):
# [OtherHand(127) | Board(76) | Discards(50) | LastAction(55) | Belief(350)] = 658
# ==========================================================

OTHER_HAND_START = 0
OTHER_HAND_LEN = 127

BOARD_START = OTHER_HAND_START + OTHER_HAND_LEN  # 127
BOARD_LEN = 76

DISCARDS_START = BOARD_START + BOARD_LEN  # 203
DISCARDS_LEN = 50

LAST_ACTION_START = DISCARDS_START + DISCARDS_LEN  # 253
LAST_ACTION_LEN = 55

BELIEF_START = LAST_ACTION_START + LAST_ACTION_LEN  # 308
BELIEF_LEN = 350

OBS_DIM_EXPECTED = BELIEF_START + BELIEF_LEN  # 658

# Other-hand internal
OTHER_HAND_CARDS_START = OTHER_HAND_START
OTHER_HAND_CARDS_LEN = 125  # 5 cards * 25
OTHER_HAND_MISSING_START = OTHER_HAND_CARDS_START + OTHER_HAND_CARDS_LEN  # 125
OTHER_HAND_MISSING_LEN = 2

# Board internal
BOARD_DECK_START = BOARD_START
BOARD_DECK_LEN = 40
BOARD_FIREWORKS_START = BOARD_DECK_START + BOARD_DECK_LEN  # 167
BOARD_FIREWORKS_LEN = 25
BOARD_INFO_TOK_START = BOARD_FIREWORKS_START + BOARD_FIREWORKS_LEN  # 192
BOARD_INFO_TOK_LEN = 8
BOARD_LIFE_TOK_START = BOARD_INFO_TOK_START + BOARD_INFO_TOK_LEN  # 200
BOARD_LIFE_TOK_LEN = 3

# Discards internal
DISCARDS_COLOUR_LEN = 10
DISCARDS_NUM_COLOURS = 5

# Last-action internal fields (sum = 55)
LA_ACTING_IDX_LEN = 2
LA_MOVETYPE_LEN = 4
LA_TARGET_IDX_LEN = 2
LA_COLOR_LEN = 5
LA_RANK_LEN = 5
LA_REVEAL_OUT_LEN = 5
LA_POSITION_LEN = 5
LA_CARD_LEN = 25
LA_SCORED_LEN = 1
LA_ADD_INFO_LEN = 1

LA_ACTING_IDX_START = LAST_ACTION_START
LA_MOVETYPE_START = LA_ACTING_IDX_START + LA_ACTING_IDX_LEN
LA_TARGET_IDX_START = LA_MOVETYPE_START + LA_MOVETYPE_LEN
LA_COLOR_START = LA_TARGET_IDX_START + LA_TARGET_IDX_LEN
LA_RANK_START = LA_COLOR_START + LA_COLOR_LEN
LA_REVEAL_OUT_START = LA_RANK_START + LA_RANK_LEN
LA_POSITION_START = LA_REVEAL_OUT_START + LA_REVEAL_OUT_LEN
LA_CARD_START = LA_POSITION_START + LA_POSITION_LEN
LA_SCORED_START = LA_CARD_START + LA_CARD_LEN
LA_ADD_INFO_START = LA_SCORED_START + LA_SCORED_LEN


# ==========================================================
# Masking utilities (unchanged)
# ==========================================================


def _inverted_dropout_mask(rng, shape, p_drop, dtype):
    """Returns a mask in {0, 1/(1-p_drop)} with expected value 1."""
    if p_drop <= 0.0:
        return jnp.ones(shape, dtype=dtype)
    keep = 1.0 - p_drop
    keep = jnp.maximum(keep, 1e-6)
    m = jax.random.bernoulli(rng, p=keep, shape=shape).astype(dtype)
    return m / keep


def _apply_block_dropout(x, *, rng, p_drop, block_shape):
    """x: (T,B,...) ; block_shape broadcastable to x."""
    m = _inverted_dropout_mask(rng, block_shape, p_drop, x.dtype)
    return x * m


def slot_dropout_belief(belief, rng, p_slot, training):
    """belief: (T,B,350) where 350 = 10 slots * 35 dims. Drops whole slots."""
    if (p_slot <= 0.0) or (not training):
        return belief

    T, B, D = belief.shape
    assert D == 350, f"Expected belief dim 350, got {D}"

    keep = 1.0 - p_slot
    keep = jnp.maximum(keep, 1e-6)

    belief4 = belief.reshape(T, B, 10, 35)
    slot_mask = jax.random.bernoulli(rng, p=keep, shape=(T, B, 10, 1)).astype(belief.dtype)
    belief4 = belief4 * slot_mask
    return belief4.reshape(T, B, 350)


def structured_mask_obs_for_encoder(
    obs_seq: chex.Array,
    *,
    rng: chex.PRNGKey,
    training: bool,
    p_hand_card: float,
    p_discards_colour: float,
    p_last_action_block: float,
    p_last_action_reveal_fields: float,
    p_last_action_card: float,
    p_board_feature: float,
    belief_slot_dropout: float,
):
    """Applies structured corruption to obs_seq for ENCODER input only. obs_seq: (T,B,658)."""
    if (not training) or (
        p_hand_card <= 0.0
        and p_discards_colour <= 0.0
        and p_last_action_block <= 0.0
        and p_last_action_reveal_fields <= 0.0
        and p_last_action_card <= 0.0
        and p_board_feature <= 0.0
        and belief_slot_dropout <= 0.0
    ):
        return obs_seq

    T, B, D = obs_seq.shape
    assert D == OBS_DIM_EXPECTED, f"Expected obs dim {OBS_DIM_EXPECTED}, got {D}"

    obs_enc = obs_seq
    rng, rng_hand, rng_board, rng_disc, rng_la_block, rng_la_fields, rng_belief = jax.random.split(
        rng, 7
    )

    # (1) Other-hand per-card block dropout (5 cards * 25 dims)
    if p_hand_card > 0.0:
        hand_cards = obs_enc[
            :, :, OTHER_HAND_CARDS_START : OTHER_HAND_CARDS_START + OTHER_HAND_CARDS_LEN
        ]  # (T,B,125)
        hand_cards4 = hand_cards.reshape(T, B, 5, 25)
        hand_cards4 = _apply_block_dropout(
            hand_cards4, rng=rng_hand, p_drop=p_hand_card, block_shape=(T, B, 5, 1)
        )
        hand_cards = hand_cards4.reshape(T, B, 125)
        obs_enc = obs_enc.at[
            :, :, OTHER_HAND_CARDS_START : OTHER_HAND_CARDS_START + OTHER_HAND_CARDS_LEN
        ].set(hand_cards)

    # (2) Board mild feature dropout
    if p_board_feature > 0.0:
        board = obs_enc[:, :, BOARD_START : BOARD_START + BOARD_LEN]  # (T,B,76)
        board = _apply_block_dropout(
            board, rng=rng_board, p_drop=p_board_feature, block_shape=(T, B, BOARD_LEN)
        )
        obs_enc = obs_enc.at[:, :, BOARD_START : BOARD_START + BOARD_LEN].set(board)

    # (3) Discards per-colour block dropout (5 blocks * 10 dims)
    if p_discards_colour > 0.0:
        disc = obs_enc[:, :, DISCARDS_START : DISCARDS_START + DISCARDS_LEN]  # (T,B,50)
        disc4 = disc.reshape(T, B, DISCARDS_NUM_COLOURS, DISCARDS_COLOUR_LEN)
        disc4 = _apply_block_dropout(
            disc4,
            rng=rng_disc,
            p_drop=p_discards_colour,
            block_shape=(T, B, DISCARDS_NUM_COLOURS, 1),
        )
        disc = disc4.reshape(T, B, 50)
        obs_enc = obs_enc.at[:, :, DISCARDS_START : DISCARDS_START + DISCARDS_LEN].set(disc)

    # (4) Last action: (a) block dropout + (b) heavier dropout on reveal/card fields
    if (
        (p_last_action_block > 0.0)
        or (p_last_action_reveal_fields > 0.0)
        or (p_last_action_card > 0.0)
    ):
        la = obs_enc[:, :, LAST_ACTION_START : LAST_ACTION_START + LAST_ACTION_LEN]  # (T,B,55)

        if p_last_action_block > 0.0:
            keep = jnp.maximum(1.0 - p_last_action_block, 1e-6)
            la_keep = jax.random.bernoulli(rng_la_block, p=keep, shape=(T, B, 1)).astype(la.dtype)
            la = la * la_keep

        if p_last_action_reveal_fields > 0.0:
            keep = jnp.maximum(1.0 - p_last_action_reveal_fields, 1e-6)
            for start, length in [
                (LA_COLOR_START - LAST_ACTION_START, LA_COLOR_LEN),
                (LA_RANK_START - LAST_ACTION_START, LA_RANK_LEN),
                (LA_REVEAL_OUT_START - LAST_ACTION_START, LA_REVEAL_OUT_LEN),
                (LA_POSITION_START - LAST_ACTION_START, LA_POSITION_LEN),
            ]:
                m = jax.random.bernoulli(rng_la_fields, p=keep, shape=(T, B, 1)).astype(la.dtype)
                la = la.at[:, :, start : start + length].set(la[:, :, start : start + length] * m)

        if p_last_action_card > 0.0:
            keep = jnp.maximum(1.0 - p_last_action_card, 1e-6)
            m = jax.random.bernoulli(rng_la_fields, p=keep, shape=(T, B, 1)).astype(la.dtype)
            start = LA_CARD_START - LAST_ACTION_START
            la = la.at[:, :, start : start + LA_CARD_LEN].set(
                la[:, :, start : start + LA_CARD_LEN] * m
            )

        obs_enc = obs_enc.at[:, :, LAST_ACTION_START : LAST_ACTION_START + LAST_ACTION_LEN].set(la)

    # (5) Belief per-slot dropout
    if belief_slot_dropout > 0.0:
        belief = obs_enc[:, :, BELIEF_START : BELIEF_START + BELIEF_LEN]
        belief = slot_dropout_belief(belief, rng_belief, belief_slot_dropout, training)
        obs_enc = obs_enc.at[:, :, BELIEF_START : BELIEF_START + BELIEF_LEN].set(belief)

    return obs_enc


# ==========================================================
# Generic helpers (unchanged)
# ==========================================================


def batchify(x, agent_list):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((len(agent_list), -1))


def unbatchify(x: chex.Array, agent_list: Sequence[str]):
    return {a: x[i] for i, a in enumerate(agent_list)}


# ==========================================================
# Trajectory factory (unchanged)
# ==========================================================


class StepCarry(NamedTuple):
    rng: chex.PRNGKey
    prev_obs: dict
    prev_env_state: hanabi_game.State


class Transition(NamedTuple):
    observations: chex.Array  # (A, obs_dim)
    actions: chex.Array  # (A, 1)
    legal_actions: chex.Array  # (A, act_dim)


def make_trajectory_batcher(num_players: int):
    hand = 5 if num_players <= 3 else 4
    env: HanabiEnv = make("hanabi", hand_size=hand, num_agents=num_players)

    def get_trajectory_batch(rng: chex.PRNGKey, deck, actions):
        def _step(carry: StepCarry, step_actions):
            step_rng, _rng = jax.random.split(carry.rng)
            env_act = unbatchify(step_actions, env.agents)
            obs, env_state, *_ = env.step_env(_rng, carry.prev_env_state, env_act)

            reached_terminal = carry.prev_env_state.terminal
            env_state = jax.lax.cond(
                reached_terminal, lambda: carry.prev_env_state, lambda: env_state
            )
            obs = jax.lax.cond(reached_terminal, lambda: carry.prev_obs, lambda: obs)

            legal_moves_batched = batchify(env.get_legal_moves(carry.prev_env_state), env.agents)
            prev_obs_batched = batchify(carry.prev_obs, env.agents)
            act_batched = batchify(env_act, env.agents)

            new_carry = StepCarry(prev_obs=obs, prev_env_state=env_state, rng=step_rng)
            new_tr = Transition(prev_obs_batched, act_batched, legal_moves_batched)
            return new_carry, new_tr

        rng, _rng = jax.random.split(rng)
        init_obs, init_state = env.reset_from_deck_of_pairs(deck)
        init = StepCarry(prev_obs=init_obs, prev_env_state=init_state, rng=_rng)
        _, traj = jax.lax.scan(_step, init, actions)
        return traj

    return get_trajectory_batch


def transform_trajectory_batch(trajectory_batch, game_len_mask):
    obs_batch, act_batch, legal_batch = jax.tree.map(
        lambda x: jnp.swapaxes(x, 1, 2).reshape(-1, x.shape[1], x.shape[-1]),
        trajectory_batch,
    )
    mask_batch = jnp.repeat(game_len_mask, trajectory_batch[0].shape[2], axis=0)
    return obs_batch, act_batch, legal_batch, mask_batch


def to_time_major(obs_ba: chex.Array, act_ba: chex.Array):
    obs_tm = jnp.swapaxes(obs_ba, 0, 1)
    act_tm = jnp.swapaxes(act_ba, 0, 1)
    return obs_tm, act_tm


# ==========================================================
# NEW: Mixed dataset batch sampler (surgical addition)
# ==========================================================


class GamesBatch(NamedTuple):
    """Match what HanabiLiveGamesDataloader yields (fields you use)."""

    decks: chex.Array
    actions: chex.Array
    game_len_masks: chex.Array


SamplingMode = Literal["cycle", "with_replacement"]


class MixtureGamesDataloader:
    """
    Deterministic mixed-batch loader that yields batches composed of:
      k_h human games + k_o obl games, per step.

    Human oversampling modes:
      - "cycle": iterate through a shuffled permutation without replacement,
                 reshuffling when exhausted.
      - "with_replacement": sample indices i.i.d. each step.

    OBL sampling default:
      - "with_replacement" (OBL is large; avoids expensive epoch bookkeeping)
    """

    def __init__(
        self,
        human_ds: HanabiLiveGamesDataset,
        obl_ds: HanabiLiveGamesDataset,
        batch_size: int,
        human_frac: float,
        rng: chex.PRNGKey,
        steps_per_epoch: int,
        human_sampling: SamplingMode = "cycle",
        obl_sampling: SamplingMode = "with_replacement",
    ):
        assert 0.0 < human_frac < 1.0
        self.human_ds = human_ds
        self.obl_ds = obl_ds
        self.batch_size = int(batch_size)
        self.human_frac = float(human_frac)
        self.steps_per_epoch = int(steps_per_epoch)

        self.k_h = int(round(self.batch_size * self.human_frac))
        self.k_h = max(1, min(self.batch_size - 1, self.k_h))
        self.k_o = self.batch_size - self.k_h

        self.human_sampling = human_sampling
        self.obl_sampling = obl_sampling

        self._rng = rng

        # For "cycle" mode
        self._human_perm = None  # type: Optional[jnp.ndarray]
        self._human_ptr = 0

        self._reset_human_cycle()

    def __len__(self):
        return self.steps_per_epoch

    def _reset_human_cycle(self):
        if self.human_sampling != "cycle":
            return
        self._rng, key = jax.random.split(self._rng)
        self._human_perm = jax.random.permutation(key, len(self.human_ds))
        self._human_ptr = 0

    def _sample_human_indices(self) -> jnp.ndarray:
        if self.human_sampling == "with_replacement":
            self._rng, key = jax.random.split(self._rng)
            return jax.random.randint(key, (self.k_h,), 0, len(self.human_ds))
        elif self.human_sampling == "cycle":
            assert self._human_perm is not None
            # If not enough left in this permutation, reshuffle and continue.
            if self._human_ptr + self.k_h > len(self.human_ds):
                self._reset_human_cycle()
            idx = self._human_perm[self._human_ptr : self._human_ptr + self.k_h]
            self._human_ptr += self.k_h
            return idx
        else:
            raise ValueError(f"Unknown human_sampling={self.human_sampling}")

    def _sample_obl_indices(self) -> jnp.ndarray:
        if self.obl_sampling == "with_replacement":
            self._rng, key = jax.random.split(self._rng)
            return jax.random.randint(key, (self.k_o,), 0, len(self.obl_ds))
        elif self.obl_sampling == "cycle":
            # Implement if you ever want it; default is with_replacement.
            raise NotImplementedError(
                "obl_sampling='cycle' not implemented (use with_replacement)."
            )
        else:
            raise ValueError(f"Unknown obl_sampling={self.obl_sampling}")

    def _stack_rows(self, rows) -> GamesBatch:
        # Rows are NamedTuples from HanabiLiveGamesDataset. We stack fields.
        # We use jnp.stack to keep things JAX-friendly.
        decks = jnp.stack([r.decks for r in rows], axis=0)
        actions = jnp.stack([r.actions for r in rows], axis=0)
        masks = jnp.stack([r.game_len_masks for r in rows], axis=0)
        return GamesBatch(decks=decks, actions=actions, game_len_masks=masks)

    def __iter__(self):
        # Make epoch iteration deterministic given current RNG state.
        for _ in range(self.steps_per_epoch):
            h_idx = self._sample_human_indices()
            o_idx = self._sample_obl_indices()

            # Fetch rows (Python loop; datasets are not JAX pytrees anyway)
            h_rows = [self.human_ds[int(i)] for i in list(h_idx)]
            o_rows = [self.obl_ds[int(i)] for i in list(o_idx)]

            # Combine and shuffle within the batch so ordering doesn't encode domain.
            rows = h_rows + o_rows
            self._rng, key = jax.random.split(self._rng)
            perm = list(jax.random.permutation(key, len(rows)))
            rows = [rows[int(i)] for i in perm]

            yield self._stack_rows(rows)


# ==========================================================
# VAE (unchanged from your script)
# ==========================================================


class _MLP(nn.Module):
    features: Tuple[int, ...]
    act: str = "relu"

    @nn.compact
    def __call__(self, x, *, training: bool):
        act_fn = dict(relu=nn.relu, gelu=nn.gelu, tanh=nn.tanh)[self.act]
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = act_fn(x)
        return nn.Dense(self.features[-1])(x)


class VAE(nn.Module):
    preprocessing_features: Tuple[int, ...]
    lstm_features: Tuple[int, ...]
    postprocessing_features: Tuple[int, ...]
    action_dim: int
    latent_dim: int = 16
    dropout: float = 0.0
    act_fn: str = "relu"
    beta: float = 1.0

    belief_slot_dropout: float = 0.0
    p_hand_card: float = 0.0
    p_discards_colour: float = 0.0
    p_last_action_block: float = 0.0
    p_last_action_reveal_fields: float = 0.0
    p_last_action_card: float = 0.0
    p_board_feature: float = 0.0

    @nn.compact
    def __call__(self, *, obs_seq, act_seq, training: bool):
        T, B, _ = obs_seq.shape

        if training:
            mask_rng = self.make_rng("enc_mask")
        else:
            mask_rng = jax.random.PRNGKey(0)

        obs_enc = structured_mask_obs_for_encoder(
            obs_seq,
            rng=mask_rng,
            training=training,
            p_hand_card=self.p_hand_card,
            p_discards_colour=self.p_discards_colour,
            p_last_action_block=self.p_last_action_block,
            p_last_action_reveal_fields=self.p_last_action_reveal_fields,
            p_last_action_card=self.p_last_action_card,
            p_board_feature=self.p_board_feature,
            belief_slot_dropout=self.belief_slot_dropout,
        )

        enc_in = obs_enc.reshape(T * B, -1)
        h = _MLP(self.preprocessing_features + (128,), self.act_fn)(enc_in, training=training)
        mu = nn.Dense(self.latent_dim)(h)
        lv = nn.Dense(self.latent_dim)(h)
        mu, lv = [y.reshape(T, B, -1).mean(0) for y in (mu, lv)]
        std, eps = jnp.exp(0.5 * lv), jax.random.normal(self.make_rng("latent"), lv.shape)
        z = mu + std * eps
        kl = 0.5 * jnp.sum(mu**2 + std**2 - 1.0 - lv, axis=1).mean()

        dec_in = jnp.concatenate([obs_seq, jnp.repeat(z[None], T, axis=0)], -1)
        dec_in = jnp.swapaxes(dec_in, 0, 1)  # (B,T,D+Z)
        lstm_block = MultiLayerLstm(
            preprocessing_features=self.preprocessing_features,
            lstm_features=self.lstm_features,
            postprocessing_features=self.postprocessing_features,
            action_dim=self.action_dim,
            dropout_rate=self.dropout,
            activation_fn_name=self.act_fn,
        )
        carry0 = lstm_block.initialize_carry(batch_size=B)
        _, logits_bt = lstm_block(x=dec_in, carry=carry0, training=training)  # (B,T,A)
        logits = jnp.swapaxes(logits_bt, 0, 1)  # (T,B,A)

        recon = optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, self.action_dim), act_seq.reshape(-1)
        ).mean()
        loss = recon + self.beta * kl
        return logits, {"loss": loss, "recon": recon, "kl": kl, "mu": mu, "lv": lv, "z": z}


# ==========================================================
# Training utilities (unchanged)
# ==========================================================


class TrainState(train_state_lib.TrainState):
    key: chex.PRNGKey = struct.field(pytree_node=False)
    beta: jnp.ndarray


def get_train_state(rng, in_shape, cfg):
    rng, init_rng = jax.random.split(rng)
    model = VAE(
        preprocessing_features=tuple(cfg.preprocessing_features),
        lstm_features=tuple(cfg.lstm_features),
        postprocessing_features=tuple(cfg.postprocessing_features),
        action_dim=cfg.action_dim,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
        act_fn=cfg.act_fn,
        beta=cfg.beta,
        belief_slot_dropout=cfg.belief_slot_dropout,
        p_hand_card=cfg.p_hand_card,
        p_discards_colour=cfg.p_discards_colour,
        p_last_action_block=cfg.p_last_action_block,
        p_last_action_reveal_fields=cfg.p_last_action_reveal_fields,
        p_last_action_card=cfg.p_last_action_card,
        p_board_feature=cfg.p_board_feature,
    )

    T, B, D = in_shape
    dummy_o = jnp.ones((T, B, D))
    dummy_a = jnp.zeros((T, B, 1), jnp.int32)
    params = model.init(init_rng, obs_seq=dummy_o, act_seq=dummy_a, training=False)["params"]
    rng, dropout_rng = jax.random.split(rng)

    sched = optax.linear_schedule(
        cfg.lr_initial, cfg.lr_final, cfg.num_epochs * cfg.steps_per_epoch
    )
    tx = optax.adamw(sched)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        beta=jnp.array(0.0, dtype=jnp.float32),
        key=dropout_rng,
    ), model


@jax.jit
def train_step(state: TrainState, obs, act, target_kl, beta_adapt_rate):
    def loss_fn(params, beta, rng_key):
        step_key, latent_key, belief_key, enc_mask_key = jax.random.split(rng_key, 4)
        _, outs = state.apply_fn(
            {"params": params},
            obs_seq=obs,
            act_seq=act,
            training=True,
            rngs={
                "dropout": step_key,
                "latent": latent_key,
                "belief_dropout": belief_key,
                "enc_mask": enc_mask_key,
            },
        )
        loss_val = outs["recon"] + beta * outs["kl"]
        return loss_val, outs

    rng_key = jax.random.fold_in(state.key, state.step)
    (loss_val, outs), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params, state.beta, rng_key
    )

    err = outs["kl"] - target_kl
    new_beta = state.beta * jnp.exp(beta_adapt_rate * err / target_kl)
    new_beta = jnp.clip(new_beta, 1e-4, 5.0)

    new_state = state.apply_gradients(grads=grads).replace(beta=new_beta)

    metrics = {"loss": loss_val, "recon": outs["recon"], "kl": outs["kl"], "beta": new_beta}
    return new_state, metrics


# ==========================================================
# Metrics + evaluation (unchanged from your script)
# ==========================================================


class EvaluationMetrics(TypedDict):
    loss: float
    accuracy: float
    non_noop_accuracy: float


def masked_cross_entropy_loss(
    logits: chex.Array, actions: chex.Array, mask: chex.Array
) -> chex.Array:
    actions = actions.squeeze(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, actions)
    return (loss * mask).sum() / mask.sum()


def compute_metrics(
    logits: chex.Array, actions: chex.Array, mask: chex.Array
) -> EvaluationMetrics:
    actions_sq = actions.squeeze(-1)
    preds = jnp.argmax(logits, -1)
    correct = preds == actions_sq

    acc = (correct * mask).sum() / mask.sum()
    non_noop_mask = (actions_sq != actions_sq.max()) * mask
    non_noop_correct = (correct * non_noop_mask).sum()
    non_noop_accuracy = non_noop_correct / non_noop_mask.sum()

    ce_loss = masked_cross_entropy_loss(logits, actions, mask)
    return EvaluationMetrics(loss=ce_loss, accuracy=acc, non_noop_accuracy=non_noop_accuracy)


def evaluate(
    model_apply,
    params,
    obs_seq_tm: chex.Array,
    act_seq_tm: chex.Array,
    mask_ba_t: chex.Array,
    legal_moves_ba_t: chex.Array,
    rng=jax.random.PRNGKey(0),
) -> EvaluationMetrics:
    logits_tm, _ = model_apply(
        {"params": params},
        obs_seq=obs_seq_tm,
        act_seq=act_seq_tm,
        training=False,
        rngs={"latent": rng, "dropout": rng},
    )

    logits_bt = jnp.swapaxes(logits_tm, 0, 1)  # (B*,T,A)
    logits_bt = logits_bt - (1 - legal_moves_ba_t) * 1e10

    return compute_metrics(
        logits_bt,
        jnp.swapaxes(act_seq_tm, 0, 1),
        mask_ba_t,
    )


# ==========================================================
# Main entry (surgical changes are in data loading)
# ==========================================================


def _infer_steps_per_epoch_from_human(
    human_ds_len: int, batch_size: int, human_frac: float
) -> int:
    """
    Define "one epoch" as roughly one pass over the *human* dataset during the mixed stage.
    Humans are the scarce signal; this keeps LR schedule / logging intuitive.

    In mixed stage, each step uses k_h humans.
    """
    k_h = int(round(batch_size * human_frac))
    k_h = max(1, min(batch_size - 1, k_h))
    return int(math.ceil(human_ds_len / k_h))


def run(cfg: ml_collections.ConfigDict):
    rng = jax.random.PRNGKey(cfg.seed)

    # ----- dataset & loader -----
    aug_key = jax.random.fold_in(rng, cfg.seed) if cfg.should_augment_data else None

    # Human train dataset (always exists)
    human_train_ds = HanabiLiveGamesDataset(
        file=cfg.human_train_data_path,
        color_shuffle_key=aug_key,
        limited_data_regime_ratio=getattr(cfg, "limited_data_ratio", None),
        limited_data_regime_key=None,
        limited_data_regime_games=getattr(cfg, "limited_data_regime_games", None),
    )

    # Optional OBL train dataset (only used if mix.enabled)
    mix_enabled = bool(getattr(cfg, "mix_enabled", False))
    if mix_enabled:
        obl_train_ds = HanabiLiveGamesDataset(
            file=cfg.obl_train_data_path,
            color_shuffle_key=aug_key,
            limited_data_regime_ratio=None,
            limited_data_regime_key=None,
            limited_data_regime_games=None,
        )
        assert human_train_ds.num_players == obl_train_ds.num_players, (
            "Human/OBL num_players mismatch."
        )
    else:
        obl_train_ds = None

    # Steps per epoch: either user-specified or inferred from human set
    if getattr(cfg, "steps_per_epoch", None) is None:
        steps_per_epoch = _infer_steps_per_epoch_from_human(
            human_ds_len=len(human_train_ds),
            batch_size=cfg.batch_size,
            human_frac=float(getattr(cfg, "human_frac", 0.7)) if mix_enabled else 1.0,
        )
    else:
        steps_per_epoch = int(cfg.steps_per_epoch)
    cfg.steps_per_epoch = steps_per_epoch

    # Build loaders
    rng, human_loader_rng = jax.random.split(rng)
    human_train_dl = HanabiLiveGamesDataloader(human_train_ds, cfg.batch_size, human_loader_rng)

    if mix_enabled:
        rng, mix_loader_rng = jax.random.split(rng)
        mix_dl = MixtureGamesDataloader(
            human_ds=human_train_ds,
            obl_ds=obl_train_ds,
            batch_size=cfg.batch_size,
            human_frac=float(cfg.human_frac),
            rng=mix_loader_rng,
            steps_per_epoch=steps_per_epoch,
            human_sampling=str(getattr(cfg, "human_sampling", "cycle")),
            obl_sampling=str(getattr(cfg, "obl_sampling", "with_replacement")),
        )
        warmup_epochs = int(getattr(cfg, "warmup_epochs", 0))
    else:
        mix_dl = None
        warmup_epochs = 0

    # vmapped, jit-compiled trajectory factory
    get_trajectory_batch_vjit = jax.jit(
        jax.vmap(make_trajectory_batcher(human_train_ds.num_players), in_axes=0)
    )

    # ------------ validation loader (human validation) ----------------
    val_ds = HanabiLiveGamesDataset(cfg.val_data_path)
    val_dl = HanabiLiveGamesDataloader(val_ds)  # full pass (small)
    val_batch = next(iter(val_dl))
    rng, val_rng = jax.random.split(rng)
    val_keys = jax.random.split(val_rng, len(val_ds))
    val_traj = get_trajectory_batch_vjit(val_keys, val_batch.decks, val_batch.actions)
    v_obs_ba, v_act_ba, v_legals_ba, v_mask_ba_t = transform_trajectory_batch(
        val_traj, val_batch.game_len_masks
    )
    v_obs_tm, v_act_tm = to_time_major(v_obs_ba, v_act_ba)

    # ----- shape inference -----
    # Use a human batch for shape (always valid)
    dummy_batch = next(iter(human_train_dl))
    rng, dummy_rng = jax.random.split(rng)
    dummy_keys = jax.random.split(dummy_rng, dummy_batch.actions.shape[0])
    dummy_traj = get_trajectory_batch_vjit(dummy_keys, dummy_batch.decks, dummy_batch.actions)
    dummy_obs_ba, dummy_act_ba, _, _ = transform_trajectory_batch(
        dummy_traj, dummy_batch.game_len_masks
    )
    dummy_obs_seq, dummy_act_seq = to_time_major(dummy_obs_ba, dummy_act_ba)
    T, B, D = dummy_obs_seq.shape

    state, model = get_train_state(rng, (T, B, D), cfg)
    best_val_acc: Optional[float] = None

    # ----- training loop -----
    for ep in range(cfg.num_epochs):
        # Choose dataloader for this epoch
        if mix_enabled and (ep >= warmup_epochs):
            train_iterable = mix_dl
            epoch_len = len(mix_dl)
            epoch_desc = f"Epoch {ep + 1}/{cfg.num_epochs} [mixed h={cfg.human_frac:.2f}]"
        else:
            # Human-only epoch
            train_iterable = human_train_dl
            epoch_len = len(human_train_dl)
            epoch_desc = f"Epoch {ep + 1}/{cfg.num_epochs} [human]"

        acc = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

        for batch in tqdm(train_iterable, desc=epoch_desc, total=epoch_len):
            rng, batch_rng = jax.random.split(rng)
            traj_keys = jax.random.split(batch_rng, batch.actions.shape[0])

            traj_b = get_trajectory_batch_vjit(traj_keys, batch.decks, batch.actions)
            obs_ba, act_ba, _, _ = transform_trajectory_batch(traj_b, batch.game_len_masks)
            obs_seq, act_seq = to_time_major(obs_ba, act_ba)

            state, metrics = train_step(
                state, obs_seq, act_seq, cfg.target_kl, cfg.beta_adapt_rate
            )
            for k in acc:
                acc[k] += float(metrics[k])

        for k in acc:
            acc[k] /= float(epoch_len)

        # ---------------- validation -----------------------
        rng, eval_key = jax.random.split(rng)
        val_metrics = evaluate(
            model.apply,
            state.params,
            v_obs_tm,
            v_act_tm,
            v_mask_ba_t,
            v_legals_ba,
            rng=eval_key,
        )
        val_acc = float(val_metrics["non_noop_accuracy"])

        print(
            f"Epoch {ep + 1:02d} | "
            f"loss {acc['loss']:.4f}  recon {acc['recon']:.4f}  kl {acc['kl']:.4f} | "
            f"beta {float(state.beta):.3f} | "
            f"val-acc {val_metrics['accuracy']:.3%}  "
            f"val-non-noop {val_acc:.3%}"
        )

        # ----- optional save -----
        if cfg.save_path:
            os.makedirs(cfg.save_path, exist_ok=True)

            if (best_val_acc is None) or (val_acc > best_val_acc):
                if best_val_acc is not None:
                    pattern_params = os.path.join(
                        cfg.save_path,
                        f"epoch*_seed{cfg.seed}_valacc{best_val_acc:.3f}.safetensors",
                    )
                    pattern_cfg = pattern_params.replace(".safetensors", ".yaml")
                    for p in glob.glob(pattern_params):
                        os.remove(p)
                    for p in glob.glob(pattern_cfg):
                        os.remove(p)

                best_val_acc = val_acc
                fname_params = os.path.join(
                    cfg.save_path,
                    f"epoch{ep}_seed{cfg.seed}_valacc{best_val_acc:.3f}.safetensors",
                )
                from jaxmarl.wrappers.baselines import save_params

                save_params(state.params, fname_params)
                OmegaConf.save(
                    OmegaConf.create(cfg.to_dict()),
                    fname_params.replace(".safetensors", ".yaml"),
                )


# ----------------------------------------------------------
# Hydra entry point
# ----------------------------------------------------------

if __name__ == "__main__":
    import hydra

    @hydra.main(version_base=None, config_path="config", config_name="vae_2p_mixture")
    def _main(cfg):
        cfg = ml_collections.ConfigDict(OmegaConf.to_container(cfg, resolve=True))
        run(cfg)

    _main()
