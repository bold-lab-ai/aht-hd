# br_rl_vae_lstm.py
import functools
import glob
import os
import re
from typing import List, NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import ml_collections
import numpy as np
import optax
import wandb
from flax.core import freeze, unfreeze
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
from jaxmarl.wrappers.baselines import LogWrapper, save_params
from omegaconf import OmegaConf

from ah2ac2.nn.multi_layer_lstm import LstmCellWithHiddenStateReset

jax.config.update("jax_debug_nans", True)

"""
br_rl_vae_lstm.py
  - Each step:
      1) run *all K* frozen partners on the full actor batch (A = E*N) (Choice 1: python loop)
      2) stack partner actions as (K, A)
      3) gather the selected partner action per actor using idx_actor (repeat partner_idx_env)
  - Partner hidden state becomes a bank: (K, L, A, H)
  - Partner sampling uses jax.random.fold_in(_rng, k).
"""


obs_dim = 658
action_dim = 21


# =============================================================
# Safetensors loading (unchanged)
# =============================================================


def load_flax_safetensors_any(path: str):
    """Load a Flax params/vars tree from a .safetensors file.

    Supports flat keys with '/' or ',' separators.
    Returns a FrozenDict tree (vars-like or params-like depending on file).
    """
    from safetensors.numpy import load_file as st_load

    flat = st_load(path)  # dict[str -> np.ndarray]

    def split_key(k: str):
        parts = [p for p in re.split(r"[\/,]", k) if p]
        return tuple(parts)

    items = {split_key(k): jnp.asarray(v) for k, v in flat.items()}
    tree = unflatten_dict(items)
    return freeze(tree)


def _find_first_subtree_with_keys(d: dict, wanted: set):
    """DFS through nested dicts to find the first subtree that contains any of `wanted` keys."""
    stack = [([], d)]
    while stack:
        path, cur = stack.pop()
        if isinstance(cur, dict):
            cur_keys = set(cur.keys())
            if len(cur_keys.intersection(wanted)) > 0:
                return path, cur
            for k, v in cur.items():
                if isinstance(v, dict):
                    stack.append((path + [k], v))
    return None, None


def extract_actor_params_any(ckpt_tree):
    """Extract the actor subtree from an RL/RL_VAE checkpoint.

    Expected (most common):
      ckpt_tree = {"params": {"actor": <ActorLstm params>, "critic": ...}}

    We return a FrozenDict suitable for applying a module with a top-level submodule named "actor":
      FrozenDict({"params": {"actor": actor_params}})
    """
    p = unfreeze(ckpt_tree)
    params = p["params"] if ("params" in p and isinstance(p["params"], dict)) else p

    if isinstance(params, dict) and "actor" in params:
        actor = params["actor"]
        return freeze({"params": {"actor": actor}})

    # Fallback: search deeper for something containing an "actor" key.
    _path, subtree = _find_first_subtree_with_keys(params, {"actor"})
    if subtree is not None and "actor" in subtree:
        actor = subtree["actor"]
        return freeze({"params": {"actor": actor}})

    # Helpful diagnostics
    top_keys = list(params.keys())[:20] if isinstance(params, dict) else type(params)
    flat_keys = []
    try:
        flat_keys = list(flatten_dict(params).keys())[:50]
    except Exception:
        pass
    raise RuntimeError(
        "Could not locate an 'actor' subtree in checkpoint.\n"
        f"Top-level keys (first 20): {top_keys}\n"
        f"Some flattened keys (first 50): {flat_keys}\n"
        "Expected to find an 'actor' subtree somewhere under params."
    )


# =============================================================
# Networks (unchanged)
# =============================================================


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
        # x: (obs, resets), obs:(T,B,D), resets:(T,B)
        x, resets = x

        # Concatenate z to obs (RL_VAE policies are trained with obs+z input dim).
        if z is not None:
            T = x.shape[0]
            if z.ndim == 2:  # (B,Z)
                z = jnp.broadcast_to(z[None, ...], (T,) + z.shape)
            x = jnp.concatenate([x, z], axis=-1)

        for feature_size in self.preprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = nn.LayerNorm()(x)
            x = self.act_fn(x)

        new_cs, new_hs = [], []
        for i, feature_size in enumerate(self.lstm_features):
            layer = self.lstm(feature_size)
            layer_carry = jax.tree_util.tree_map(lambda c: c[i], carry)
            layer_carry, x = layer(layer_carry, (x, resets))
            new_c, new_h = layer_carry
            new_cs.append(new_c)
            new_hs.append(new_h)

        embedding = x
        for feature_size in self.postprocessing_features:
            x = nn.Dense(feature_size)(x)
            x = self.act_fn(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Match rl_vae training: name the head "policy_out".
        logits = nn.Dense(self.action_dim, name="policy_out")(x)
        new_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_carry, logits, embedding

    @nn.nowrap
    def initialize_carry(self, batch_size):
        cs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in self.lstm_features]
        hs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in self.lstm_features]
        return jnp.stack(cs), jnp.stack(hs)


class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: ...

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.config.fc_dim_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config.gru_hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor_mean)
        action_logits = actor_mean - (1 - avail_actions) * 1e10
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(
            self.config.fc_dim_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class PartnerRLVAEPolicy(nn.Module):
    """A thin wrapper that turns a frozen RL_VAE actor into a policy used as BR's partner."""

    actor: ActorLstm

    @nn.compact
    def __call__(self, carry, x, training: bool = False):
        # x: (obs, dones, avail_actions, z) time-major
        obs, dones, avail_actions, z = x
        assert obs.shape[0] == 1, f"PartnerRLVAEPolicy expects T=1, got {obs.shape[0]}"

        carry, logits, _ = self.actor(carry, (obs, dones), z=z, training=training)
        action_logits = logits - (1.0 - avail_actions) * 1e10
        pi = distrax.Categorical(logits=action_logits)
        dummy_value = jnp.zeros((obs.shape[1],), dtype=jnp.float32)
        return carry, pi, dummy_value

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return self.actor.initialize_carry(batch_size)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    train_mask: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def bc_annealing_schedule(config):
    start_at_update_step = config.anneal_start // config.num_steps // config.num_envs
    end_at_update_step = config.anneal_horizon // config.num_steps // config.num_envs
    return optax.linear_schedule(
        init_value=0.0,
        end_value=1.0,
        transition_begin=start_at_update_step,
        transition_steps=end_at_update_step - start_at_update_step,
    )


# =============================================================
# Training
# =============================================================


def make_train(config, partner_actor_params_list: List, z_bank: jnp.ndarray):
    """Train a BR against per-episode sampled RL_VAE partner policies."""

    env = jaxmarl.make(config.env_name, **config.env_kwargs)
    config.num_actors = env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_steps // config.num_envs
    config.minibatch_size = config.num_actors * config.num_steps // config.num_minibatches
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        )
        return config.lr * frac

    def train(rng):
        # -----------------
        # Networks
        # -----------------
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

        partner_actor = ActorLstm(
            preprocessing_features=list(config.rl_vae_actor.preprocessing_features),
            lstm_features=list(config.rl_vae_actor.lstm_features),
            postprocessing_features=list(config.rl_vae_actor.postprocessing_features),
            action_dim=action_dim,
            dropout_rate=float(config.rl_vae_actor.dropout),
            activation_fn_name=str(config.rl_vae_act_fn),
        )
        partner_policy = PartnerRLVAEPolicy(actor=partner_actor)

        rng, _rng = jax.random.split(rng)
        K = int(z_bank.shape[0])
        z_dim = int(z_bank.shape[1])

        init_x_rl = (
            jnp.zeros((1, config.num_envs, obs_dim)),
            jnp.zeros((1, config.num_envs)),
            jnp.zeros((1, config.num_envs, action_dim)),
        )
        init_hstate = ScannedRNN.initialize_carry(config.num_envs, config.gru_hidden_dim)
        network_params = network.init(_rng, init_hstate, init_x_rl)

        # Init partner policy vars for shape-check only
        init_x_partner = (
            init_x_rl[0],
            init_x_rl[1],
            init_x_rl[2],
            jnp.zeros((1, config.num_envs, z_dim)),
        )
        partner_init_carry = partner_policy.initialize_carry(batch_size=config.num_envs)
        _ = partner_policy.init(_rng, partner_init_carry, init_x_partner)

        # -----------------
        # Optimizer / TrainState
        # -----------------
        lr = config.lr if not config.anneal_lr else linear_schedule
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=lr, eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # -----------------
        # Annealing schedule
        # -----------------
        partner_anneal_schedule = bc_annealing_schedule(config)

        # -----------------
        # Init envs
        # -----------------
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        init_hstate = ScannedRNN.initialize_carry(config.num_actors, config.gru_hidden_dim)

        # =========================
        # PLAN A CHANGE #1:
        # Partner hidden state becomes a *bank* of size K:
        #   partner_hstate_bank: (K, L, A, H)
        # =========================
        A = config.num_actors
        cs0, hs0 = partner_policy.initialize_carry(batch_size=A)  # each (L, A, H)
        partner_cs_bank = jnp.broadcast_to(cs0[None, ...], (K,) + cs0.shape)
        partner_hs_bank = jnp.broadcast_to(hs0[None, ...], (K,) + hs0.shape)
        partner_hstate_bank = (partner_cs_bank, partner_hs_bank)

        # Which seat is the BR seat? (unchanged)
        position_idxs = jnp.linspace(
            0, env.num_agents, config.num_envs, dtype=jnp.int32, endpoint=False
        )
        train_mask_dict = {a: position_idxs == i for i, a in enumerate(env.agents)}
        train_idxs_mask = batchify(train_mask_dict, env.agents, config.num_actors).squeeze()

        # Initial anneal mask
        init_partner_anneal_mask = jax.random.uniform(
            _rng, (config.num_envs,)
        ) < partner_anneal_schedule(0)

        # Initial partner index per env
        rng, _rng = jax.random.split(rng)
        partner_idx_env = jax.random.randint(_rng, (config.num_envs,), 0, K)

        def _make_train_mask(annealing_mask):
            # When annealing_mask=True for an env: only the BR seat is trained; other seat uses partner policy.
            # When annealing_mask=False: all seats use BR actions (self-play).
            full_anneal_mask = jnp.tile(annealing_mask, env.num_agents)
            return jnp.where(full_anneal_mask, train_idxs_mask, True)

        # -----------------
        # TRAIN LOOP
        # -----------------
        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                partner_hstate_bank,
                partner_anneal_mask,
                partner_idx_env,
                rng,
            ) = runner_state

            hstate_start = hstate

            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    partner_hstate_bank,
                    partner_anneal_mask,
                    partner_idx_env,
                    rng,
                ) = runner_state

                rng, _rng = jax.random.split(rng)

                # Legal moves + batchify
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config.num_actors)
                )

                obs_batch = batchify(last_obs, env.agents, config.num_actors)
                ac_in = (
                    obs_batch[None, :],  # (1, A, D)
                    last_done[None, :],  # (1, A)
                    avail_actions[None, :],  # (1, A, A_dim)
                )

                # --- BR policy action (unchanged)
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action_br = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action_br)

                # =========================
                # PLAN A CHANGE #2:
                # Compute partner actions for *all K* policies on full actor batch (Choice 1: python loop)
                # then gather selected partner action per actor using partner_idx_env.
                # RNG sampling uses fold_in(_rng, k).
                # =========================
                _E = config.num_envs
                N = env.num_agents
                A = config.num_actors
                a_ids = jnp.arange(A)

                # idx per actor: (E,) -> (A,)
                idx_actor = jnp.repeat(partner_idx_env, N, axis=0)

                # partner_hstate_bank: (K, L, A, H)
                cs_bank, hs_bank = partner_hstate_bank
                new_cs_list = []
                new_hs_list = []
                partner_action_list = []

                # We keep z_bank on device; each k uses a fixed z_k broadcast to (1, A, Z)
                for k in range(K):
                    carry_k = (cs_bank[k], hs_bank[k])  # each (L, A, H)

                    z_k = z_bank[k]  # (Z,)
                    z_tm = jnp.broadcast_to(z_k[None, None, :], (1, A, z_dim))  # (1, A, Z)

                    partner_in = (ac_in[0], ac_in[1], ac_in[2], z_tm)

                    # Apply frozen partner k
                    new_carry_k, partner_pi_k, _ = partner_policy.apply(
                        partner_actor_params_list[k],
                        carry_k,
                        partner_in,
                        # training=False by default
                    )

                    # Update hidden
                    new_cs_list.append(new_carry_k[0])
                    new_hs_list.append(new_carry_k[1])

                    # Sample actions for this partner using fold_in RNG
                    key_k = jax.random.fold_in(_rng, k)
                    a_k = partner_pi_k.sample(seed=key_k)  # shape (1, A) typically
                    a_k = jnp.reshape(a_k, (A,))  # (A,)
                    partner_action_list.append(a_k)

                # Stack hidden banks back: (K, L, A, H)
                cs_bank = jnp.stack(new_cs_list, axis=0)
                hs_bank = jnp.stack(new_hs_list, axis=0)
                partner_hstate_bank = (cs_bank, hs_bank)

                # Stack partner actions: (K, A)
                partner_action_bank = jnp.stack(partner_action_list, axis=0)

                # Gather selected partner action per actor: (A,)
                partner_action = partner_action_bank[idx_actor, a_ids]

                # --- Mix BR vs partner actions using annealing mask (unchanged)
                action_pick_mask = _make_train_mask(partner_anneal_mask)
                action = jnp.where(action_pick_mask, action_br, partner_action)

                # Step env (unchanged)
                env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
                env_act = jax.tree.map(lambda x: x.squeeze(), env_act)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree.map(lambda x: x.reshape((config.num_actors)), info)
                done_batch = batchify(done, env.agents, config.num_actors).squeeze()

                # Refresh anneal mask / partner index at episode boundaries (unchanged)
                new_partner_anneal_mask = jnp.where(
                    done["__all__"],
                    jax.random.uniform(_rng, (config.num_envs,))
                    < partner_anneal_schedule(update_steps),
                    partner_anneal_mask,
                )
                new_partner_idx_env = jnp.where(
                    done["__all__"],
                    jax.random.randint(_rng, (config.num_envs,), 0, K),
                    partner_idx_env,
                )

                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                    action_pick_mask,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    hstate,
                    partner_hstate_bank,
                    new_partner_anneal_mask,
                    new_partner_idx_env,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # Calculate advantages (unchanged)
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                partner_hstate_bank,
                partner_anneal_mask,
                partner_idx_env,
                rng,
            ) = runner_state

            last_obs_batch = batchify(last_obs, env.agents, config.num_actors)
            avail_actions_full = jnp.ones((config.num_actors, env.action_space(env.agents[0]).n))
            ac_in_last = (last_obs_batch[None, :], last_done[None, :], avail_actions_full)
            _, _, last_val = network.apply(train_state.params, hstate, ac_in_last)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.discount_factor * next_value * (1 - done) - value
                    gae = delta + config.discount_factor * config.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # PPO update (unchanged)
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        train_mask = jax.lax.stop_gradient(traj_batch.train_mask)

                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config.clip_eps, config.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(
                            where=train_mask
                        )

                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean(where=train_mask)) / (
                            gae.std(where=train_mask) + 1e-8
                        )
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean(where=train_mask)
                        entropy = pi.entropy().mean(where=train_mask)

                        approx_kl = ((ratio - 1) - logratio).mean(where=train_mask)
                        clip_frac = jnp.mean(
                            jnp.abs(ratio - 1) > config.clip_eps, where=train_mask
                        )

                        total_loss = (
                            loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                        )
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(init_hstate, (1, config.num_actors, -1))
                batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
                permutation = jax.random.permutation(_rng, config.num_actors)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x, [x.shape[0], config.num_minibatches, -1] + list(x.shape[2:])
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (train_state, hstate_start, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]

            metric = traj_batch.info
            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
            }
            metric["num_partner_envs"] = partner_anneal_mask.sum()
            metric["update_steps"] = update_steps
            metric["partner_anneal_rate"] = partner_anneal_schedule(update_steps)

            rng = update_state[-1]

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"] * config.num_envs * config.num_steps,
                        **metric["loss"],
                        "num_partner_envs": metric["num_partner_envs"],
                        "partner_anneal_rate": metric["partner_anneal_rate"],
                    },
                    step=metric["update_steps"],
                )

            jax.experimental.io_callback(callback, None, metric)

            def save_last_policy(params, update_step):
                if config.save_path is None:
                    return

                delete_path_pattern = os.path.join(
                    config.save_path, f"seed{config.seed}_step*_{config.num_players}p.*"
                )
                old_file = glob.glob(delete_path_pattern)
                if old_file:
                    # original script saved 2 files; keep same behavior
                    try:
                        os.remove(old_file[0])
                        os.remove(old_file[1])
                    except Exception:
                        pass

                save_path_params = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_step{update_step}_{config.num_players}p.safetensors",
                )
                save_params(params, save_path_params)
                save_path_config = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_step{update_step}_{config.num_players}p.yaml",
                )
                OmegaConf.save(OmegaConf.create(config.to_dict()), save_path_config)

            jax.experimental.io_callback(save_last_policy, None, train_state.params, update_steps)

            update_steps = update_steps + 1
            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                partner_hstate_bank,
                partner_anneal_mask,
                partner_idx_env,
                rng,
            )
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config.num_actors), dtype=bool),
            init_hstate,
            partner_hstate_bank,
            init_partner_anneal_mask,
            partner_idx_env,
            _rng,
        )
        runner_state, _ = jax.lax.scan(_update_step, (runner_state, 0), None, config.num_updates)
        return {"runner_state": runner_state}

    return train


# =============================================================
# Hydra entry
# =============================================================


@hydra.main(version_base=None, config_path="config", config_name="br_rl_vae_lstm_2p")
def main(config):
    OmegaConf.set_struct(config, False)

    # Load RL_VAE partner bank
    assert hasattr(config, "rl_vae_policies") and len(config.rl_vae_policies) > 0
    partner_params_list = []
    z_list = []
    for item in config.rl_vae_policies:
        params_path = item["params"]
        z_path = item["z"]
        ckpt_tree = load_flax_safetensors_any(params_path)
        partner_params_list.append(extract_actor_params_any(ckpt_tree))
        z_list.append(np.load(z_path).astype(np.float32))

    z_bank = jnp.asarray(np.stack(z_list, axis=0), dtype=jnp.float32)

    config = OmegaConf.to_container(config)
    config = ml_collections.ConfigDict(config)

    wandb.init(
        name=f"seed{config.seed}_{config.wandb_name}",
        project=config.project,
        config=config.to_dict(),
        mode=config.wandb_mode,
    )

    rng = jax.random.PRNGKey(config.seed)
    train_jit = jax.jit(make_train(config, partner_params_list, z_bank))
    _ = train_jit(rng)


if __name__ == "__main__":
    main()
