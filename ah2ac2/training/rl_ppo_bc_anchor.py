import os
from typing import List, NamedTuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import wandb
from flax.core import freeze, unfreeze
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import unflatten_dict
from hydra.utils import to_absolute_path
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, save_params
from omegaconf import OmegaConf

from ah2ac2.nn.multi_layer_lstm import LstmCellWithHiddenStateReset, MultiLayerLstm

# =============================================================
# Utilities
# =============================================================


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
    if "params" not in tree:  # normalize to {'params': ...}
        tree = {"params": tree}
    return freeze(tree)


def zeros_lstm_carry(lstm_features: List[int], batch_size: int):
    """Create zero carry for a stacked LSTM: (Cs, Hs) with shape (num_layers, B, hidden)."""
    cs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in lstm_features]
    hs = [jnp.zeros((batch_size, h), dtype=jnp.float32) for h in lstm_features]
    return jnp.stack(cs), jnp.stack(hs)


def reset_carry_on_done(carry, done_flags):
    """Zero the carry (all layers) where done_flags==True.
    carry: (Cs, Hs) with shapes (L,B,H_l)
    done_flags: (B,) bool
    """
    cs, hs = carry
    mask = (1.0 - done_flags.astype(jnp.float32))[None, :, None]  # (1,B,1)
    return cs * mask, hs * mask


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def categorical_kl_from_logits(logits_p, logits_q):
    """KL(p || q) with both distributions specified by unnormalized logits."""
    logp = jax.nn.log_softmax(logits_p, axis=-1)
    logq = jax.nn.log_softmax(logits_q, axis=-1)
    p = jnp.exp(logp)
    return jnp.sum(p * (logp - logq), axis=-1)


# =============================================================
# BC anchor adapter + loader
# =============================================================
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
    def __call__(self, carry, x, training: bool = False):
        # x = (obs, resets) with obs: (T,B,D) where T=1 in this codepath
        obs, resets = x
        x_bt = jnp.swapaxes(obs, 0, 1)  # (B,1,D)
        carry, logits_bt = self.net(x=x_bt, carry=carry, training=training)  # (B,1,A)
        logits = jnp.swapaxes(logits_bt, 0, 1)  # (1,B,A)
        return carry, logits, logits

    @nn.nowrap
    def initialize_carry(self, batch_size: int):
        return zeros_lstm_carry(self.lstm_features, batch_size)


def load_bc_anchor_vars(params_path: str):
    """Wrap a BC MultiLayerLstm checkpoint as {'params': {'net': ...}} for the adapter."""
    full_vars = load_flax_safetensors(params_path)  # {'params': ...}
    p = unfreeze(full_vars)
    assert "params" in p, f"'params' not found in {params_path}"
    return freeze({"params": {"net": p["params"]}})


# =============================================================
# Actor / Critic (policy being trained) with Advantage head for Q-hat
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

        return new_carry, policy_logits, embedding

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return zeros_lstm_carry(self.lstm_features, batch_size)


class ActorCriticRNN(nn.Module):
    actor: ActorLstm
    critic_features: int

    @nn.compact
    def __call__(self, hidden, x, z_batch: jnp.ndarray = None):
        obs, dones, avail_actions = x

        # Actor:
        hidden, actor_logits, embedding = self.actor(hidden, (obs, dones), z=z_batch)
        action_logits = actor_logits - (1.0 - avail_actions) * 1e10
        pi = distrax.Categorical(logits=action_logits)

        # Critic (on actor embedding at T=1):
        critic = nn.Dense(
            self.critic_features, kernel_init=orthogonal(2.0), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        value = jnp.squeeze(value, axis=-1)  # (T,B)

        return hidden, pi, value

    @nn.nowrap
    def initialize_carry(self, batch_size):
        return self.actor.initialize_carry(batch_size)


# =============================================================
# Training
# =============================================================
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
    anchor_logits: jnp.ndarray


# ---------- Main training factory ----------


def make_train(config):
    env = make("hanabi", num_agents=config.num_players)
    env = LogWrapper(env)

    # Derived config
    config.num_actors = env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_steps // config.num_envs
    config.minibatch_size = config.num_actors * config.num_steps // config.num_minibatches
    config.num_test_actors = env.num_agents * config.num_test_envs

    # Defaults for KL controller
    if "bc_policy_kl_weight" not in config:
        config.bc_policy_kl_weight = 0.0
    if "kl_target" not in config:
        config.kl_target = 0.02
    if "kl_alpha" not in config:
        config.kl_alpha = 0.05
    if "kl_max" not in config:
        config.kl_max = 10.0

    action_dim = 21
    obs_dim = 658

    def linear_schedule(count):
        frac = (
            1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
        )
        return config.lr * frac

    # --- Build BC anchor ---
    bc_anchor_module = ActorLstmBCAdapter(
        preprocessing_features=config.network.preprocessing_features,
        lstm_features=config.network.lstm_features,
        postprocessing_features=config.network.postprocessing_features,
        action_dim=action_dim,
        dropout_rate=config.network.dropout,
        activation_fn_name=config.network.act_fn,
    )
    bc_anchor_vars = load_bc_anchor_vars(to_absolute_path(config.anchor_bc_params_path))

    def train(rng):
        # INIT NETWORK (actor init = BC decoder params, policy consumes fixed z as extra input)
        actor_module = ActorLstm(
            preprocessing_features=config.network.preprocessing_features,
            lstm_features=config.network.lstm_features,
            postprocessing_features=config.network.postprocessing_features,
            action_dim=action_dim,
            dropout_rate=config.network.dropout,
            activation_fn_name=config.network.act_fn,
        )
        network = ActorCriticRNN(
            actor=actor_module, critic_features=config.network.critic_features
        )

        # Initialize param tree
        init_x = (
            jnp.zeros((1, config.num_envs, obs_dim), dtype=jnp.float32),
            jnp.zeros((1, config.num_envs), dtype=jnp.float32),
            jnp.zeros((1, config.num_envs, action_dim), dtype=jnp.float32),
        )
        init_hstate = network.initialize_carry(batch_size=config.num_envs)

        rng, _rng = jax.random.split(rng)
        network_params = network.init(_rng, init_hstate, init_x)

        def shape_safe_update(dst, src, skip_top_keys=()):
            for k, v in src.items():
                if k in skip_top_keys:
                    continue
                if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
                    shape_safe_update(dst[k], v, skip_top_keys=())
                elif k in dst:
                    # only copy when both are array-like and shapes match
                    try:
                        if getattr(dst[k], "shape", None) == getattr(v, "shape", None):
                            dst[k] = v
                    except Exception:
                        pass

        params = unfreeze(network_params)
        bc_net = unfreeze(bc_anchor_vars)["params"]["net"]

        # Skip actor heads completely to avoid any chance of collision
        shape_safe_update(params["params"]["actor"], bc_net, skip_top_keys=("policy_out",))

        network_params = freeze(params)

        lr = config.lr if not config.lr_linear_schedule else linear_schedule
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(learning_rate=lr, eps=1e-5),
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENVS
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=0)(reset_rng)
        hstate = network.initialize_carry(config.num_actors)

        # Init anchor carry for all actors
        anchor_carry = zeros_lstm_carry(list(config.network.lstm_features), config.num_actors)

        # Greedy evaluation (unchanged)
        def get_greedy_metrics(rng, train_state):
            def _greedy_env_step(runner_state, _):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                legal_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                legal_actions = jax.lax.stop_gradient(
                    batchify(legal_actions, env.agents, config.num_test_actors)
                )

                obs_batch = batchify(last_obs, env.agents, config.num_test_actors)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    legal_actions[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

                action = jnp.argmax(pi.probs, axis=-1)
                env_act = unbatchify(action, env.agents, config.num_test_envs, env.num_agents)
                env_act = jax.tree.map(lambda x: x.squeeze(), env_act)

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_test_envs)
                obsv, env_state, step_rewards, step_dones, step_infos = jax.vmap(
                    env.step, in_axes=0
                )(rng_step, env_state, env_act)

                done_batch = batchify(step_dones, env.agents, config.num_test_actors).squeeze()
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, (step_infos, step_rewards, step_dones)

            rng, _rng = jax.random.split(rng)
            test_reset_rng = jax.random.split(_rng, config.num_test_envs)
            test_obsv, test_env_state = jax.vmap(env.reset, in_axes=0)(test_reset_rng)
            test_init_hstate = network.initialize_carry(config.num_test_actors)

            test_runner_state = (
                train_state,
                test_env_state,
                test_obsv,
                jnp.zeros((config.num_test_actors), dtype=bool),
                test_init_hstate,
                _rng,
            )
            test_runner_state, (final_infos, final_rewards, final_dones) = jax.lax.scan(
                _greedy_env_step, test_runner_state, None, 89
            )

            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(
                    jnp.argmax(dones) == 0.0, dones.size, jnp.argmax(dones)
                )
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                first_episode_rewards = jnp.where(first_episode_mask, rewards, 0.0)
                return first_episode_rewards.sum()

            first_returns = jax.vmap(first_episode_returns, in_axes=1)(
                final_rewards["__all__"], final_dones["__all__"]
            )
            return {
                "mean": jnp.nanmean(first_returns),
                "max": jnp.nanmax(first_returns),
                "min": jnp.nanmin(first_returns),
                "std": jnp.nanstd(first_returns),
                "perfect": jnp.nansum(first_returns == 25),
                "zero": jnp.nansum(first_returns == 0),
            }

        # UPDATE loop body
        def _update_step(update_runner_state, _):
            # COLLECT TRAJECTORIES
            runner_state, (update_steps, best_greedy_score, kl_lambda) = update_runner_state

            train_state, env_state, last_obs, last_done, hstate0, anchor_carry, rng = runner_state

            def _env_step(runner_state, _):
                train_state, env_state, last_obs, last_done, hstate, anchor_carry, rng = (
                    runner_state
                )

                # Legal actions
                legal_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                legal_actions = jax.lax.stop_gradient(
                    batchify(legal_actions, env.agents, config.num_actors)
                )

                obs_batch = batchify(last_obs, env.agents, config.num_actors)
                anchor_carry = reset_carry_on_done(anchor_carry, last_done)
                anchor_inputs = (obs_batch[None, :], last_done[None, :])  # T=1
                anchor_carry, anchor_logits, _ = bc_anchor_module.apply(
                    bc_anchor_vars, anchor_carry, anchor_inputs, training=False
                )
                anchor_logits = anchor_logits.squeeze(0)  # (B, A)
                anchor_logits = anchor_logits - (1.0 - legal_actions) * 1e10

                # Policy step
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    legal_actions[np.newaxis, :],
                )
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)

                rng, _rng = jax.random.split(rng)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config.num_envs, env.num_agents)
                env_act = jax.tree.map(lambda x: x.squeeze(), env_act)

                # Step env
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=0)(
                    rng_step, env_state, env_act
                )
                info = jax.tree.map(lambda x: x.reshape((config.num_actors)), info)
                done_batch = batchify(done, env.agents, config.num_actors).squeeze()

                transition = Transition(
                    global_done=jnp.tile(done["__all__"], env.num_agents),
                    done=last_done,
                    action=action.squeeze(),
                    value=value.squeeze(),
                    reward=batchify(reward, env.agents, config.num_actors).squeeze(),
                    log_prob=log_prob.squeeze(),
                    obs=obs_batch,
                    info=info,
                    avail_actions=legal_actions,
                    anchor_logits=anchor_logits,
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    hstate,
                    anchor_carry,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config.num_steps
            )

            # ADVANTAGES
            train_state, env_state, last_obs, last_done, hstate, anchor_carry, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config.num_actors)
            legal_actions_full = jnp.ones((config.num_actors, action_dim), dtype=jnp.float32)
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
                legal_actions_full[np.newaxis, :],
            )
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
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

            # GREEDY EVAL
            rng, _rng = jax.random.split(rng)
            greedy_metrics = get_greedy_metrics(_rng, train_state)

            # CHECKPOINTING (unchanged, elided)
            def save_best_policy(best_score, current_score, params):
                if config.save_path is None or best_score >= current_score:
                    return
                prev_prefix = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_score{best_score:.3f}_klt{config.kl_target}_{config.num_players}p",
                )
                for suf in (".safetensors", ".yaml", "_z.npy", "_z0.npy", "_z1.npy"):
                    try:
                        os.remove(prev_prefix + suf)
                    except FileNotFoundError:
                        pass

                prefix = os.path.join(
                    config.save_path,
                    f"seed{config.seed}_score{current_score:.3f}_klt{config.kl_target}_{config.num_players}p",
                )
                save_params(params, prefix + ".safetensors")
                OmegaConf.save(OmegaConf.create(config.to_dict()), prefix + ".yaml")

            jax.experimental.io_callback(
                save_best_policy,
                None,
                best_greedy_score,
                greedy_metrics["mean"],
                train_state.params,
            )

            # def save_last_policy(current_score, params):
            #     if not config.save_path or not getattr(config, "save_last_policy", False):
            #         return
            #     pattern = os.path.join(
            #         config.save_path,
            #         f"recent_seed{config.seed}_score*_klt{config.kl_target}_{config.num_players}p*",
            #     )
            #     for fp in glob.glob(pattern):
            #         try:
            #             os.remove(fp)
            #         except OSError:
            #             pass

            #     prefix = os.path.join(
            #         config.save_path,
            #         f"recent_seed{config.seed}_score{current_score:.3f}_klt{config.kl_target}_{config.num_players}p",
            #     )
            #     save_params(params, prefix + ".safetensors")
            #     OmegaConf.save(OmegaConf.create(config.to_dict()), prefix + ".yaml")

            # jax.experimental.io_callback(
            #     save_last_policy, None, greedy_metrics["mean"], train_state.params
            # )

            # UPDATE NETWORK (PPO + anchor KL)
            def _update_epoch(update_state, _):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets, kl_lambda = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets, kl_lambda):
                        # Re-run policy
                        ac_ins = (traj_batch.obs, traj_batch.done, traj_batch.avail_actions)
                        _, pi, value = network.apply(params, init_hstate, ac_ins)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config.clip_eps, config.clip_eps
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Policy loss (clipped)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.clip_eps,
                                1.0 + config.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()

                        # Diagnostics
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config.clip_eps)

                        # Anchor KL: KL(anchor || policy)
                        pi_logits = pi.logits_parameter()
                        anchor_kl = categorical_kl_from_logits(
                            traj_batch.anchor_logits, pi_logits
                        ).mean()

                        ppo_loss = (
                            loss_actor + config.vf_coef * value_loss - config.ent_coef * entropy
                        )
                        total_loss = ppo_loss + kl_lambda * anchor_kl
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                            anchor_kl,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets, kl_lambda
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng, kl_lambda = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                permutation = jax.random.permutation(_rng, config.num_actors)

                batch_no_lambda = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                shuffled_no_lambda = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch_no_lambda
                )

                minibatches_no_lambda = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x, [x.shape[0], config.num_minibatches, -1] + list(x.shape[2:])
                        ),
                        1,
                        0,
                    ),
                    shuffled_no_lambda,
                )

                kl_lambda_mb = jnp.repeat(kl_lambda[None], config.num_minibatches, axis=0)
                minibatches = (*minibatches_no_lambda, kl_lambda_mb)

                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)

                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                    kl_lambda,
                )
                return update_state, total_loss

            # Pack initial states for update
            init_hstate_for_update = jax.lax.stop_gradient(hstate0)
            update_state = (
                train_state,
                init_hstate_for_update,
                traj_batch,
                advantages,
                targets,
                rng,
                kl_lambda,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]

            measured_kl = jnp.mean(loss_info[1][6])  # anchor_kl
            measured_kl = jnp.nan_to_num(measured_kl, nan=0.0, posinf=1e6, neginf=0.0)

            new_kl_lambda = jnp.clip(
                kl_lambda + config.kl_alpha * (measured_kl - config.kl_target),
                0.0,
                config.kl_max,
            )

            metric = traj_batch.info
            ratio_0 = loss_info[1][3].at[0, 0].get().mean()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            # Log core + theory
            metric["loss"] = {
                "total_loss": loss_info[0],
                "value_loss": loss_info[1][0],
                "actor_loss": loss_info[1][1],
                "entropy": loss_info[1][2],
                "ratio": loss_info[1][3],
                "ratio_0": ratio_0,
                "approx_kl": loss_info[1][4],
                "clip_frac": loss_info[1][5],
                "anchor_kl": loss_info[1][6],
                "measured_anchor_kl": measured_kl,
                "kl_lambda_used": kl_lambda,
                "kl_lambda_next": new_kl_lambda,
                "kl_target": jnp.array(config.kl_target),
            }
            metric["greedy_metrics"] = greedy_metrics
            metric["update_steps"] = update_steps
            rng = update_state[-2]

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"] * config.num_envs * config.num_steps,
                        **metric["loss"],
                        "greedy_metrics": metric["greedy_metrics"],
                    }
                )

            jax.experimental.io_callback(callback, None, metric)

            update_steps = update_steps + 1
            new_best_greedy_score = jnp.maximum(best_greedy_score, greedy_metrics["mean"])

            # Keep runner_state hot for next outer update
            runner_state = (train_state, env_state, last_obs, last_done, hstate, anchor_carry, rng)
            return (
                runner_state,
                (update_steps, new_best_greedy_score, new_kl_lambda),
            ), None

        # Seed, set up initial runner_state and scan updates
        rng, _rng = jax.random.split(rng)

        kl_lambda0 = jnp.array(config.bc_policy_kl_weight, dtype=jnp.float32)

        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros(config.num_actors, dtype=bool),
            hstate,
            anchor_carry,
            _rng,
        )
        jax.lax.scan(
            _update_step,
            (runner_state, (0, 0.0, kl_lambda0)),
            None,
            config.num_updates,
        )
        return {"runner_state": runner_state}

    return train


# =============================================================
# Hydra entry
# =============================================================
@hydra.main(version_base=None, config_path="config", config_name="rl_ppo_bc_anchor_2p")
def main(config):
    OmegaConf.set_struct(config, False)

    # Optionally merge in a network checkpoint config if provided
    if config.network.get("config_checkpoint_path"):
        network_config = {"network": OmegaConf.load(config.network.config_checkpoint_path)}
        config = OmegaConf.merge(config, network_config)

    if "network" in config and "seed" in config.network:
        config.seed = config.network.seed

    config = OmegaConf.to_container(config)
    config = ml_collections.ConfigDict(config)

    wandb.init(
        name=f"seed{config.seed}_{config.wandb_name}",
        project=config.wandb_project,
        config=config.to_dict(),
        mode=config.wandb_mode,
    )

    rng = jax.random.PRNGKey(config.seed)
    train_jit = jax.jit(make_train(config))
    _ = train_jit(rng)


if __name__ == "__main__":
    main()
