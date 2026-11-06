# ah2ac2/tools/encode_human_latents.py
# Save seat-paired latents (posterior means) from a VAE for random trajectories.

import argparse
import json
import os
from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze
from flax.traverse_util import unflatten_dict
from jaxmarl import make
from jaxmarl.environments.hanabi.hanabi import HanabiEnv

from ah2ac2.datasets.dataloader import HanabiLiveGamesDataloader

# Dataset helpers (same ones used by bc.py / vae_2p.py)
from ah2ac2.datasets.dataset import HanabiLiveGamesDataset


# -------------------------
# Utilities
# -------------------------
def load_flax_safetensors(path: str):
    """Load a Flax params tree from a .safetensors file; tolerate '/', ',' separators."""
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


def unwrap_state(s):
    return getattr(s, "env_state", s)


def batchify(x, agent_list):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((len(agent_list), -1))


def unbatchify(x, agent_list):
    return {a: x[i] for i, a in enumerate(agent_list)}


class StepCarry(nn.Module):
    pass


class Transition(nn.Module):
    pass


def make_trajectory_batcher(num_players: int):
    hand_size = 5 if num_players <= 3 else 4
    env: HanabiEnv = make("hanabi", hand_size=hand_size, num_agents=num_players)

    def get_trajectory_batch(rng: jax.Array, deck: jax.Array, actions: jax.Array):
        def _step(carry, step_actions):
            step_rng, _rng = jax.random.split(carry["rng"])
            env_act = unbatchify(step_actions, env.agents)
            obs, env_state, *_ = env.step_env(_rng, carry["prev_env_state"], env_act)

            reached_terminal = carry["prev_env_state"].terminal
            env_state = jax.lax.cond(
                reached_terminal, lambda: carry["prev_env_state"], lambda: env_state
            )
            obs = jax.lax.cond(reached_terminal, lambda: carry["prev_obs"], lambda: obs)

            legal_moves_batched = batchify(
                env.get_legal_moves(carry["prev_env_state"]), env.agents
            )
            prev_obs_batched = batchify(carry["prev_obs"], env.agents)
            act_batched = batchify(env_act, env.agents)

            new_carry = {"prev_obs": obs, "prev_env_state": env_state, "rng": step_rng}
            new_tr = (prev_obs_batched, act_batched, legal_moves_batched)
            return new_carry, new_tr

        rng, _rng = jax.random.split(rng)
        init_obs, init_state = env.reset_from_deck_of_pairs(deck)
        init = {"prev_obs": init_obs, "prev_env_state": init_state, "rng": _rng}
        _, traj = jax.lax.scan(_step, init, actions)
        # traj is a tuple of (T, A, ·) arrays
        return traj, env

    return get_trajectory_batch


def transform_to_tm(obs_t_a_f, act_t_a_1):
    """(T, A, F)/(T, A, 1) → time-major with a batch of A: (T, B=A, ·)."""
    return obs_t_a_f, act_t_a_1


# -------------------------
# Minimal encoder probe (matches the VAE encoder)
# -------------------------
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


class VAEEncoderProbe(nn.Module):
    preprocessing_features: Tuple[int, ...]
    latent_dim: int
    action_dim: int
    act_fn: str = "relu"

    @nn.compact
    def __call__(self, obs_tm: jax.Array, act_tm: jax.Array, *, training: bool = False):
        """
        Return posterior mean and log-variance for each seat in the batch.
        obs_tm: (T, B, D), act_tm: (T, B, 1)
        """
        T, B, _ = obs_tm.shape
        one_hot = jax.nn.one_hot(act_tm.squeeze(-1), self.action_dim)  # (T,B,A)
        enc_in = jnp.concatenate([obs_tm, one_hot], axis=-1).reshape(T * B, -1)

        h = _MLP(self.preprocessing_features + (128,), self.act_fn)(enc_in, training=training)
        mu = nn.Dense(self.latent_dim)(h)
        lv = nn.Dense(self.latent_dim)(h)

        # pool over time → (B, L)
        mu, lv = [y.reshape(T, B, -1).mean(0) for y in (mu, lv)]
        return mu, lv  # (B, L), (B, L)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("Encode paired (seat0, seat1) latents from a VAE.")
    ap.add_argument("--vae_ckpt", type=str, required=True, help="Path to vae_*.safetensors")
    ap.add_argument(
        "--dataset", type=str, default="ah2ac2/datasets/data/2_player_games_test.safetensors"
    )
    ap.add_argument("--num_players", type=int, default=2)
    ap.add_argument("--num_pairs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="outputs/human_latents")

    # Architecture (must match the VAE you trained)
    ap.add_argument("--preprocessing_features", type=int, nargs="+", default=[1024])
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--act_fn", type=str, default="gelu")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---------- Build env to infer action_dim ----------
    hand_size = 5 if args.num_players <= 3 else 4
    env = make("hanabi", hand_size=hand_size, num_agents=args.num_players)
    key0 = jax.random.PRNGKey(args.seed)
    _obs0, es0 = env.reset(key0)
    legal0 = env.get_legal_moves(unwrap_state(es0))
    first_agent = list(env.agents)[0]
    action_dim = int(legal0[first_agent].shape[-1])

    # ---------- Load dataset as one big batch ----------
    ds = HanabiLiveGamesDataset(file=args.dataset)
    dl = HanabiLiveGamesDataloader(ds)  # full pass (small test set)
    full = next(iter(dl))  # has .decks, .actions, .game_len_masks
    N = len(ds)

    # ---------- Sample unique indices ----------
    rng = np.random.RandomState(args.seed)
    chosen = rng.choice(N, size=min(args.num_pairs, N), replace=False).tolist()

    # ---------- Build trajectory factory ----------
    get_traj = make_trajectory_batcher(args.num_players)

    # ---------- Encoder probe (reuses trained params) ----------
    enc = VAEEncoderProbe(
        preprocessing_features=tuple(args.preprocessing_features),
        latent_dim=args.latent_dim,
        action_dim=action_dim,
        act_fn=args.act_fn,
    )
    # initialize once to materialize the param structure (shapes don't have to match exactly)
    dummy_o = jnp.ones((2, args.num_players, full.decks.shape[-1]), jnp.float32)
    dummy_a = jnp.zeros((2, args.num_players, 1), jnp.int32)
    _ = enc.init(jax.random.PRNGKey(0), dummy_o, dummy_a, training=False)

    # load trained params (full VAE params are fine; apply will pick the needed keys)
    trained_vars = load_flax_safetensors(args.vae_ckpt)
    trained_params = trained_vars["params"]  # freeze

    manifest = []
    for k, idx in enumerate(chosen):
        deck_k = full.decks[idx]
        acts_k = full.actions[idx]  # (T, A, 1)

        # Rebuild exact trajectory (obs, acts) for that single game
        traj, _ = get_traj(jax.random.PRNGKey(1234 + idx), deck_k, acts_k)
        obs_t_a_f, act_t_a_1, _ = traj  # each is (T, A, ·)

        # Time-major encoder inputs (T, B=A, ·)
        obs_tm, act_tm = obs_t_a_f, act_t_a_1

        # Posterior means per seat: (A=2, L)
        mu_b_l, _ = enc.apply({"params": trained_params}, obs_tm, act_tm, training=False)

        z_pair = np.asarray(mu_b_l)  # (2, L)
        z_p0 = z_pair[0]
        z_p1 = z_pair[1]

        # Save three artifacts for convenience
        base = os.path.join(args.outdir, f"pair_{k:03d}")
        np.save(base + "_both.npy", z_pair)
        np.save(base + "_p0.npy", z_p0)
        np.save(base + "_p1.npy", z_p1)

        manifest.append(
            {
                "pair_id": k,
                "dataset_index": int(idx),
                "paths": {
                    "both": base + "_both.npy",
                    "p0": base + "_p0.npy",
                    "p1": base + "_p1.npy",
                },
                "shapes": {
                    "both": list(z_pair.shape),
                    "p0": list(z_p0.shape),
                    "p1": list(z_p1.shape),
                },
            }
        )

    with open(os.path.join(args.outdir, "manifest.json"), "w") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "num_players": args.num_players,
                "latent_dim": args.latent_dim,
                "action_dim": action_dim,
                "pairs": manifest,
            },
            f,
            indent=2,
        )

    print(f"Saved {len(manifest)} latent pairs to {args.outdir}")
    for row in manifest:
        print(f"  pair_{row['pair_id']:03d}: p0={row['paths']['p0']}  p1={row['paths']['p1']}")


if __name__ == "__main__":
    main()
