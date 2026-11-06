"""H-Group convention audit — rollout generation (Phase 2).

Generates K=500 self-play rollouts per bot for the 4 latent (rl_vae) and 1
BC (rl_bc) bots. Outputs go under ``results/hgroup_audit/rollouts/{bot_id}/``.

Pre-registered constants (LOCKED at Phase 1, do not modify post-pre-reg):

* ``K = 500``                           — rollouts per bot
* ``SEEDS = list(range(K))``            — same seed set across bots
* ``SAMPLING = "greedy"``               — masked argmax over logits
* ``BOTS``                              — 5 checkpoint paths
* ``OUTPUT_DIR = "results/hgroup_audit/rollouts"``

Spec reference: ``hgroup_audit_spec.md`` §5.1.

Usage::

    python -m ah2ac2.tools.hgroup_rollout                # all 5 bots
    python -m ah2ac2.tools.hgroup_rollout --bot Z_i000   # single bot
    python -m ah2ac2.tools.hgroup_rollout --smoke        # 4 games × Z_i000
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

# ---------------------------------------------------------------------------
# Pre-registration constants (do not modify; deviations -> DEVIATIONS.md)
# ---------------------------------------------------------------------------

K: int = 500
SEEDS: list[int] = list(range(K))
SAMPLING: str = "greedy"
OUTPUT_DIR: str = "results/hgroup_audit/rollouts"

BOTS: dict[str, dict[str, str | None]] = {
    "Z_i000": {
        "type": "rl_vae",
        "checkpoint": "checkpoints/reconstructed_partners_2k/es_final_Z_i000_SP=4.148_XPavg=1.349/seed1_score16.832_klt0.114_2p.safetensors",
        "z_path": "checkpoints/vae_lowdim_masked/human_vae/latent_dim32/2k/es_refine_4/es_final_Z_i000_SP=4.148_XPavg=1.349.npy",
    },
    "Z_i001": {
        "type": "rl_vae",
        "checkpoint": "checkpoints/reconstructed_partners_2k/es_final_Z_i001_SP=3.789_XPavg=1.385/seed1_score17.014_klt0.119_2p.safetensors",
        "z_path": "checkpoints/vae_lowdim_masked/human_vae/latent_dim32/2k/es_refine_4/es_final_Z_i001_SP=3.789_XPavg=1.385.npy",
    },
    "Z_i002": {
        "type": "rl_vae",
        "checkpoint": "checkpoints/reconstructed_partners_2k/es_final_Z_i002_SP=6.516_XPavg=2.151/seed1_score19.031_klt0.0947_2p.safetensors",
        "z_path": "checkpoints/vae_lowdim_masked/human_vae/latent_dim32/2k/es_refine_4/es_final_Z_i002_SP=6.516_XPavg=2.151.npy",
    },
    "Z_i003": {
        "type": "rl_vae",
        "checkpoint": "checkpoints/reconstructed_partners_2k/es_final_Z_i003_SP=6.703_XPavg=1.969/seed1_score19.988_klt0.094_2p.safetensors",
        "z_path": "checkpoints/vae_lowdim_masked/human_vae/latent_dim32/2k/es_refine_4/es_final_Z_i003_SP=6.703_XPavg=1.969.npy",
    },
    "BC": {
        "type": "rl_bc",
        "checkpoint": "checkpoints/kl_anchored/human_2k_0.404/kl_target0.26/seed1_score21.871_klt0.26_2p.safetensors",
        "z_path": None,
    },
}

# Per-bot architecture (read from each checkpoint's YAML — all are identical).
ACTOR_ARCH = {
    "preprocessing_features": [1024],
    "lstm_features": [512],
    "postprocessing_features": [256],
    "dropout_rate": 0.0,
    "activation_fn_name": "gelu",
}

# Hanabi-2p constants (used at runtime, not metric-bearing).
NUM_PLAYERS = 2
MAX_T = 89  # safe upper bound for game length in 2p Hanabi
DECK_SIZE = 50
ACTION_DIM = 21  # 5 play + 5 discard + 5 color clue + 5 rank clue + 1 noop


# ---------------------------------------------------------------------------
# Public API: anyone importing this module gets the locked constants AND can
# call run_rollouts(...) to generate rollouts.
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry. See module docstring for usage."""
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument(
        "--bot",
        type=str,
        default=None,
        help="Run only this bot id (one of: " + ", ".join(BOTS) + "). Default: all.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test mode: K=4, only Z_i000 unless --bot is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Parallel games per JIT call. Lower this if you OOM on larger bots.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Override the locked OUTPUT_DIR (e.g. for smoke tests).",
    )
    args = parser.parse_args()

    bots_to_run = [args.bot] if args.bot else list(BOTS)
    for b in bots_to_run:
        if b not in BOTS:
            raise SystemExit(f"Unknown bot id: {b}. Known: {list(BOTS)}")

    k = 4 if args.smoke else K
    seeds = list(range(k))

    print(
        f"[hgroup_rollout] {len(bots_to_run)} bot(s) × K={k} games "
        f"(SAMPLING={SAMPLING}, output_dir={args.output_dir})"
    )

    # Late imports — heavy JAX setup only happens when rollout actually runs,
    # so `import hgroup_rollout` at Phase 1 / Phase 3 stays cheap.
    run_rollouts(
        bots=bots_to_run,
        seeds=seeds,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def run_rollouts(
    *,
    bots: list[str],
    seeds: list[int],
    output_dir: str,
    batch_size: int,
) -> None:
    """Run self-play rollouts for each bot over the given seeds; save safetensors."""
    # Heavy imports deferred to runtime so module remains importable in
    # environments without JAX/jaxmarl (e.g. unit-test workers).
    import jax
    import numpy as np
    from jaxmarl import make

    # Build env once. We use the raw env (not LogWrapper) so we can call
    # reset_from_deck_of_pairs (deterministic, deck-as-input).
    env = make("hanabi", num_agents=NUM_PLAYERS)
    agents_tuple = tuple(env.agents)
    print(f"[hgroup_rollout] env ready (agents={agents_tuple})")

    # Sanity: action_dim = 21 in 2p Hanabi.
    rng0 = jax.random.PRNGKey(0)
    obs0, st0 = env.reset(rng0)
    legal0 = env.get_legal_moves(st0)
    action_dim_runtime = int(np.asarray(legal0[agents_tuple[0]]).shape[-1])
    assert action_dim_runtime == ACTION_DIM, (
        f"Expected action_dim={ACTION_DIM}, got {action_dim_runtime} from env."
    )

    for bot_id in bots:
        run_one_bot(
            bot_id=bot_id,
            seeds=seeds,
            env=env,
            agents_tuple=agents_tuple,
            output_dir=output_dir,
            batch_size=batch_size,
        )


def run_one_bot(
    *,
    bot_id: str,
    seeds: list[int],
    env: Any,
    agents_tuple: tuple,
    output_dir: str,
    batch_size: int,
) -> None:
    import jax
    import jax.numpy as jnp
    import numpy as np
    from safetensors.numpy import save_file

    bot_cfg = BOTS[bot_id]
    print(f"\n[hgroup_rollout] === {bot_id} ({bot_cfg['type']}) ===")
    actor, vars_, z = load_actor(bot_id)

    # Prepare jit'd vmapped rollout. Bind closure over `actor`, `vars_`, `z`.
    rollout_fn = build_rollout_fn(
        env=env,
        agents_tuple=agents_tuple,
        actor=actor,
        vars_=vars_,
        z=z,
    )
    rollout_many = jax.jit(jax.vmap(rollout_fn, in_axes=(0,)))

    # Output buffers
    n = len(seeds)
    actions_arr = np.full((n, MAX_T, NUM_PLAYERS), ACTION_DIM - 1, dtype=np.int32)
    decks_arr = np.zeros((n, DECK_SIZE, 2), dtype=np.int32)
    num_actions = np.zeros((n,), dtype=np.int32)
    scores_arr = np.zeros((n,), dtype=np.int32)
    seeds_arr = np.asarray(seeds, dtype=np.int32)

    rng_keys = jnp.stack([jax.random.PRNGKey(s) for s in seeds])

    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_keys = rng_keys[start:end]
        batch_actions, batch_decks, batch_counts, batch_scores = rollout_many(batch_keys)
        actions_arr[start:end] = np.asarray(batch_actions)
        decks_arr[start:end] = np.asarray(batch_decks)
        num_actions[start:end] = np.asarray(batch_counts)
        scores_arr[start:end] = np.asarray(batch_scores)
        elapsed = time.time() - t0
        print(
            f"  games {end:>4}/{n} | last batch: {elapsed:.1f}s elapsed | "
            f"score-so-far={scores_arr[:end].mean():.2f}±{scores_arr[:end].std():.2f}"
        )
    elapsed = time.time() - t0

    print(
        f"[hgroup_rollout] {bot_id}: {n} games in {elapsed:.1f}s | "
        f"mean score = {scores_arr.mean():.3f} ± {scores_arr.std() / np.sqrt(n):.3f}"
    )

    # Save
    out_dir = os.path.join(output_dir, bot_id)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "games.safetensors")
    save_file(
        {
            "actions": actions_arr,
            "decks": decks_arr,
            "num_actions": num_actions,
            "scores": scores_arr,
            "seeds": seeds_arr,
        },
        out_path,
    )
    # Also write a small JSON metadata file for human inspection
    import json

    meta = {
        "bot_id": bot_id,
        "type": bot_cfg["type"],
        "checkpoint": bot_cfg["checkpoint"],
        "z_path": bot_cfg["z_path"],
        "K": n,
        "seeds_first_last": [int(seeds_arr[0]), int(seeds_arr[-1])],
        "sampling": SAMPLING,
        "mean_score": float(scores_arr.mean()),
        "score_std": float(scores_arr.std()),
        "score_sem": float(scores_arr.std() / np.sqrt(n)),
        "max_T": MAX_T,
        "action_dim": ACTION_DIM,
        "num_players": NUM_PLAYERS,
        "elapsed_seconds": float(elapsed),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[hgroup_rollout] saved -> {out_path}")


def load_actor(bot_id: str):
    """Load actor module + vars + z for the given bot id."""
    import jax.numpy as jnp
    import numpy as np

    # crossplay_eval has the canonical loaders for rl/rl_vae/rl_bc and the
    # ActorLstm definitions that match the training-time checkpoints.
    from ah2ac2.tools.crossplay_eval import (
        ActorLstm,
        ActorLstmPolicyOut,
        actor_uses_policy_out,
        load_actor_vars_for_rl,
    )

    cfg = BOTS[bot_id]
    ckpt = cfg["checkpoint"]
    actor_vars = load_actor_vars_for_rl(ckpt)
    use_policy_out = actor_uses_policy_out(actor_vars["params"])
    cls = ActorLstmPolicyOut if use_policy_out else ActorLstm
    actor = cls(
        action_dim=ACTION_DIM,
        **ACTOR_ARCH,
    )
    if cfg["type"] == "rl_vae":
        z_arr = np.load(cfg["z_path"]).astype(np.float32)
        z = jnp.asarray(z_arr)
    else:  # rl_bc
        z = jnp.zeros((0,), dtype=jnp.float32)
    print(
        f"[hgroup_rollout] loaded {bot_id}: actor={cls.__name__}, "
        f"z.shape={tuple(z.shape)}, ckpt={os.path.basename(ckpt)}"
    )
    return actor, actor_vars, z


def build_rollout_fn(*, env, agents_tuple, actor, vars_, z):
    """Build a JIT-friendly self-play rollout function for ONE rng key."""
    import jax
    import jax.numpy as jnp

    num_players = NUM_PLAYERS
    action_dim = ACTION_DIM
    noop_id = action_dim - 1
    max_T = MAX_T
    _deck_size = DECK_SIZE

    def sample_deck_pairs(key):
        """Build a (50,2) (color, rank) array with multiplicities [3,2,2,2,1].

        Ranks are 1..5 (matches the deck convention in jaxmarl Hanabi).
        """
        mult = [3, 2, 2, 2, 1]
        colors_l = []
        ranks_l = []
        for c in range(5):
            for r, n in enumerate(mult):
                colors_l.extend([c] * n)
                ranks_l.extend([r] * n)
        deck = jnp.stack(
            [
                jnp.array(colors_l, dtype=jnp.int32),
                jnp.array(ranks_l, dtype=jnp.int32),
            ],
            axis=1,
        )
        perm = jax.random.permutation(key, deck.shape[0])
        return deck[perm]

    def rollout_one(rng):
        rng, k_deck = jax.random.split(rng)
        deck_pairs = sample_deck_pairs(k_deck)
        obs, env_state = env.reset_from_deck_of_pairs(deck_pairs)
        legal = env.get_legal_moves(env_state)
        hidden = actor.initialize_carry(num_players)

        def step_fn(carry, _):
            rng, env_state, obs, legal, hidden, terminated, score = carry

            def do_pad(c):
                rng, env_state, obs, legal, hidden, terminated, score = c
                act_rec = jnp.full((num_players,), noop_id, dtype=jnp.int32)
                return c, (act_rec, jnp.bool_(True), jnp.array(0.0, dtype=jnp.float32))

            def do_step(c):
                rng, env_state, obs, legal, hidden, terminated, score = c
                # Stack agent obs / legal moves into batched tensors.
                obs_b = jnp.stack([obs[a] for a in agents_tuple])  # (B, D)
                legal_b = jnp.stack([legal[a] for a in agents_tuple])  # (B, A)

                # ActorLstm expects (T, B, D) with T=1 and resets (T, B).
                x = obs_b[None, :, :]  # (1, B, D)
                resets = jnp.zeros((1, num_players), dtype=jnp.bool_)
                # z arg: replicate across the batch (B, L). For rl_bc z.size==0, pass None.
                z_arg = (
                    None if z.size == 0 else jnp.broadcast_to(z[None, :], (num_players,) + z.shape)
                )

                new_hidden, logits, _ = actor.apply(
                    vars_, hidden, (x, resets), z=z_arg, training=False
                )
                # logits: (1, B, A) — drop time dim.
                masked = logits[0] - (1.0 - legal_b) * 1e10
                actions_per_seat = jnp.argmax(masked, axis=-1).astype(jnp.int32)  # (B,)

                # Only the current player's action is taken; others get noop.
                cur_idx = jnp.argmax(env_state.cur_player_idx).astype(jnp.int32)
                act_rec = jnp.full((num_players,), noop_id, dtype=jnp.int32)
                act_rec = act_rec.at[cur_idx].set(actions_per_seat[cur_idx])

                env_act = {agents_tuple[i]: act_rec[i] for i in range(num_players)}
                rng, k = jax.random.split(rng)
                obs2, env_state2, reward, dones, _info = env.step_env(k, env_state, env_act)
                legal2 = env.get_legal_moves(env_state2)

                done_all = dones["__all__"].astype(jnp.bool_)
                score2 = score + jnp.asarray(reward["__all__"]).astype(jnp.float32)
                new_carry = (
                    rng,
                    env_state2,
                    obs2,
                    legal2,
                    new_hidden,
                    terminated | done_all,
                    score2,
                )
                return new_carry, (
                    act_rec,
                    done_all,
                    jnp.asarray(reward["__all__"], dtype=jnp.float32),
                )

            return jax.lax.cond(terminated, do_pad, do_step, carry)

        carry0 = (rng, env_state, obs, legal, hidden, jnp.bool_(False), jnp.array(0.0))
        (_, _, _, _, _, term_final, score_final), (acts_TA, terms_T, _rews_T) = jax.lax.scan(
            step_fn, carry0, xs=None, length=max_T
        )
        # Number of real (non-padded) steps = number of False entries in terms_T BEFORE the
        # first True (terms_T flips True at the step that ends the game and stays True).
        # Simpler: count entries that ran do_step. After do_step the carry's `terminated`
        # only flips True at the step that produced the done signal; that step IS a real
        # step. So num_actions = number of `terminated == False BEFORE this step` = first
        # index where terminated is True.
        # terms_T[t] is the do_step's `done_all` for step t (True at the terminal step).
        # In padded steps it's True. So the run length is the index of the first True + 1.
        is_done = terms_T  # (max_T,) bool
        # Cumulative OR over time: True from the terminal step onwards.
        cum_done = jnp.cumsum(is_done.astype(jnp.int32)) > 0
        num_real = jnp.sum(jnp.logical_not(cum_done)) + 1  # +1 for the terminal step itself
        num_real = jnp.minimum(num_real, max_T).astype(jnp.int32)
        return (
            acts_TA,
            deck_pairs,
            num_real,
            jnp.round(score_final).astype(jnp.int32),
        )

    return rollout_one


__all__ = [
    "K",
    "SEEDS",
    "SAMPLING",
    "OUTPUT_DIR",
    "BOTS",
    "ACTOR_ARCH",
    "MAX_T",
    "ACTION_DIM",
    "NUM_PLAYERS",
    "main",
    "run_rollouts",
    "load_actor",
    "build_rollout_fn",
]


if __name__ == "__main__":
    main()
