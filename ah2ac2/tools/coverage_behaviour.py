"""coverage_behaviour.py — Stage 5 of the human-coverage analysis.

Build per-seat-game 24-dimensional behaviour-statistic vectors from games.safetensors
files (bot rollouts or human dataset).

Composition (24 dims):

* discard slot dist (M1) ............................. 5  (Laplace +0.5 smoothed)
* focus slot dist (M3) ............................... 5
* blind-play slot dist (M10) ......................... 5
* save rank dist (M8) ................................ 5
* color-clue rate (M5) ............................... 1
* clued-discard rate (M2) ............................ 1
* blind-play rate (M9) ............................... 1
* chop-clue rate (M6) ................................ 1

Vectors are computed per (game, seat) by filtering TurnStates to actions taken
by ``target_seat`` before invoking each extractor. The resulting matrix is
saved as ``{name}_per_seat_raw.npy`` (raw, pre-zscore) and
``{name}_per_seat.npy`` (z-scored using human-cloud statistics).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter

import numpy as np
from jaxmarl import make
from safetensors.numpy import load_file

from ah2ac2.tools.hgroup_metrics import (
    HAND_SIZE,
    NUM_RANKS,
    TurnState,
    compute_m1,
    compute_m2,
    compute_m3,
    compute_m5,
    compute_m6,
    compute_m8,
    compute_m9,
    compute_m10,
)
from ah2ac2.tools.hgroup_replay import replay_game

SLOT_BINS = list(range(1, HAND_SIZE + 1))  # 1..5
RANK_BINS = list(range(1, NUM_RANKS + 1))  # 1..5

LAPLACE_ALPHA = 0.5


def _normalise_counter(c: Counter, bins: list[int], alpha: float = LAPLACE_ALPHA) -> np.ndarray:
    """Laplace-smoothed normalised distribution over a fixed bin list."""
    counts = np.array([c.get(b, 0) for b in bins], dtype=np.float64)
    counts = counts + alpha
    return counts / counts.sum()


def per_seat_vector(turn_states: list[TurnState], seat: int) -> np.ndarray:
    """Return 24-d behaviour vector for one (game, seat)."""
    # Filter by actor seat
    ts_seat = [ts for ts in turn_states if ts.seat == seat]

    m1 = _normalise_counter(compute_m1(ts_seat), SLOT_BINS)
    m3 = _normalise_counter(compute_m3(ts_seat), SLOT_BINS)
    m10 = _normalise_counter(compute_m10(ts_seat), SLOT_BINS)
    m8 = _normalise_counter(compute_m8(ts_seat), RANK_BINS)
    m5 = compute_m5(ts_seat)
    m2 = compute_m2(ts_seat)
    m9 = compute_m9(ts_seat)
    m6 = compute_m6(ts_seat)
    return np.concatenate([m1, m3, m10, m8, [m5, m2, m9, m6]]).astype(np.float32)


def build_matrix(
    games_path: str, env, max_games: int | None = None, num_seats: int = 2
) -> np.ndarray:
    data = load_file(games_path)
    actions_all = np.asarray(data["actions"])
    decks_all = np.asarray(data["decks"])
    num_actions_all = np.asarray(data["num_actions"])
    N = actions_all.shape[0]
    if max_games is not None:
        N = min(N, max_games)

    rows: list[np.ndarray] = []
    t0 = time.time()
    for i in range(N):
        ts = replay_game(
            actions_all[i],
            decks_all[i],
            num_actions=int(num_actions_all[i]),
            env=env,
        )
        for seat in range(num_seats):
            rows.append(per_seat_vector(ts, seat))
        if (i + 1) % max(1, N // 10) == 0:
            print(f"    game {i + 1}/{N}  elapsed={time.time() - t0:.1f}s")
    return np.stack(rows, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--human_games", required=True, help="games.safetensors for the human dataset")
    p.add_argument(
        "--human_num_games",
        type=int,
        default=5000,
        help="cap human games to match the encoder run",
    )
    p.add_argument(
        "--human_seed",
        type=int,
        default=0,
        help="seed used to subsample human games (must match coverage_encode)",
    )
    p.add_argument(
        "--bot_rollouts_dir",
        required=True,
        help="directory containing one subfolder per bot with games.safetensors",
    )
    p.add_argument(
        "--bots",
        nargs="+",
        default=["BC_proxy_1", "BC_proxy_2", "Latent_proxy_1", "Latent_proxy_2", "Latent_proxy_3"],
    )
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    env = make("hanabi", num_agents=2)

    # Subsample humans deterministically (must match coverage_encode encode --seed)
    raw = load_file(args.human_games)
    n_total = raw["actions"].shape[0]
    if args.human_num_games is not None and args.human_num_games < n_total:
        rng_np = np.random.RandomState(args.human_seed)
        idx = rng_np.choice(n_total, size=args.human_num_games, replace=False)
        idx.sort()
        sub_actions = raw["actions"][idx]
        sub_decks = raw["decks"][idx]
        sub_num_actions = raw["num_actions"][idx]
        # write a tmp games.safetensors for human subsample
        from safetensors.numpy import save_file

        sub_path = os.path.join(args.out_dir, "human_subsample.safetensors")
        save_file(
            {
                "actions": sub_actions.astype(np.int32),
                "decks": sub_decks.astype(np.int32),
                "num_actions": sub_num_actions.astype(np.int32),
            },
            sub_path,
        )
        human_games_path = sub_path
        N_humans = args.human_num_games
    else:
        human_games_path = args.human_games
        N_humans = n_total

    print(f"[stage 5] building behaviour matrix for HUMAN ({N_humans} games)…")
    H = build_matrix(human_games_path, env)
    np.save(os.path.join(args.out_dir, "Human_per_seat_raw.npy"), H)
    print(f"  human raw: shape={H.shape}")

    bot_raws: dict[str, np.ndarray] = {}
    for bot in args.bots:
        gp = os.path.join(args.bot_rollouts_dir, bot, "games.safetensors")
        if not os.path.exists(gp):
            raise FileNotFoundError(gp)
        print(f"[stage 5] building behaviour matrix for {bot}…")
        M = build_matrix(gp, env)
        bot_raws[bot] = M
        np.save(os.path.join(args.out_dir, f"{bot}_per_seat_raw.npy"), M)
        print(f"  {bot} raw: shape={M.shape}")

    # Z-score against human stats
    h_mean = H.mean(axis=0)
    h_std = H.std(axis=0)
    h_std = np.where(h_std < 1e-6, 1.0, h_std)
    np.save(os.path.join(args.out_dir, "behaviour_zscore.npy"), np.stack([h_mean, h_std]))
    H_z = ((H - h_mean) / h_std).astype(np.float32)
    np.save(os.path.join(args.out_dir, "Human_per_seat.npy"), H_z)
    for bot, M in bot_raws.items():
        Mz = ((M - h_mean) / h_std).astype(np.float32)
        np.save(os.path.join(args.out_dir, f"{bot}_per_seat.npy"), Mz)

    # Also dump composition metadata
    composition = {
        "dims_total": int(H.shape[1]),
        "blocks": [
            {"name": "M1_discard_slot", "dims": "0:5"},
            {"name": "M3_focus_slot", "dims": "5:10"},
            {"name": "M10_blindplay_slot", "dims": "10:15"},
            {"name": "M8_save_rank", "dims": "15:20"},
            {"name": "M5_color_clue_rate", "dims": "20"},
            {"name": "M2_clued_discard_rate", "dims": "21"},
            {"name": "M9_blind_play_rate", "dims": "22"},
            {"name": "M6_chop_clue_rate", "dims": "23"},
        ],
        "laplace_alpha": LAPLACE_ALPHA,
        "human_n_seat_rows": int(H.shape[0]),
        "bots_n_seat_rows": {b: int(m.shape[0]) for b, m in bot_raws.items()},
    }
    with open(os.path.join(args.out_dir, "behaviour_composition.json"), "w") as f:
        json.dump(composition, f, indent=2)
    print(f"Saved behaviour matrices to {args.out_dir}")


if __name__ == "__main__":
    main()
