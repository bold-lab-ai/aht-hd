"""coverage_drop_dims.py — slice columns out of existing behaviour matrices.

Reads ``*_per_seat_raw.npy`` matrices from a source directory, drops the
specified columns, re-z-scores against the human cloud's surviving dims, and
writes ``*_per_seat[_raw].npy`` + ``behaviour_zscore.npy`` to a target dir.

Used to test how robust the Stage 5 coverage ranking is to dropping
event-sparse blocks (M4 / M8 / M10) that are dominated by Laplace smoothing.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

CLOUDS = [
    "Human",
    "BC_proxy_1",
    "BC_proxy_2",
    "Latent_proxy_1",
    "Latent_proxy_2",
    "Latent_proxy_3",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", default="outputs/coverage/behaviour_space")
    p.add_argument("--dst_dir", required=True)
    p.add_argument(
        "--drop_blocks",
        nargs="+",
        default=["M10"],
        choices=["M1", "M3", "M10", "M8", "M5", "M2", "M9", "M6"],
        help="behaviour-vector blocks to drop (M1/M3/M10/M8 are 5d slot/rank dists; M5/M2/M9/M6 are scalars)",
    )
    args = p.parse_args()
    os.makedirs(args.dst_dir, exist_ok=True)

    # Slice ranges (24-d layout from coverage_behaviour.py)
    block_slices = {
        "M1": slice(0, 5),
        "M3": slice(5, 10),
        "M10": slice(10, 15),
        "M8": slice(15, 20),
        "M5": slice(20, 21),
        "M2": slice(21, 22),
        "M9": slice(22, 23),
        "M6": slice(23, 24),
    }
    block_names = {
        "M1": "discard slot dist",
        "M3": "focus slot dist",
        "M10": "blind-play slot dist",
        "M8": "save rank dist",
        "M5": "color clue rate",
        "M2": "clued discard rate",
        "M9": "blind play rate",
        "M6": "chop clue rate",
    }
    # Build a keep mask
    D = 24
    keep = np.ones(D, dtype=bool)
    for blk in args.drop_blocks:
        keep[block_slices[blk]] = False
    keep_idx = np.where(keep)[0]
    print(f"Dropping blocks: {args.drop_blocks}  -> keeping {keep_idx.shape[0]}/{D} dims")

    # Load human raw, recompute zscore stats on surviving dims
    human_raw = np.load(os.path.join(args.src_dir, "Human_per_seat_raw.npy"))
    H_kept = human_raw[:, keep_idx]
    h_mean = H_kept.mean(axis=0)
    h_std = H_kept.std(axis=0)
    h_std = np.where(h_std < 1e-6, 1.0, h_std)
    np.save(os.path.join(args.dst_dir, "behaviour_zscore.npy"), np.stack([h_mean, h_std]))

    for cloud in CLOUDS:
        raw = np.load(os.path.join(args.src_dir, f"{cloud}_per_seat_raw.npy"))
        kept = raw[:, keep_idx]
        np.save(os.path.join(args.dst_dir, f"{cloud}_per_seat_raw.npy"), kept)
        z = ((kept - h_mean) / h_std).astype(np.float32)
        np.save(os.path.join(args.dst_dir, f"{cloud}_per_seat.npy"), z)
        print(f"  {cloud}: {raw.shape} -> {kept.shape}")

    composition = {
        "dims_total": int(keep_idx.shape[0]),
        "kept_blocks": [b for b in block_slices if keep[block_slices[b]].all()],
        "dropped_blocks": list(args.drop_blocks),
        "dropped_block_meanings": [block_names[b] for b in args.drop_blocks],
        "source_dir": args.src_dir,
    }
    with open(os.path.join(args.dst_dir, "behaviour_composition.json"), "w") as f:
        json.dump(composition, f, indent=2)
    print(f"Wrote {args.dst_dir}")


if __name__ == "__main__":
    main()
