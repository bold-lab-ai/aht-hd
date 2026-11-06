"""Paired whole-game easiness/compatibility analysis; optional synthetic demo."""

import argparse
import json
from pathlib import Path

import numpy as np

from ah2ac2.tools.experiment3_stats import bootstrap_ci, compatibility_effects, sign_flip_pvalue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input",
        type=Path,
        help="NPZ: bc_on_bc, bc_on_latent, latent_on_bc, latent_on_latent; each [game]",
    )
    source.add_argument(
        "--demo",
        action="store_true",
        help="Generate synthetic paired returns; these are not paper results",
    )
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("outputs/score-analysis.json"))
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    keys = ["bc_on_bc", "bc_on_latent", "latent_on_bc", "latent_on_latent"]
    if args.demo:
        rng = np.random.default_rng(args.seed)
        shared = rng.normal(size=128)
        values = dict(
            zip(
                keys,
                [
                    19 + shared + rng.normal(0, 0.3, 128),
                    16 + shared + rng.normal(0, 0.3, 128),
                    18 + shared + rng.normal(0, 0.3, 128),
                    17 + shared + rng.normal(0, 0.3, 128),
                ],
            )
        )
    else:
        with np.load(args.input, allow_pickle=False) as data:
            values = {k: np.asarray(data[k], dtype=float) for k in keys}
    shapes = {a.shape for a in values.values()}
    if len(shapes) != 1 or any(
        a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() for a in values.values()
    ):
        parser.error(
            "All four arrays must be finite, equally sized vectors with at least two paired games"
        )
    effects = compatibility_effects(**values)
    result = {
        "synthetic_demo": args.demo,
        "games": len(values[keys[0]]),
        "uncertainty": "whole-game resampling conditional on fixed checkpoints",
        "effects": {},
    }
    for key in ["easiness", "interaction"]:
        x = effects[key]
        result["effects"][key] = {
            "mean": float(x.mean()),
            "ci95": bootstrap_ci(x, replicates=args.replicates, seed=args.seed),
            "two_sided_sign_flip_p": sign_flip_pvalue(
                x, permutations=args.replicates, seed=args.seed
            ),
        }
    result["multiplicity"] = (
        "Raw p-values for one method pair; apply holm_adjust across planned method pairs when making multiple claims."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
