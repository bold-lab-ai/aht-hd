"""Human-grounding certificate for the latent proxy suite.

The pipeline has four explicit stages:

``prepare``
    Score development games, encode their seat-specific posteriors, fit
    development-only mixture weights and density references, and write a lock.
``audit``
    Verify the lock, score the untouched audit games, and run the predeclared
    likelihood and latent-support analyses without refitting.
``rollout``
    Generate common-deck SP/ordered-XP matrices for the raw latent references
    and their reconstructed proxies.
``report``
    Combine the audit and rollout artifacts into JSON/CSV/Markdown/LaTeX
    certificate tables, including the held-out 19-D behaviour diagnostics.

All paths, suites, seeds, and analysis settings live in a YAML manifest.  The
audit stage refuses to run if any frozen development input has changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from flax.core import freeze, unfreeze
from jaxmarl import make
from safetensors.numpy import load_file, save_file

from ah2ac2.nn.checkpoint_utils import unwrap_vae_params
from ah2ac2.tools.coverage_analyze import fit_ledoit_wolf
from ah2ac2.tools.coverage_behaviour import build_matrix
from ah2ac2.tools.coverage_sensitivity import fit_gmm, gmm_logpdf
from ah2ac2.tools.crossplay_eval import (
    ActorLstm,
    ActorLstmBCAdapter,
    ActorLstmPolicyOut,
    actor_uses_policy_out,
    load_actor_vars_for_bc,
    load_actor_vars_for_rl,
    load_actor_vars_for_vae,
)
from ah2ac2.tools.encode_human_latents import make_trajectory_batcher
from ah2ac2.tools.hgroup_rollout import (
    ACTION_DIM,
    DECK_SIZE,
    MAX_T,
    NUM_PLAYERS,
)
from ah2ac2.tools.rollout_encode_pca import (
    EncoderOnly,
    load_flax_safetensors,
)

DEFAULT_MANIFEST = "rebuttal-experiments/experiment-2/manifest.yaml"
LOCK_NAME = "development_lock.json"
PREPARED_NAME = "development_scores.npz"
AUDIT_NAME = "audit_scores.npz"
CALIBRATION_NAME = "calibration.npz"
WEIGHTS_NAME = "fitted_weights.json"

# 24-D coverage layout with M10_blindplay_slot[10:15] removed.
BEHAVIOUR_KEEP = np.r_[0:10, 15:24]
BEHAVIOUR_NAMES = (
    [f"discard slot {i}" for i in range(1, 6)]
    + [f"focus slot {i}" for i in range(1, 6)]
    + [f"save rank {i}" for i in range(1, 6)]
    + [
        "color-clue rate",
        "clued-discard rate",
        "blind-play rate",
        "chop-clue rate",
    ]
)


# ---------------------------------------------------------------------------
# Generic numerical helpers
# ---------------------------------------------------------------------------


def logsumexp(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.exp(x - np.expand_dims(logsumexp(x, axis=axis), axis=axis))


def fit_mixture_weights(
    log_likelihood: np.ndarray,
    *,
    max_iter: int = 10_000,
    tolerance: float = 1e-10,
    min_weight: float = 1e-12,
) -> np.ndarray:
    """Trajectory-mixture MLE by EM; the component is fixed per row."""
    ll = np.asarray(log_likelihood, dtype=np.float64)
    if ll.ndim != 2 or ll.shape[0] == 0 or ll.shape[1] == 0:
        raise ValueError(f"Expected non-empty [trajectory, component] array, got {ll.shape}")
    weights = np.full(ll.shape[1], 1.0 / ll.shape[1], dtype=np.float64)
    for _ in range(max_iter):
        resp = softmax(ll + np.log(np.maximum(weights, min_weight))[None, :], axis=1)
        updated = np.maximum(resp.mean(axis=0), min_weight)
        updated /= updated.sum()
        if np.max(np.abs(updated - weights)) < tolerance:
            weights = updated
            break
        weights = updated
    return weights


def mixture_nll(
    log_likelihood: np.ndarray,
    counts: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized per-trajectory NLL and posterior responsibilities."""
    ll = np.asarray(log_likelihood, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if ll.ndim != 2 or counts.shape != (ll.shape[0],):
        raise ValueError(f"Incompatible shapes: log_likelihood={ll.shape}, counts={counts.shape}")
    if weights.shape != (ll.shape[1],) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError(f"Invalid mixture weights with shape {weights.shape}")
    weights = weights / weights.sum()
    joint = ll + np.log(np.maximum(weights, 1e-300))[None, :]
    log_mix = logsumexp(joint, axis=1)
    nll = -log_mix / np.maximum(counts, 1.0)
    responsibilities = softmax(joint, axis=1)
    return nll, responsibilities


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 1_000
    for start in range(0, replicates, chunk):
        end = min(start + chunk, replicates)
        idx = rng.integers(0, values.size, size=(end - start, values.size))
        means[start:end] = values[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(x) for x in np.quantile(means, [alpha, 1.0 - alpha]))


def sign_flip_pvalue(
    paired_differences: np.ndarray,
    *,
    permutations: int,
    seed: int,
) -> float:
    """Two-sided paired randomization test of a zero mean difference."""
    delta = np.asarray(paired_differences, dtype=np.float64).reshape(-1)
    observed = abs(float(delta.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    chunk = 2_000
    for start in range(0, permutations, chunk):
        n = min(chunk, permutations - start)
        signs = rng.integers(0, 2, size=(n, delta.size), dtype=np.int8) * 2 - 1
        permuted = np.abs((signs * delta[None, :]).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed - 1e-15))
    return float((exceed + 1) / (permutations + 1))


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm family-wise adjusted p-values in the original order."""
    p = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(p[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int((base + int.from_bytes(digest[:4], "little")) % (2**32))


def effective_components(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = w[w > 0]
    return float(np.exp(-np.sum(w * np.log(w))))


def empirical_percentile(reference: np.ndarray, value: float) -> float:
    """Percent at or below value; high distance percentile means more outlying."""
    ref = np.asarray(reference, dtype=np.float64)
    return float(100.0 * np.mean(ref <= value))


def lower_density_percentile(reference_logp: np.ndarray, values_logp: np.ndarray) -> np.ndarray:
    """Percent of reference observations no denser than each supplied point."""
    reference = np.sort(np.asarray(reference_logp, dtype=np.float64))
    values = np.asarray(values_logp, dtype=np.float64)
    return 100.0 * np.searchsorted(reference, values, side="right") / reference.size


def mahalanobis_distances(x: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    chol = np.linalg.cholesky(np.asarray(covariance, dtype=np.float64))
    whitened = np.linalg.solve(chol, (x - mean).T)
    return np.sqrt(np.sum(whitened * whitened, axis=0))


def kth_neighbor_distances(
    queries: np.ndarray,
    reference: np.ndarray,
    *,
    k: int,
    exclude_self: bool = False,
    chunk_size: int = 256,
) -> np.ndarray:
    """Chunked Euclidean kNN distances without an sklearn dependency."""
    q = np.asarray(queries, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if k < 1 or k + int(exclude_self) > r.shape[0]:
        raise ValueError(f"k={k} is invalid for reference size {r.shape[0]}")
    output = np.empty(q.shape[0], dtype=np.float64)
    kth_index = k if exclude_self else k - 1
    for start in range(0, q.shape[0], chunk_size):
        end = min(start + chunk_size, q.shape[0])
        diff = q[start:end, None, :] - r[None, :, :]
        dist2 = np.einsum("qrd,qrd->qr", diff, diff)
        if exclude_self:
            if q.shape != r.shape or not np.shares_memory(q, r):
                # Array identity is not preserved through asarray; exact row
                # equality identifies self for the audit leave-one-out call.
                equal = np.all(q[start:end, None, :] == r[None, :, :], axis=2)
                dist2[equal] = np.inf
                kth_index = k - 1
            else:
                rows = np.arange(start, end)
                dist2[np.arange(end - start), rows] = np.inf
                kth_index = k - 1
        output[start:end] = np.sqrt(np.partition(dist2, kth_index, axis=1)[:, kth_index])
    return output


# ---------------------------------------------------------------------------
# Manifest, provenance, and artifact I/O
# ---------------------------------------------------------------------------


def load_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Manifest must be a mapping with version: 1")
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def repo_path(manifest: Mapping[str, Any], value: str) -> Path:
    root = Path(manifest.get("repo_root", ".")).resolve()
    return (root / value).resolve()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest_hash(manifest: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in manifest.items() if not k.startswith("_")}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def input_paths_for_lock(manifest: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {
        "development_dataset": repo_path(manifest, manifest["datasets"]["development"]),
        "vae_checkpoint": repo_path(manifest, manifest["vae"]["checkpoint"]),
    }
    for component in manifest["components"]:
        paths[f"component:{component['name']}"] = repo_path(manifest, component["checkpoint"])
        if component.get("z_path"):
            paths[f"latent:{component['name']}"] = repo_path(manifest, component["z_path"])
    return paths


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    component_names = [c["name"] for c in manifest["components"]]
    if len(component_names) != len(set(component_names)):
        raise ValueError("Component names must be unique")
    known = set(component_names)
    for suite, members in manifest["suites"].items():
        missing = set(members) - known
        if missing:
            raise ValueError(f"Suite {suite!r} references unknown components: {sorted(missing)}")
    for layer, members in manifest["rollout_layers"].items():
        missing = set(members) - known
        if missing:
            raise ValueError(
                f"Rollout layer {layer!r} references unknown components: {sorted(missing)}"
            )
    for label, path in {
        **input_paths_for_lock(manifest),
        "audit_dataset": repo_path(manifest, manifest["datasets"]["audit"]),
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if len(BEHAVIOUR_NAMES) != 19:
        raise AssertionError("The behavior signature must contain exactly 19 axes")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def overlap_counts(*dataset_paths: Path) -> dict[str, int]:
    ids = [set(load_file(str(path))["game_ids"].tolist()) for path in dataset_paths]
    labels = [path.name for path in dataset_paths]
    result: dict[str, int] = {}
    for i, j in itertools.combinations(range(len(ids)), 2):
        result[f"{labels[i]}::{labels[j]}"] = len(ids[i] & ids[j])
    return result


def development_overlap_with_vae_training(manifest: Mapping[str, Any]) -> int:
    """Reproduce the limited-data permutation used for the saved VAE."""
    vae = manifest["vae"]
    training = load_file(str(repo_path(manifest, manifest["datasets"]["training_reference"])))[
        "game_ids"
    ]
    development = set(
        load_file(str(repo_path(manifest, manifest["datasets"]["development"])))[
            "game_ids"
        ].tolist()
    )
    seed = int(vae["training_seed"])
    rng = jax.random.PRNGKey(seed)
    limited_key = jax.random.fold_in(rng, seed)
    _, permutation_key = jax.random.split(limited_key)
    indices = np.asarray(
        jax.random.permutation(permutation_key, len(training))[: int(vae["training_games"])]
    )
    return len(development & set(training[indices].tolist()))


def save_score_artifact(path: Path, artifact: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **artifact)


def load_score_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------
# Policies and demonstrated-trajectory scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedPolicy:
    name: str
    layer: str
    kind: str
    actor: Any
    variables: Any
    z: jnp.ndarray


def load_policy(component: Mapping[str, Any], architecture: Mapping[str, Any]) -> LoadedPolicy:
    kind = component["kind"]
    checkpoint = component["checkpoint_resolved"]
    common = {
        "preprocessing_features": list(architecture["preprocessing_features"]),
        "lstm_features": list(architecture["lstm_features"]),
        "postprocessing_features": list(architecture["postprocessing_features"]),
        "action_dim": int(architecture.get("action_dim", ACTION_DIM)),
        "dropout_rate": float(architecture.get("dropout", 0.0)),
        "activation_fn_name": architecture.get("act_fn", "gelu"),
    }
    if kind == "raw_bc":
        actor = ActorLstmBCAdapter(**common)
        variables = load_actor_vars_for_bc(checkpoint)
        z = jnp.zeros((0,), dtype=jnp.float32)
    elif kind == "raw_latent":
        actor = ActorLstm(**common)
        variables = load_actor_vars_for_vae(checkpoint)
        z = jnp.asarray(np.load(component["z_path_resolved"]), dtype=jnp.float32)
    elif kind in {"reconstructed_bc", "reconstructed_latent"}:
        variables = load_actor_vars_for_rl(checkpoint)
        actor_cls = ActorLstmPolicyOut if actor_uses_policy_out(variables["params"]) else ActorLstm
        actor = actor_cls(**common)
        if kind == "reconstructed_latent":
            z = jnp.asarray(np.load(component["z_path_resolved"]), dtype=jnp.float32)
        else:
            z = jnp.zeros((0,), dtype=jnp.float32)
    else:
        raise ValueError(f"Unknown component kind: {kind}")
    return LoadedPolicy(
        name=component["name"],
        layer=component["layer"],
        kind=kind,
        actor=actor,
        variables=variables,
        z=z,
    )


def resolve_components(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    for original in manifest["components"]:
        component = dict(original)
        component["checkpoint_resolved"] = str(repo_path(manifest, component["checkpoint"]))
        if component.get("z_path"):
            component["z_path_resolved"] = str(repo_path(manifest, component["z_path"]))
        resolved.append(component)
    return resolved


def initialize_policy_carry(policy: LoadedPolicy, batch_size: int):
    if isinstance(policy.actor, ActorLstmBCAdapter):
        return policy.actor.bind(policy.variables).net.initialize_carry(batch_size)
    return policy.actor.initialize_carry(batch_size)


def make_policy_scorer(policy: LoadedPolicy):
    """Create a compiled scorer for a replayed seat-major trajectory batch."""

    @jax.jit
    def score(
        observations_bt: jax.Array,
        actions_bt: jax.Array,
        legal_bt: jax.Array,
        active_bt: jax.Array,
    ) -> jax.Array:
        # Inputs are [seat-trajectory, time, feature]. Actor is time-major.
        batch_size = observations_bt.shape[0]
        obs_tb = jnp.swapaxes(observations_bt, 0, 1)
        resets_tb = jnp.zeros(obs_tb.shape[:2], dtype=jnp.bool_)
        z_arg = None
        if policy.z.size:
            z_arg = jnp.broadcast_to(policy.z[None, :], (batch_size, policy.z.shape[0]))
        carry = initialize_policy_carry(policy, batch_size)
        _, logits_tb, _ = policy.actor.apply(
            policy.variables,
            carry,
            (obs_tb, resets_tb),
            z=z_arg,
            training=False,
        )
        logits_bt = jnp.swapaxes(logits_tb, 0, 1)
        masked_logits = jnp.where(legal_bt > 0, logits_bt, -1e30)
        log_prob = jax.nn.log_softmax(masked_logits, axis=-1)
        chosen = jnp.take_along_axis(
            log_prob, actions_bt[..., None].astype(jnp.int32), axis=-1
        ).squeeze(-1)
        return (chosen * active_bt).sum(axis=1)

    return score


def build_encoder(manifest: Mapping[str, Any]):
    vae = manifest["vae"]
    encoder = EncoderOnly(
        preprocessing_features=tuple(vae["preprocessing_features"]),
        latent_dim=int(vae["latent_dim"]),
        act_fn=vae["act_fn"],
    )
    variables = load_flax_safetensors(str(repo_path(manifest, vae["checkpoint"])))
    params = unfreeze(variables["params"])
    params = unwrap_vae_params(params)
    return encoder, freeze({"params": params})


def make_encoder_apply(encoder, variables):
    @jax.jit
    def encode(observations_bt: jax.Array, time_mask_bt: jax.Array):
        return encoder.apply(
            variables,
            jnp.swapaxes(observations_bt, 0, 1),
            jnp.swapaxes(time_mask_bt, 0, 1),
        )

    return encode


def score_dataset(
    *,
    manifest: Mapping[str, Any],
    dataset_path: Path,
    batch_size: int,
    limit_games: int | None,
    seed: int,
) -> dict[str, np.ndarray]:
    data = load_file(str(dataset_path))
    n_total = int(data["actions"].shape[0])
    n = min(n_total, limit_games) if limit_games is not None else n_total
    if n < 1:
        raise ValueError("No games selected")

    components = resolve_components(manifest)
    policies = [load_policy(c, manifest["architecture"]) for c in components]
    scorers = [make_policy_scorer(policy) for policy in policies]
    encoder, encoder_variables = build_encoder(manifest)
    encode = make_encoder_apply(encoder, encoder_variables)

    actions_all = np.asarray(data["actions"][:n], dtype=np.int32)
    decks_all = np.asarray(data["decks"][:n], dtype=np.int32)
    num_actions_all = np.asarray(data["num_actions"][:n], dtype=np.int32)
    game_ids = np.asarray(data["game_ids"][:n])
    scores = np.asarray(data.get("scores", np.full(n, -1))[:n])
    num_seats = actions_all.shape[2]
    max_t = actions_all.shape[1]
    noop_id = int(manifest["architecture"].get("action_dim", ACTION_DIM)) - 1

    game_loglik = np.zeros((n, len(policies)), dtype=np.float64)
    seat_loglik = np.zeros((n, num_seats, len(policies)), dtype=np.float64)
    seat_counts = np.zeros((n, num_seats), dtype=np.int32)
    posterior_mu = np.zeros((n, num_seats, int(manifest["vae"]["latent_dim"])), dtype=np.float32)
    posterior_logvar = np.zeros_like(posterior_mu)

    get_trajectory = make_trajectory_batcher(num_seats)

    def replay_one(rng, deck, actions):
        trajectory, _ = get_trajectory(rng, deck, actions)
        return trajectory

    replay_many = jax.jit(jax.vmap(replay_one, in_axes=0))
    all_keys = jax.random.split(jax.random.PRNGKey(seed), n)

    started = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        actions = jnp.asarray(actions_all[start:end])
        decks = jnp.asarray(decks_all[start:end])
        counts = jnp.asarray(num_actions_all[start:end])
        trajectory = replay_many(all_keys[start:end], decks, actions)
        observations, replay_actions, legal = trajectory

        b = end - start
        observations_bt = jnp.swapaxes(observations, 1, 2).reshape(
            b * num_seats, max_t, observations.shape[-1]
        )
        replay_actions_bt = jnp.swapaxes(replay_actions, 1, 2).reshape(b * num_seats, max_t)
        legal_bt = jnp.swapaxes(legal, 1, 2).reshape(b * num_seats, max_t, legal.shape[-1])
        valid_gt = jnp.arange(max_t)[None, :] < counts[:, None]
        valid_gst = jnp.broadcast_to(valid_gt[:, None, :], (b, num_seats, max_t))
        actions_gst = jnp.swapaxes(actions, 1, 2)
        active_gst = valid_gst & (actions_gst != noop_id)
        active_bt = active_gst.reshape(b * num_seats, max_t).astype(jnp.float32)
        time_mask_bt = valid_gst.reshape(b * num_seats, max_t).astype(jnp.float32)

        active_counts = np.asarray(active_gst.sum(axis=2), dtype=np.int32)
        if not np.all(active_counts.sum(axis=1) == num_actions_all[start:end]):
            raise ValueError(
                f"Active action count mismatch in games [{start}:{end}]; "
                "expected one non-noop action per valid timestep"
            )
        seat_counts[start:end] = active_counts

        mu, logvar = encode(observations_bt, time_mask_bt)
        posterior_mu[start:end] = np.asarray(mu).reshape(b, num_seats, -1)
        posterior_logvar[start:end] = np.asarray(logvar).reshape(b, num_seats, -1)

        for component_index, scorer in enumerate(scorers):
            ll = np.asarray(
                scorer(observations_bt, replay_actions_bt, legal_bt, active_bt)
            ).reshape(b, num_seats)
            seat_loglik[start:end, :, component_index] = ll
            game_loglik[start:end, component_index] = ll.sum(axis=1)

        print(
            f"  scored {end:>4}/{n} games "
            f"({len(policies)} components, elapsed={time.time() - started:.1f}s)"
        )

    if not np.isfinite(game_loglik).all():
        raise FloatingPointError("Non-finite demonstrated-trajectory log likelihood")
    return {
        "component_names": np.asarray([p.name for p in policies]),
        "game_ids": game_ids,
        "human_scores": scores,
        "game_counts": num_actions_all,
        "seat_counts": seat_counts,
        "game_loglik": game_loglik,
        "seat_loglik": seat_loglik,
        "posterior_mu": posterior_mu,
        "posterior_logvar": posterior_logvar,
    }


# ---------------------------------------------------------------------------
# Development fitting and held-out likelihood analysis
# ---------------------------------------------------------------------------


def component_index(artifact: Mapping[str, np.ndarray]) -> dict[str, int]:
    return {str(name): i for i, name in enumerate(artifact["component_names"].tolist())}


def suite_columns(
    manifest: Mapping[str, Any],
    artifact: Mapping[str, np.ndarray],
    suite_name: str,
) -> list[int]:
    index = component_index(artifact)
    return [index[name] for name in manifest["suites"][suite_name]]


def fit_all_suite_weights(
    manifest: Mapping[str, Any],
    artifact: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {"game": {}, "seat": {}}
    seat_ll = artifact["seat_loglik"].reshape(-1, artifact["seat_loglik"].shape[-1])
    for suite_name in manifest["suites"]:
        cols = suite_columns(manifest, artifact, suite_name)
        result["game"][suite_name] = fit_mixture_weights(artifact["game_loglik"][:, cols]).tolist()
        result["seat"][suite_name] = fit_mixture_weights(seat_ll[:, cols]).tolist()
    return result


def behavior_matrix_19(dataset_path: Path, limit_games: int | None) -> np.ndarray:
    env = make("hanabi", num_agents=NUM_PLAYERS)
    matrix_24 = build_matrix(str(dataset_path), env, max_games=limit_games, num_seats=NUM_PLAYERS)
    return np.asarray(matrix_24[:, BEHAVIOUR_KEEP], dtype=np.float64)


def fit_calibration(
    manifest: Mapping[str, Any],
    artifact: Mapping[str, np.ndarray],
    behavior_raw: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    latent = artifact["posterior_mu"].reshape(-1, artifact["posterior_mu"].shape[-1])
    latent_mean, latent_cov = fit_ledoit_wolf(latent)

    behavior_mean = behavior_raw.mean(axis=0)
    behavior_std = behavior_raw.std(axis=0)
    behavior_std = np.where(behavior_std < 1e-6, 1.0, behavior_std)
    behavior_z = (behavior_raw - behavior_mean) / behavior_std
    behavior_lw_mean, behavior_lw_cov = fit_ledoit_wolf(behavior_z)

    max_k = int(manifest["analysis"]["behavior_gmm_max_components"])
    fits = [fit_gmm(behavior_z, k, seed=int(manifest["seed"]) + k) for k in range(1, max_k + 1)]
    selected = min(fits, key=lambda fit: fit["bic"])
    selected_k = len(selected["pis"])
    metadata = {
        "behavior_gmm_candidates": [
            {"components": len(fit["pis"]), "bic": float(fit["bic"]), "ll": float(fit["ll"])}
            for fit in fits
        ],
        "behavior_gmm_selected_components": selected_k,
    }
    arrays = {
        "latent_mean": np.asarray(latent_mean),
        "latent_cov": np.asarray(latent_cov),
        "behavior_mean": behavior_mean,
        "behavior_std": behavior_std,
        "behavior_lw_mean": np.asarray(behavior_lw_mean),
        "behavior_lw_cov": np.asarray(behavior_lw_cov),
        "behavior_gmm_weights": np.asarray(selected["pis"]),
        "behavior_gmm_means": np.asarray(selected["mus"]),
        "behavior_gmm_covariances": np.asarray(selected["covs"]),
    }
    return arrays, metadata


def summarize_suite(
    *,
    log_likelihood: np.ndarray,
    counts: np.ndarray,
    members: Sequence[str],
    weights: np.ndarray,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    nll, responsibilities = mixture_nll(log_likelihood, counts, weights)
    best = np.argmax(log_likelihood, axis=1)
    shares = np.bincount(best, minlength=len(members)) / len(best)
    share_cis: dict[str, list[float]] = {}
    for i, member in enumerate(members):
        lo, hi = bootstrap_mean_ci(
            (best == i).astype(np.float64),
            replicates=bootstrap_replicates,
            seed=stable_seed(seed, f"share:{member}"),
        )
        share_cis[member] = [lo, hi]
    mean = float(nll.mean())
    ci = bootstrap_mean_ci(nll, replicates=bootstrap_replicates, seed=seed)
    log_mix = -nll * counts
    return (
        {
            "members": list(members),
            "weights": np.asarray(weights).tolist(),
            "effective_components": effective_components(weights),
            "macro_nll": mean,
            "macro_nll_ci95": list(ci),
            "action_weighted_nll": float(-log_mix.sum() / counts.sum()),
            "best_component_fraction": {
                member: float(shares[i]) for i, member in enumerate(members)
            },
            "best_component_fraction_ci95": share_cis,
            "mean_responsibility": {
                member: float(responsibilities[:, i].mean()) for i, member in enumerate(members)
            },
        },
        nll,
    )


def likelihood_analysis(
    manifest: Mapping[str, Any],
    audit: Mapping[str, np.ndarray],
    fitted_weights: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    analysis = manifest["analysis"]
    bootstrap_replicates = int(analysis["bootstrap_replicates"])
    base_seed = int(manifest["seed"])
    index = component_index(audit)
    seat_ll_all = audit["seat_loglik"].reshape(-1, audit["seat_loglik"].shape[-1])
    seat_counts = audit["seat_counts"].reshape(-1)
    results: dict[str, Any] = {"game": {}, "seat_sensitivity": {}}
    per_game: dict[str, np.ndarray] = {}

    for suite_name, members in manifest["suites"].items():
        cols = [index[name] for name in members]
        for weighting in ("equal", "fitted"):
            if weighting == "equal":
                game_weights = np.full(len(cols), 1.0 / len(cols))
                seat_weights = game_weights
            else:
                game_weights = np.asarray(fitted_weights["game"][suite_name])
                seat_weights = np.asarray(fitted_weights["seat"][suite_name])
            label = f"{suite_name}:{weighting}"
            result, nll = summarize_suite(
                log_likelihood=audit["game_loglik"][:, cols],
                counts=audit["game_counts"],
                members=members,
                weights=game_weights,
                bootstrap_replicates=bootstrap_replicates,
                seed=stable_seed(base_seed, f"game:{label}"),
            )
            results["game"][label] = result
            per_game[label] = nll
            seat_result, _ = summarize_suite(
                log_likelihood=seat_ll_all[:, cols],
                counts=seat_counts,
                members=members,
                weights=seat_weights,
                bootstrap_replicates=bootstrap_replicates,
                seed=stable_seed(base_seed, f"seat:{label}"),
            )
            results["seat_sensitivity"][label] = seat_result

    contrast_results: list[dict[str, Any]] = []
    raw_pvalues: list[float] = []
    for contrast in manifest["contrasts"]:
        lhs = f"{contrast['lhs']}:equal"
        rhs = f"{contrast['rhs']}:equal"
        delta = per_game[lhs] - per_game[rhs]
        seed = stable_seed(base_seed, contrast["name"])
        ci = bootstrap_mean_ci(delta, replicates=bootstrap_replicates, seed=seed)
        pvalue = sign_flip_pvalue(
            delta,
            permutations=int(analysis["sign_flip_permutations"]),
            seed=seed,
        )
        raw_pvalues.append(pvalue)
        contrast_results.append(
            {
                "name": contrast["name"],
                "lhs": lhs,
                "rhs": rhs,
                "mean_delta_nll": float(delta.mean()),
                "delta_ci95": list(ci),
                "pvalue_two_sided": pvalue,
                "negative_favors_lhs": True,
            }
        )
    for row, adjusted in zip(contrast_results, holm_adjust(raw_pvalues)):
        row["pvalue_holm"] = adjusted
    results["predeclared_contrasts"] = contrast_results
    return results, per_game


def latent_support_analysis(
    manifest: Mapping[str, Any],
    audit: Mapping[str, np.ndarray],
    calibration: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    audit_mu = audit["posterior_mu"].reshape(-1, audit["posterior_mu"].shape[-1])
    mean = calibration["latent_mean"]
    covariance = calibration["latent_cov"]
    audit_mahal = mahalanobis_distances(audit_mu, mean, covariance)
    k_values = [int(x) for x in manifest["analysis"]["knn_k_values"]]
    human_knn = {
        k: kth_neighbor_distances(audit_mu, audit_mu, k=k, exclude_self=True) for k in k_values
    }

    index = component_index(audit)
    results: dict[str, Any] = {}
    chol = np.linalg.cholesky(covariance)
    for latent in manifest["certificate_latents"]:
        z = np.asarray(np.load(repo_path(manifest, latent["z_path"])), dtype=np.float64)
        z_mahal = float(mahalanobis_distances(z[None, :], mean, covariance)[0])
        z_knn = {k: float(kth_neighbor_distances(z[None, :], audit_mu, k=k)[0]) for k in k_values}
        whitened = np.linalg.solve(chol, (audit_mu - z).T)
        distances = np.sqrt(np.sum(whitened * whitened, axis=0))
        nearest_indices = np.argsort(distances)[: int(manifest["analysis"]["nearest_humans"])]

        proxy_col = index[latent["proxy_component"]]
        proxy_seat_ll = audit["seat_loglik"][:, :, proxy_col].reshape(-1)
        proxy_seat_counts = audit["seat_counts"].reshape(-1)
        recon_members = manifest["suites"]["recon_latent"]
        recon_cols = [index[name] for name in recon_members]
        recon_seat_ll = audit["seat_loglik"][:, :, recon_cols].reshape(-1, len(recon_cols))
        recon_resp = softmax(recon_seat_ll, axis=1)
        proxy_position = recon_members.index(latent["proxy_component"])

        nearest: list[dict[str, Any]] = []
        for flat_index in nearest_indices:
            game_index = int(flat_index // NUM_PLAYERS)
            seat = int(flat_index % NUM_PLAYERS)
            nearest.append(
                {
                    "game_index": game_index,
                    "game_id": int(audit["game_ids"][game_index]),
                    "seat": seat,
                    "whitened_distance": float(distances[flat_index]),
                    "component_seat_nll": float(
                        -proxy_seat_ll[flat_index] / max(proxy_seat_counts[flat_index], 1)
                    ),
                    "equal_weight_responsibility_within_latent_suite": float(
                        recon_resp[flat_index, proxy_position]
                    ),
                }
            )

        results[latent["id"]] = {
            "raw_component": latent["raw_component"],
            "proxy_component": latent["proxy_component"],
            "z_norm": float(np.linalg.norm(z)),
            "mahalanobis_distance": z_mahal,
            "mahalanobis_percentile_under_audit_humans": empirical_percentile(
                audit_mahal, z_mahal
            ),
            "knn": {
                str(k): {
                    "distance": z_knn[k],
                    "percentile_under_audit_leave_one_out": empirical_percentile(
                        human_knn[k], z_knn[k]
                    ),
                }
                for k in k_values
            },
            "nearest_audit_human_seats": nearest,
        }
    return results


def behavior_density_summary(
    points_raw: np.ndarray,
    calibration: Mapping[str, np.ndarray],
    audit_reference_logp: np.ndarray,
    audit_reference_mahal: np.ndarray,
) -> dict[str, Any]:
    z = (points_raw - calibration["behavior_mean"]) / calibration["behavior_std"]
    fit = {
        "pis": calibration["behavior_gmm_weights"],
        "mus": calibration["behavior_gmm_means"],
        "covs": calibration["behavior_gmm_covariances"],
    }
    logp = gmm_logpdf(z, fit)
    density_pct = lower_density_percentile(audit_reference_logp, logp)
    mahal = mahalanobis_distances(
        z, calibration["behavior_lw_mean"], calibration["behavior_lw_cov"]
    )
    mahal_pct = np.asarray([empirical_percentile(audit_reference_mahal, value) for value in mahal])
    return {
        "rows": int(z.shape[0]),
        "median_density_percentile": float(np.median(density_pct)),
        "fraction_below_density_percentile_1": float(np.mean(density_pct < 1.0)),
        "fraction_below_density_percentile_5": float(np.mean(density_pct < 5.0)),
        "median_mahalanobis_distance": float(np.median(mahal)),
        "median_mahalanobis_percentile": float(np.median(mahal_pct)),
        "fraction_above_mahalanobis_percentile_99": float(np.mean(mahal_pct > 99.0)),
        "mean_signature_z": z.mean(axis=0).tolist(),
    }


# ---------------------------------------------------------------------------
# Common-deck rollouts
# ---------------------------------------------------------------------------


def sample_deck_pairs(key: jax.Array) -> jax.Array:
    multiplicities = [3, 2, 2, 2, 1]
    cards = [
        (color, rank)
        for color in range(5)
        for rank, copies in enumerate(multiplicities)
        for _ in range(copies)
    ]
    deck = jnp.asarray(cards, dtype=jnp.int32)
    return deck[jax.random.permutation(key, deck.shape[0])]


def policy_forward(policy: LoadedPolicy, hidden, obs_b, legal_b):
    num_players = obs_b.shape[0]
    z_arg = None
    if policy.z.size:
        z_arg = jnp.broadcast_to(policy.z[None, :], (num_players, policy.z.shape[0]))
    obs_tb = obs_b[None, :, :]
    resets = jnp.zeros((1, num_players), dtype=jnp.bool_)
    hidden, logits, _ = policy.actor.apply(
        policy.variables,
        hidden,
        (obs_tb, resets),
        z=z_arg,
        training=False,
    )
    masked = jnp.where(legal_b > 0, logits[0], -1e30)
    return hidden, jnp.argmax(masked, axis=-1).astype(jnp.int32)


def build_pair_rollout(env, agents: tuple[str, ...], seat0: LoadedPolicy, seat1: LoadedPolicy):
    noop = ACTION_DIM - 1

    def rollout_one(rng):
        rng, deck_key = jax.random.split(rng)
        deck = sample_deck_pairs(deck_key)
        obs, state = env.reset_from_deck_of_pairs(deck)
        legal = env.get_legal_moves(state)
        hidden0 = initialize_policy_carry(seat0, NUM_PLAYERS)
        hidden1 = initialize_policy_carry(seat1, NUM_PLAYERS)

        def step(carry, _):
            terminated = carry[6]

            def pad(c):
                action = jnp.full((NUM_PLAYERS,), noop, dtype=jnp.int32)
                return c, (action, jnp.bool_(False))

            def play(c):
                rng, state, obs, legal, hidden0, hidden1, terminated, score = c
                obs_b = jnp.stack([obs[agent] for agent in agents])
                legal_b = jnp.stack([legal[agent] for agent in agents])
                hidden0, proposed0 = policy_forward(seat0, hidden0, obs_b, legal_b)
                hidden1, proposed1 = policy_forward(seat1, hidden1, obs_b, legal_b)
                current = jnp.argmax(state.cur_player_idx).astype(jnp.int32)
                selected = jax.lax.select(current == 0, proposed0[0], proposed1[1])
                action = jnp.full((NUM_PLAYERS,), noop, dtype=jnp.int32)
                action = action.at[current].set(selected)
                env_action = {agents[i]: action[i] for i in range(NUM_PLAYERS)}
                rng, step_key = jax.random.split(rng)
                obs2, state2, reward, dones, _ = env.step_env(step_key, state, env_action)
                legal2 = env.get_legal_moves(state2)
                done = dones["__all__"].astype(jnp.bool_)
                score2 = score + jnp.asarray(reward["__all__"], dtype=jnp.float32)
                updated = (
                    rng,
                    state2,
                    obs2,
                    legal2,
                    hidden0,
                    hidden1,
                    terminated | done,
                    score2,
                )
                return updated, (action, jnp.bool_(True))

            return jax.lax.cond(terminated, pad, play, carry)

        initial = (
            rng,
            state,
            obs,
            legal,
            hidden0,
            hidden1,
            jnp.bool_(False),
            jnp.float32(0.0),
        )
        final, (actions, valid) = jax.lax.scan(step, initial, xs=None, length=MAX_T)
        score = final[-1]
        return (
            actions,
            deck,
            valid.sum().astype(jnp.int32),
            jnp.round(score).astype(jnp.int32),
        )

    return jax.jit(jax.vmap(rollout_one))


def run_rollout_pair(
    runner,
    rng_keys: jax.Array,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    n = rng_keys.shape[0]
    output = {
        "actions": np.full((n, MAX_T, NUM_PLAYERS), ACTION_DIM - 1, dtype=np.int32),
        "decks": np.zeros((n, DECK_SIZE, 2), dtype=np.int32),
        "num_actions": np.zeros(n, dtype=np.int32),
        "scores": np.zeros(n, dtype=np.int32),
    }
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        actions, decks, counts, scores = runner(rng_keys[start:end])
        output["actions"][start:end] = np.asarray(actions)
        output["decks"][start:end] = np.asarray(decks)
        output["num_actions"][start:end] = np.asarray(counts)
        output["scores"][start:end] = np.asarray(scores)
    return output


def cmd_rollout(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    out_dir = Path(args.output_dir or manifest["output_dir"]).resolve()
    components = resolve_components(manifest)
    loaded = {
        component["name"]: load_policy(component, manifest["architecture"])
        for component in components
    }
    episodes = args.episodes or int(manifest["rollout"]["episodes"])
    batch_size = args.batch_size or int(manifest["rollout"]["batch_size"])
    seed = int(manifest["rollout"]["seed"])
    rng_keys = jnp.stack([jax.random.PRNGKey(seed + i) for i in range(episodes)])
    env = make("hanabi", num_agents=NUM_PLAYERS)
    agents = tuple(env.agents)

    rollout_root = out_dir / "rollouts"
    rollout_root.mkdir(parents=True, exist_ok=True)
    for layer, names in manifest["rollout_layers"].items():
        print(f"[rollout] layer={layer}, components={names}, episodes={episodes}")
        scores = np.zeros((len(names), len(names), episodes), dtype=np.int16)
        for i, name0 in enumerate(names):
            for j, name1 in enumerate(names):
                started = time.time()
                print(f"  {name0} -> {name1}")
                runner = build_pair_rollout(env, agents, loaded[name0], loaded[name1])
                result = run_rollout_pair(runner, rng_keys, batch_size=batch_size)
                scores[i, j] = result["scores"]
                if i == j:
                    component_dir = rollout_root / layer / name0
                    component_dir.mkdir(parents=True, exist_ok=True)
                    save_file(result, str(component_dir / "games.safetensors"))
                print(
                    f"    score={result['scores'].mean():.3f} "
                    f"sem={result['scores'].std() / math.sqrt(episodes):.3f} "
                    f"elapsed={time.time() - started:.1f}s"
                )
        np.savez_compressed(
            rollout_root / f"{layer}_crossplay.npz",
            component_names=np.asarray(names),
            scores=scores,
            episode_seeds=np.arange(seed, seed + episodes, dtype=np.int64),
        )
    write_json(
        rollout_root / "rollout_provenance.json",
        {
            "manifest_sha256": canonical_manifest_hash(manifest),
            "episodes": episodes,
            "batch_size": batch_size,
            "seed": seed,
            "common_decks_across_all_pairs": True,
            "ordered_crossplay": True,
        },
    )


# ---------------------------------------------------------------------------
# Pipeline commands
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    dev = repo_path(manifest, manifest["datasets"]["development"])
    audit = repo_path(manifest, manifest["datasets"]["audit"])
    training = repo_path(manifest, manifest["datasets"]["training_reference"])
    overlaps = overlap_counts(dev, audit, training)
    print(json.dumps({"manifest": args.manifest, "overlap_counts": overlaps}, indent=2))


def cmd_prepare(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    out_dir = Path(args.output_dir or manifest["output_dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_path = repo_path(manifest, manifest["datasets"]["development"])
    limit = args.limit_games

    print(f"[prepare] scoring development data: {dev_path}")
    artifact = score_dataset(
        manifest=manifest,
        dataset_path=dev_path,
        batch_size=args.batch_size,
        limit_games=limit,
        seed=int(manifest["seed"]),
    )
    save_score_artifact(out_dir / PREPARED_NAME, artifact)

    print("[prepare] extracting development 19-D behavior vectors")
    behavior_raw = behavior_matrix_19(dev_path, limit)
    np.save(out_dir / "development_behavior_19d_raw.npy", behavior_raw)

    print("[prepare] fitting trajectory weights and human-reference densities")
    fitted_weights = fit_all_suite_weights(manifest, artifact)
    write_json(out_dir / WEIGHTS_NAME, fitted_weights)
    calibration, calibration_meta = fit_calibration(manifest, artifact, behavior_raw)
    np.savez_compressed(out_dir / CALIBRATION_NAME, **calibration)
    write_json(out_dir / "calibration_metadata.json", calibration_meta)

    inputs = input_paths_for_lock(manifest)
    lock = {
        "manifest_sha256": canonical_manifest_hash(manifest),
        "git_revision": git_revision(),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input_sha256": {label: sha256_file(path) for label, path in inputs.items()},
        "development_games": int(artifact["game_ids"].shape[0]),
        "development_limit": limit,
        "canonical": limit is None,
        "component_names": artifact["component_names"].tolist(),
        "suite_names": list(manifest["suites"]),
        "audit_dataset_path": str(repo_path(manifest, manifest["datasets"]["audit"])),
        "split_overlap_counts": overlap_counts(
            dev_path,
            repo_path(manifest, manifest["datasets"]["audit"]),
            repo_path(manifest, manifest["datasets"]["training_reference"]),
        ),
        "development_games_used_to_train_vae": development_overlap_with_vae_training(manifest),
        "development_is_not_claimed_as_held_out": True,
    }
    write_json(out_dir / LOCK_NAME, lock)
    print(f"[prepare] wrote frozen development artifacts to {out_dir}")


def verify_development_lock(
    manifest: Mapping[str, Any],
    out_dir: Path,
    *,
    allow_noncanonical: bool,
) -> dict[str, Any]:
    lock_path = out_dir / LOCK_NAME
    if not lock_path.exists():
        raise FileNotFoundError(f"Run prepare first; missing {lock_path}")
    lock = load_json(lock_path)
    if lock["manifest_sha256"] != canonical_manifest_hash(manifest):
        raise RuntimeError("Manifest changed after development fitting; rerun prepare")
    if lock["script_sha256"] != sha256_file(Path(__file__).resolve()):
        raise RuntimeError("Pipeline code changed after development fitting; rerun prepare")
    if not lock["canonical"] and not allow_noncanonical:
        raise RuntimeError(
            "Development lock was made with --limit-games. "
            "Use --allow-noncanonical only for a smoke test."
        )
    for label, path in input_paths_for_lock(manifest).items():
        actual = sha256_file(path)
        if actual != lock["input_sha256"][label]:
            raise RuntimeError(f"Frozen input changed after prepare: {label}")
    return lock


def cmd_audit(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    out_dir = Path(args.output_dir or manifest["output_dir"]).resolve()
    lock = verify_development_lock(manifest, out_dir, allow_noncanonical=args.allow_noncanonical)
    audit_path = repo_path(manifest, manifest["datasets"]["audit"])
    limit = args.limit_games
    if lock["canonical"] and limit is not None and not args.allow_noncanonical:
        raise RuntimeError("A canonical development lock cannot produce a limited audit")

    print(f"[audit] scoring audit data: {audit_path}")
    audit = score_dataset(
        manifest=manifest,
        dataset_path=audit_path,
        batch_size=args.batch_size,
        limit_games=limit,
        seed=int(manifest["seed"]) + 1,
    )
    if audit["component_names"].tolist() != lock["component_names"]:
        raise RuntimeError("Audit component order differs from frozen development order")
    save_score_artifact(out_dir / AUDIT_NAME, audit)

    print("[audit] extracting audit 19-D behavior vectors")
    audit_behavior = behavior_matrix_19(audit_path, limit)
    np.save(out_dir / "audit_behavior_19d_raw.npy", audit_behavior)

    fitted_weights = load_json(out_dir / WEIGHTS_NAME)
    with np.load(out_dir / CALIBRATION_NAME, allow_pickle=False) as raw:
        calibration = {key: raw[key] for key in raw.files}

    print("[audit] running predeclared likelihood and latent-support analyses")
    likelihood, per_game = likelihood_analysis(manifest, audit, fitted_weights)
    latent = latent_support_analysis(manifest, audit, calibration)
    audit_behavior_summary = behavior_density_summary(
        audit_behavior,
        calibration,
        gmm_logpdf(
            (audit_behavior - calibration["behavior_mean"]) / calibration["behavior_std"],
            {
                "pis": calibration["behavior_gmm_weights"],
                "mus": calibration["behavior_gmm_means"],
                "covs": calibration["behavior_gmm_covariances"],
            },
        ),
        mahalanobis_distances(
            (audit_behavior - calibration["behavior_mean"]) / calibration["behavior_std"],
            calibration["behavior_lw_mean"],
            calibration["behavior_lw_cov"],
        ),
    )
    np.savez_compressed(out_dir / "audit_suite_nll_per_game.npz", **per_game)
    audit_results = {
        "provenance": {
            "manifest_sha256": canonical_manifest_hash(manifest),
            "development_lock_sha256": sha256_file(out_dir / LOCK_NAME),
            "audit_dataset_sha256": sha256_file(audit_path),
            "audit_games": int(audit["game_ids"].shape[0]),
            "canonical": lock["canonical"] and limit is None,
            "audit_is_disjoint_from_development": (
                len(
                    set(load_file(str(audit_path))["game_ids"].tolist())
                    & set(
                        load_file(str(repo_path(manifest, manifest["datasets"]["development"])))[
                            "game_ids"
                        ].tolist()
                    )
                )
                == 0
            ),
        },
        "likelihood": likelihood,
        "latent_support": latent,
        "audit_human_behavior_reference": audit_behavior_summary,
    }
    write_json(out_dir / "audit_results.json", audit_results)
    print(f"[audit] wrote held-out results to {out_dir}")


def rollout_matrix_summary(
    path: Path,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        names = [str(x) for x in data["component_names"].tolist()]
        scores = np.asarray(data["scores"], dtype=np.float64)
    means = scores.mean(axis=2)
    symmetric = (means + means.T) / 2.0
    rows: dict[str, Any] = {}
    for i, name in enumerate(names):
        offdiag = np.delete(symmetric[i], i)
        sp_ci = bootstrap_mean_ci(
            scores[i, i],
            replicates=bootstrap_replicates,
            seed=stable_seed(seed, f"sp:{name}"),
        )
        xp_episode = np.delete((scores[i] + scores[:, i]) / 2.0, i, axis=0).mean(axis=0)
        xp_ci = bootstrap_mean_ci(
            xp_episode,
            replicates=bootstrap_replicates,
            seed=stable_seed(seed, f"xp:{name}"),
        )
        rows[name] = {
            "self_play_mean": float(means[i, i]),
            "self_play_ci95": list(sp_ci),
            "mean_symmetrized_crossplay": float(offdiag.mean()),
            "mean_symmetrized_crossplay_ci95": list(xp_ci),
            "sp_minus_xp": float(means[i, i] - offdiag.mean()),
        }
    return {
        "component_names": names,
        "ordered_mean_matrix": means.tolist(),
        "symmetrized_mean_matrix": symmetric.tolist(),
        "components": rows,
    }


def component_behavior_and_signatures(
    manifest: Mapping[str, Any],
    out_dir: Path,
    audit: Mapping[str, np.ndarray],
    calibration: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_behavior = np.load(out_dir / "audit_behavior_19d_raw.npy")
    audit_z = (audit_behavior - calibration["behavior_mean"]) / calibration["behavior_std"]
    fit = {
        "pis": calibration["behavior_gmm_weights"],
        "mus": calibration["behavior_gmm_means"],
        "covs": calibration["behavior_gmm_covariances"],
    }
    audit_logp = gmm_logpdf(audit_z, fit)
    audit_mahal = mahalanobis_distances(
        audit_z, calibration["behavior_lw_mean"], calibration["behavior_lw_cov"]
    )
    index = component_index(audit)
    recon_members = manifest["suites"]["recon_latent"]
    recon_cols = [index[name] for name in recon_members]
    assigned_game = np.argmax(audit["game_loglik"][:, recon_cols], axis=1)

    behavior_results: dict[str, Any] = {}
    signature_results: dict[str, Any] = {}
    for layer, names in manifest["rollout_layers"].items():
        for name in names:
            games_path = out_dir / "rollouts" / layer / name / "games.safetensors"
            if not games_path.exists():
                raise FileNotFoundError(f"Missing rollout artifact: {games_path}")
            raw = behavior_matrix_19(games_path, int(manifest["rollout"]["behavior_episodes"]))
            np.save(
                out_dir / "rollouts" / layer / name / "behavior_19d_raw.npy",
                raw,
            )
            summary = behavior_density_summary(raw, calibration, audit_logp, audit_mahal)
            behavior_results[name] = summary

            proxy_signature = np.asarray(summary["mean_signature_z"])
            top = np.argsort(np.abs(proxy_signature))[::-1][:3]
            signature: dict[str, Any] = {
                "proxy_top_axes": [
                    {
                        "axis": BEHAVIOUR_NAMES[i],
                        "proxy_z": float(proxy_signature[i]),
                    }
                    for i in top
                ]
            }
            if name in recon_members:
                member_index = recon_members.index(name)
                selected_games = assigned_game == member_index
                selected_rows = np.repeat(selected_games, NUM_PLAYERS)
                human_signature = (
                    audit_z[selected_rows].mean(axis=0)
                    if np.any(selected_rows)
                    else np.full(19, np.nan)
                )
                for row, axis_index in zip(signature["proxy_top_axes"], top):
                    row["assigned_human_z"] = float(human_signature[axis_index])
                    row["direction_agrees"] = bool(
                        np.sign(proxy_signature[axis_index])
                        == np.sign(human_signature[axis_index])
                    )
                signature.update(
                    {
                        "assigned_audit_games": int(selected_games.sum()),
                        "signature_correlation": (
                            float(np.corrcoef(proxy_signature, human_signature)[0, 1])
                            if np.isfinite(human_signature).all()
                            else None
                        ),
                    }
                )
            signature_results[name] = signature
    return behavior_results, signature_results


def render_markdown(
    audit_results: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> str:
    lines = [
        "# Human-Grounding Certificate",
        "",
        (
            "Primary likelihood results use equal trajectory-level mixture weights. "
            "The component is fixed jointly for both seats for the full game."
        ),
        "",
        "## Held-Out Human Likelihood",
        "",
        "| Suite | Equal-weight NLL | 95% CI | Fitted-weight NLL |",
        "| --- | ---: | ---: | ---: |",
    ]
    game = audit_results["likelihood"]["game"]
    suite_order = certificate["suite_order"]
    for suite in suite_order:
        equal = game[f"{suite}:equal"]
        fitted = game[f"{suite}:fitted"]
        ci = equal["macro_nll_ci95"]
        lines.append(
            f"| {suite} | {equal['macro_nll']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {fitted['macro_nll']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Prespecified Paired Tests",
            "",
            "| Contrast | Delta NLL | 95% CI | Holm p |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in audit_results["likelihood"]["predeclared_contrasts"]:
        ci = row["delta_ci95"]
        lines.append(
            f"| {row['name']} | {row['mean_delta_nll']:.4f} | "
            f"[{ci[0]:.4f}, {ci[1]:.4f}] | {row['pvalue_holm']:.4g} |"
        )
    lines.extend(
        [
            "",
            "Negative delta NLL favors the left-hand suite.",
            "",
            "## Latent Proxy Certificates",
            "",
            (
                "| Proxy | Mahalanobis pct | 10-NN pct | Best-game share | "
                "Behavior density pct | SP | Mean XP |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in certificate["latent_proxy_rows"]:
        lines.append(
            f"| {row['proxy']} | {row['latent_mahalanobis_percentile']:.1f} | "
            f"{row['latent_knn10_percentile']:.1f} | "
            f"{row['best_game_fraction']:.3f} | "
            f"{row['behavior_median_density_percentile']:.1f} | "
            f"{row['self_play']:.2f} | {row['mean_crossplay']:.2f} |"
        )
    lines.extend(
        [
            "",
            (
                "The development file is calibration data, not held-out data: 991 of "
                "its 1,000 games were included in VAE training. All headline NLL and "
                "support percentiles above are computed on the disjoint audit file."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_latex(certificate: Mapping[str, Any]) -> str:
    rows = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Proxy & Latent Mah.\ pct. & 10-NN pct. & Best frac. & Behav.\ pct. & SP & XP \\",
        r"\midrule",
    ]
    for row in certificate["latent_proxy_rows"]:
        label = row["proxy"].replace("_", r"\_")
        rows.append(
            f"{label} & {row['latent_mahalanobis_percentile']:.1f} & "
            f"{row['latent_knn10_percentile']:.1f} & "
            f"{row['best_game_fraction']:.3f} & "
            f"{row['behavior_median_density_percentile']:.1f} & "
            f"{row['self_play']:.2f} & {row['mean_crossplay']:.2f} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(rows)


def cmd_report(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    out_dir = Path(args.output_dir or manifest["output_dir"]).resolve()
    audit_results = load_json(out_dir / "audit_results.json")
    audit = load_score_artifact(out_dir / AUDIT_NAME)
    with np.load(out_dir / CALIBRATION_NAME, allow_pickle=False) as raw:
        calibration = {key: raw[key] for key in raw.files}

    behavior, signatures = component_behavior_and_signatures(manifest, out_dir, audit, calibration)
    competence: dict[str, Any] = {}
    for layer in manifest["rollout_layers"]:
        competence[layer] = rollout_matrix_summary(
            out_dir / "rollouts" / f"{layer}_crossplay.npz",
            bootstrap_replicates=int(manifest["analysis"]["bootstrap_replicates"]),
            seed=stable_seed(int(manifest["seed"]), f"rollout:{layer}"),
        )

    game_likelihood = audit_results["likelihood"]["game"]["recon_latent:equal"]
    best_fraction = game_likelihood["best_component_fraction"]
    latent_proxy_rows: list[dict[str, Any]] = []
    for latent in manifest["certificate_latents"]:
        proxy = latent["proxy_component"]
        support = audit_results["latent_support"][latent["id"]]
        comp = competence["reconstructed"]["components"][proxy]
        latent_proxy_rows.append(
            {
                "latent_id": latent["id"],
                "proxy": proxy,
                "raw_reference": latent["raw_component"],
                "latent_mahalanobis_percentile": support[
                    "mahalanobis_percentile_under_audit_humans"
                ],
                "latent_knn10_percentile": support["knn"]["10"][
                    "percentile_under_audit_leave_one_out"
                ],
                "component_audit_nll": float(
                    -audit["game_loglik"][:, component_index(audit)[proxy]].sum()
                    / audit["game_counts"].sum()
                ),
                "best_game_fraction": best_fraction[proxy],
                "behavior_median_density_percentile": behavior[proxy]["median_density_percentile"],
                "behavior_fraction_below_5": behavior[proxy][
                    "fraction_below_density_percentile_5"
                ],
                "self_play": comp["self_play_mean"],
                "mean_crossplay": comp["mean_symmetrized_crossplay"],
                "sp_minus_xp": comp["sp_minus_xp"],
            }
        )

    certificate = {
        "provenance": audit_results["provenance"],
        "suite_order": list(manifest["suites"]),
        "latent_proxy_rows": latent_proxy_rows,
        "behavior_support": behavior,
        "behavior_signatures": signatures,
        "competence": competence,
        "nearest_human_seats": {
            latent["id"]: audit_results["latent_support"][latent["id"]][
                "nearest_audit_human_seats"
            ]
            for latent in manifest["certificate_latents"]
        },
    }
    write_json(out_dir / "certificate.json", certificate)
    with (out_dir / "certificate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(latent_proxy_rows[0]))
        writer.writeheader()
        writer.writerows(latent_proxy_rows)
    (out_dir / "RESULTS.md").write_text(
        render_markdown(audit_results, certificate), encoding="utf-8"
    )
    (out_dir / "certificate_table.tex").write_text(render_latex(certificate), encoding="utf-8")
    print(f"[report] wrote certificate artifacts to {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override manifest output_dir (useful for smoke tests)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate paths, suites, and split overlap")
    validate.set_defaults(func=cmd_validate)

    prepare = sub.add_parser("prepare", help="fit development-only weights and references")
    prepare.add_argument("--batch-size", type=int, default=32)
    prepare.add_argument("--limit-games", type=int, default=None)
    prepare.set_defaults(func=cmd_prepare)

    audit = sub.add_parser("audit", help="score and analyze the frozen audit set")
    audit.add_argument("--batch-size", type=int, default=32)
    audit.add_argument("--limit-games", type=int, default=None)
    audit.add_argument("--allow-noncanonical", action="store_true")
    audit.set_defaults(func=cmd_audit)

    rollout = sub.add_parser("rollout", help="run common-deck SP and ordered XP")
    rollout.add_argument("--episodes", type=int, default=None)
    rollout.add_argument("--batch-size", type=int, default=None)
    rollout.set_defaults(func=cmd_rollout)

    report = sub.add_parser("report", help="assemble final certificate and tables")
    report.set_defaults(func=cmd_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
