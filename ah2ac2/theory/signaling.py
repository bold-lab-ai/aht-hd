"""Numerical helpers extracted from the paper experiment implementation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class CertificateError(RuntimeError):
    """Raised when a proof, numerical, review, or report gate fails."""


def effective_reference(p: float, eta: float) -> float:
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must lie in [0, 1]")
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    return (1.0 - 2.0 * eta) * p + eta


def bernoulli_kl(q: float, t: float) -> float:
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must lie in [0, 1]")
    if not (0.0 < t < 1.0):
        return math.inf
    terms = []
    if q > 0.0:
        terms.append(q * math.log(q / t))
    if q < 1.0:
        terms.append((1.0 - q) * math.log((1.0 - q) / (1.0 - t)))
    return float(sum(terms))


def reduced_objective(q: float, lam: float, t: float) -> float:
    if lam <= 0.0:
        raise ValueError("lambda must be positive")
    divergence = bernoulli_kl(q, t)
    if not math.isfinite(divergence):
        return -math.inf
    return t * t + (1.0 - t) ** 2 - 2.0 * lam * divergence


def reduced_derivative(q: float, lam: float, t: float) -> float:
    if not (0.0 < t < 1.0):
        raise ValueError("t must lie in (0, 1)")
    return (4.0 * t - 2.0) + 2.0 * lam * (q / t - (1.0 - q) / (1.0 - t))


def stationary_polynomial(q: float, lam: float, t: float) -> float:
    return 2.0 * t**3 - 3.0 * t**2 + (1.0 + lam) * t - lam * q


def symmetric_boundary_maximizers(lam: float) -> list[float]:
    if lam <= 0.0:
        raise ValueError("lambda must be positive")
    if lam >= 0.5:
        return [0.5]
    offset = math.sqrt(1.0 - 2.0 * lam)
    return [(1.0 - offset) / 2.0, (1.0 + offset) / 2.0]


def stationary_points(q: float, lam: float, imaginary_tolerance: float = 1.0e-7) -> list[float]:
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in (0, 1)")
    if lam <= 0.0:
        raise ValueError("lambda must be positive")
    if q == 0.5:
        candidates = symmetric_boundary_maximizers(lam) + [0.5]
    else:
        roots = np.roots([2.0, -3.0, 1.0 + lam, -lam * q])
        candidates = [
            float(root.real)
            for root in roots
            if abs(float(root.imag)) <= imaginary_tolerance and 0.0 < float(root.real) < 1.0
        ]
    unique: list[float] = []
    for value in sorted(candidates):
        if not unique or not math.isclose(value, unique[-1], rel_tol=0.0, abs_tol=1.0e-7):
            unique.append(value)
    return unique


def global_maximizers(q: float, lam: float) -> list[float]:
    if q == 0.5:
        return symmetric_boundary_maximizers(lam)
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in (0, 1)")
    if lam <= 0.0:
        raise ValueError("lambda must be positive")

    if q > 0.5:
        lower = q
        upper = float(np.nextafter(1.0, 0.0))
    else:
        lower = float(np.nextafter(0.0, 1.0))
        upper = q
    lower_sign = reduced_derivative(q, lam, lower)
    upper_sign = reduced_derivative(q, lam, upper)
    if not (lower_sign > 0.0 and upper_sign < 0.0):
        raise CertificateError(f"Could not bracket majority-side optimum for q={q}, lambda={lam}")

    for _ in range(256):
        midpoint = lower + 0.5 * (upper - lower)
        if midpoint == lower or midpoint == upper:
            break
        if reduced_derivative(q, lam, midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    optimum = upper if q > 0.5 else lower
    return [optimum]


def full_reward(x0: float, x1: float, y0: float, y1: float) -> float:
    return 0.5 * (x0 * y0 + (1.0 - x0) * (1.0 - y1) + x1 * y1 + (1.0 - x1) * (1.0 - y0))


def full_objective(q: float, lam: float, x0: float, x1: float, y0: float, y1: float) -> float:
    penalty = sum(bernoulli_kl(q, value) for value in (x0, x1, y0, y1))
    return full_reward(x0, x1, y0, y1) - 0.5 * lam * penalty


def _bit_probability(bit: int, probability_one: float) -> float:
    return probability_one if bit else 1.0 - probability_one


def enumerate_corruption_model(
    model: str,
    p: float,
    eta_sender: float,
    eta_receiver: float | None = None,
    *,
    state_probability_one: float = 0.5,
    receiver_message: str = "executed",
    shared_local_error: bool = False,
) -> dict[str, Any]:
    """Exactly enumerate a corruption model's binary trajectory distribution."""

    eta_receiver = eta_sender if eta_receiver is None else eta_receiver
    if model not in {"coherent", "local"}:
        raise ValueError(f"Unknown corruption model: {model}")
    if receiver_message not in {"executed", "intended"}:
        raise ValueError("receiver_message must be 'executed' or 'intended'")

    sender_num = {0: 0.0, 1: 0.0}
    sender_den = {0: 0.0, 1: 0.0}
    receiver_num = {0: 0.0, 1: 0.0}
    receiver_den = {0: 0.0, 1: 0.0}
    reward = 0.0
    branches = {
        "identity_identity": 0.0,
        "identity_flip": 0.0,
        "flip_identity": 0.0,
        "flip_flip": 0.0,
    }
    trajectory_distribution = {
        f"{state},{message},{action}": 0.0
        for state in (0, 1)
        for message in (0, 1)
        for action in (0, 1)
    }

    for convention in (0, 1):
        convention_prob = p if convention == 0 else 1.0 - p
        for state in (0, 1):
            state_prob = _bit_probability(state, state_probability_one)
            if model == "coherent":
                rows = []
                for error in (0, 1):
                    probability = (
                        convention_prob * state_prob * _bit_probability(error, eta_sender)
                    )
                    executed_convention = convention ^ error
                    message = state ^ executed_convention
                    action = message ^ executed_convention
                    rows.append((probability, message, action))
            else:
                rows = []
                for sender_error in (0, 1):
                    receiver_errors = (sender_error,) if shared_local_error else (0, 1)
                    for receiver_error in receiver_errors:
                        probability = (
                            convention_prob
                            * state_prob
                            * _bit_probability(sender_error, eta_sender)
                        )
                        if not shared_local_error:
                            probability *= _bit_probability(receiver_error, eta_receiver)
                        intended_message = state ^ convention
                        message = intended_message ^ sender_error
                        if receiver_message == "executed":
                            action = message ^ convention ^ receiver_error
                        else:
                            action = intended_message ^ convention ^ receiver_error
                        rows.append((probability, message, action))

            for probability, message, action in rows:
                sender_den[state] += probability
                receiver_den[message] += probability
                if message == state:
                    sender_num[state] += probability
                if action == message:
                    receiver_num[message] += probability
                if action == state:
                    reward += probability
                sender_label = "identity" if message == state else "flip"
                receiver_label = "identity" if action == message else "flip"
                branches[f"{sender_label}_{receiver_label}"] += probability
                trajectory_distribution[f"{state},{message},{action}"] += probability

    sender_conditionals = {str(state): sender_num[state] / sender_den[state] for state in (0, 1)}
    receiver_conditionals = {
        str(message): receiver_num[message] / receiver_den[message] for message in (0, 1)
    }
    return {
        "model": model,
        "p": p,
        "eta_sender": eta_sender,
        "eta_receiver": eta_receiver,
        "state_probability_one": state_probability_one,
        "receiver_message": receiver_message,
        "shared_local_error": shared_local_error,
        "sender_identity_conditionals": sender_conditionals,
        "receiver_identity_conditionals": receiver_conditionals,
        "reward": reward,
        "branch_masses": branches,
        "observable_trajectory_distribution": trajectory_distribution,
        "total_probability": sum(branches.values()),
    }
