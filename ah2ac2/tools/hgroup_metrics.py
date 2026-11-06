"""H-Group convention audit — metric extractors (M1–M12).

Pre-registered. The 12 ``compute_mN`` docstrings quote the operational
definitions from ``hgroup_audit_spec.md`` (§3 / §4). Phase 3 fills the
bodies; the docstrings remain unchanged from Phase 1's commit.

The audit is 2-player Hanabi only. Slot indexing in this module is the
**spec convention** (1 = leftmost = newest, 5 = rightmost = oldest).
``hgroup_replay.replay_game`` converts from jaxmarl's reversed convention
before producing ``TurnState`` sequences.

Pure-Python module: no JAX / numpy at import time. Extractors consume
plain ``list[TurnState]`` and return ``Counter`` or ``float``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Locked constants (spec §4)
# ---------------------------------------------------------------------------

HAND_SIZE: int = 5
NUM_PLAYERS: int = 2
NUM_COLORS: int = 5
NUM_RANKS: int = 5
MAX_RANK: int = 5

# Per-color copy counts in a Hanabi deck (rank -> copies per color).
DECK_COMPOSITION: dict[int, int] = {1: 3, 2: 2, 3: 2, 4: 2, 5: 1}

ActionType = Literal["play", "discard", "clue_color", "clue_rank", "noop"]


# ---------------------------------------------------------------------------
# TurnState — the contract between replay and extractors
# ---------------------------------------------------------------------------


@dataclass
class TurnState:
    """One turn's worth of game state and metadata.

    Convention: ``hand_clue_history`` and ``hands`` reflect state **before**
    the action takes effect; ``post_board`` reflects state **after**.
    Slot indexing in all spec-facing fields is 1..5 (spec convention; index
    0 in the inner lists corresponds to spec slot 1 = newest).

    Common ``action_detail`` keys (populated by ``replay_game``):

    * ``action_type``: copy of the top-level field.
    * ``slot``: spec slot 1..5 — for play/discard.
    * ``color``: int 0..4, true color — for play/discard (true card identity).
    * ``rank``: int 1..5, true rank — for play/discard.
    * ``target_seat``: int — for clue actions.
    * ``color`` xor ``rank``: clue specifier.
    * ``slots_touched``: list[int] (spec) — slots in receiver's hand the
      clue actually touched (true color/rank match).
    * ``newly_touched`` / ``re_touched``: spec slots that were unclued or
      already clued before this clue.
    * ``focus_slot``: spec slot of focus card per H-Group rule, or None.
    * ``pre_chop_slot``: receiver's structural chop (rightmost-unclued)
      before this clue, or None if hand was locked.
    * ``latency``: for play actions, turns elapsed since the most recent
      clue received by the actor. None if the actor has never been clued.
    """

    seat: int
    action_type: ActionType
    action_detail: dict[str, Any] = field(default_factory=dict)
    pre_obs: dict[str, Any] | None = None
    post_board: dict[str, Any] = field(default_factory=dict)
    hand_clue_history: list[list[set[tuple[str, int]] | None]] = field(default_factory=list)
    hands: list[list[tuple[int, int] | None]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_critical(
    card: tuple[int, int],
    fireworks: dict[int, int],
    discards: list[tuple[int, int]],
) -> bool:
    """Spec §4 'Critical card definition'.

    Card c is currently critical iff c is not playable trash AND the number
    of remaining copies (= total copies in the deck - copies already
    discarded or known-played) equals 1.
    """
    color, rank = card
    # Trash: this rank for this color is already played (fireworks[color] >= rank).
    if fireworks.get(color, 0) >= rank:
        return False
    total = DECK_COMPOSITION.get(rank, 0)
    discarded_count = sum(1 for d in discards if d == card)
    # Played count: 0 (since fireworks[color] < rank means this rank not yet played).
    remaining = total - discarded_count
    return remaining == 1


def _bucket_latency(latency: int) -> int:
    """Bucket latency into {0, 1, 2, 3, 4} where 4 saturates '4+'."""
    if latency < 0:
        return 0
    if latency >= 4:
        return 4
    return latency


# ---------------------------------------------------------------------------
# Metric extractors (M1–M12)
# ---------------------------------------------------------------------------


def compute_m1(turn_states: list[TurnState]) -> Counter:
    """M1 — Discard slot distribution (Spec §3 Axis 1 / §4).

    "For every discard event in a rollout, record the slot index
    (1 = leftmost = newest, increasing rightward) of the discarded card.
    Aggregate over all discards into a normalised histogram. Categorical,
    hand_size bins."

    Returns a ``Counter`` over slot indices in {1, …, HAND_SIZE}. Caller
    normalises if a probability mass is wanted.
    """
    out: Counter = Counter()
    for ts in turn_states:
        if ts.action_type != "discard":
            continue
        slot = ts.action_detail.get("slot")
        if slot is None:
            continue
        out[int(slot)] += 1
    return out


def compute_m2(turn_states: list[TurnState]) -> float:
    """M2 — Clued-vs-unclued discard rate (Spec §3 Axis 1 / §4).

    "Fraction of discards where the discarded card had any positive clue
    information (i.e. was at any prior point touched by a clue). Scalar in
    [0, 1]."

    Returns a float in [0, 1]; conventionally 0.0 when no discards were made.
    """
    total = 0
    clued = 0
    for ts in turn_states:
        if ts.action_type != "discard":
            continue
        slot = ts.action_detail.get("slot")
        if slot is None:
            continue
        total += 1
        # pre-action hand_clue_history at this slot for the discarder
        history = ts.hand_clue_history[ts.seat][slot - 1]
        if history:
            clued += 1
    return (clued / total) if total > 0 else 0.0


def compute_m3(turn_states: list[TurnState]) -> Counter:
    """M3 — Clue focus slot distribution (Spec §3 Axis 2 / §4).

    "For every clue given, identify the focus card per [the H-Group focus
    rule: if no cards are new, focus is on the leftmost re-touched card; if
    one card is new, focus is on it; if multiple are new and one was on chop,
    focus is on chop; otherwise focus is on the leftmost newly-touched card]
    and record its slot. Aggregate over all clues into a normalised
    histogram. Categorical, hand_size bins."

    Returns a ``Counter`` over slot indices in {1, …, HAND_SIZE}.
    """
    out: Counter = Counter()
    for ts in turn_states:
        if ts.action_type not in ("clue_color", "clue_rank"):
            continue
        focus = ts.action_detail.get("focus_slot")
        if focus is None:
            continue
        out[int(focus)] += 1
    return out


def compute_m4(turn_states: list[TurnState]) -> Counter:
    """M4 — Cards-touched-per-clue distribution (Spec §3 Axis 2 / §4).

    "For every clue given, count the number of cards in the receiver's hand
    newly touched (not previously touched by any prior clue). Aggregate over
    all clues. Categorical, bins {1, 2, 3, 4, 5}."

    Returns a ``Counter`` over counts in {1, …, 5}. Clues that touch zero
    new cards are excluded from the histogram (the bin range is {1..5} per
    spec; "0 newly touched" is a degenerate edge case that violates the
    Minimum Clue Value Principle and is reported separately if needed).
    """
    out: Counter = Counter()
    for ts in turn_states:
        if ts.action_type not in ("clue_color", "clue_rank"):
            continue
        newly = ts.action_detail.get("newly_touched", [])
        n = len(newly)
        if n == 0:
            continue
        if n > 5:
            n = 5  # safety; jaxmarl can't produce > 5 in 5-card hands
        out[n] += 1
    return out


def compute_m5(turn_states: list[TurnState]) -> float:
    """M5 — Color-clue rate (Spec §3 Axis 3 / §4).

    "Fraction of clues that are color clues (vs. number clues). Scalar in
    [0, 1]."

    Returns a float in [0, 1]; conventionally 0.0 when no clues were given.
    """
    total = 0
    color = 0
    for ts in turn_states:
        if ts.action_type == "clue_color":
            total += 1
            color += 1
        elif ts.action_type == "clue_rank":
            total += 1
    return (color / total) if total > 0 else 0.0


def compute_m6(turn_states: list[TurnState]) -> float:
    """M6 — Chop-clue rate (Spec §3 Axis 4 / §4).

    "Fraction of all clues whose focus is on the receiver's current
    rightmost-unclued slot (the structural chop). Scalar in [0, 1]. Proxy
    for save vs. play balance."

    Edge case (spec §4 'Chop-clue rate edge case'): if the receiver has *no*
    unclued cards at the moment of clue-giving (locked hand), no slot counts
    as chop and the clue is excluded from the denominator. Locked-hand
    frequency is reported separately as a sanity-check side metric.
    """
    denom = 0
    num = 0
    for ts in turn_states:
        if ts.action_type not in ("clue_color", "clue_rank"):
            continue
        chop = ts.action_detail.get("pre_chop_slot")
        if chop is None:
            continue  # locked hand: excluded
        denom += 1
        if ts.action_detail.get("focus_slot") == chop:
            num += 1
    return (num / denom) if denom > 0 else 0.0


def compute_m7(turn_states: list[TurnState]) -> float:
    """M7 — Critical-card-saved rate (Spec §3 Axis 4 / §4).

    "Among chop-targeted clues, fraction targeting cards that are *currently
    critical* [non-trash AND exactly 1 copy remaining]. Scalar in [0, 1].
    Distinguishes targeted saves from liberal saves."

    Conditions on critical at the moment the chop-clue is given. Returns 0.0
    when no chop-clues were observed (denominator empty).
    """
    denom = 0
    num = 0
    for ts in turn_states:
        if ts.action_type not in ("clue_color", "clue_rank"):
            continue
        chop = ts.action_detail.get("pre_chop_slot")
        focus = ts.action_detail.get("focus_slot")
        if chop is None or focus != chop:
            continue
        target = ts.action_detail.get("target_seat")
        if target is None:
            continue
        card = ts.hands[target][chop - 1]
        if card is None:
            continue
        denom += 1
        # Pre-clue board: the *previous* turn's post_board. If this is the
        # first turn or there is no prior turn for context, use empty board.
        # Construct pre-clue board by looking at the turn before this one.
        # Simpler: use ts.post_board minus the effect of this clue. A clue
        # action does NOT change fireworks/discards, so ts.post_board is the
        # state at the moment of the clue.
        fireworks = ts.post_board.get("fireworks", {})
        discards = ts.post_board.get("discards", [])
        if _is_critical(card, fireworks, discards):
            num += 1
    return (num / denom) if denom > 0 else 0.0


def compute_m8(turn_states: list[TurnState]) -> Counter:
    """M8 — Save rank distribution (Spec §3 Axis 4 / §4).

    "Among chop-targeted clues, normalised histogram over the saved card's
    true rank: bins {1, 2, 3, 4, 5}."

    Returns a ``Counter`` over true ranks in {1, …, 5}.
    """
    out: Counter = Counter()
    for ts in turn_states:
        if ts.action_type not in ("clue_color", "clue_rank"):
            continue
        chop = ts.action_detail.get("pre_chop_slot")
        focus = ts.action_detail.get("focus_slot")
        if chop is None or focus != chop:
            continue
        target = ts.action_detail.get("target_seat")
        if target is None:
            continue
        card = ts.hands[target][chop - 1]
        if card is None:
            continue
        out[int(card[1])] += 1
    return out


def compute_m9(turn_states: list[TurnState]) -> float:
    """M9 — Blind-play rate (Spec §3 Axis 5 / §4).

    "Fraction of plays where the played card had no prior positive clue
    information. Scalar in [0, 1]."

    A play of card c is blind iff no clue prior to the turn of play touched
    card c specifically (touched ≡ clue's color or rank matched c's true
    color or rank). Negative inferences do NOT count as touching c.
    """
    total = 0
    blind = 0
    for ts in turn_states:
        if ts.action_type != "play":
            continue
        slot = ts.action_detail.get("slot")
        if slot is None:
            continue
        total += 1
        history = ts.hand_clue_history[ts.seat][slot - 1]
        if not history:
            blind += 1
    return (blind / total) if total > 0 else 0.0


def compute_m10(turn_states: list[TurnState]) -> Counter:
    """M10 — Blind-play slot distribution (Spec §3 Axis 5 / §4).

    "For blind plays only, normalised histogram of the slot the played card
    was in at the moment of play. Categorical, hand_size bins."

    Returns a ``Counter`` over slot indices in {1, …, HAND_SIZE}, keyed on
    blind plays only.
    """
    out: Counter = Counter()
    for ts in turn_states:
        if ts.action_type != "play":
            continue
        slot = ts.action_detail.get("slot")
        if slot is None:
            continue
        history = ts.hand_clue_history[ts.seat][slot - 1]
        if history:
            continue  # not blind
        out[int(slot)] += 1
    return out


def compute_m11(turn_states: list[TurnState]) -> Counter:
    """M11 — Blind-play latency (Spec §3 Axis 5 / §4).

    "For each blind play, count turns elapsed between the most recent clue
    given to *the player making the blind play* and the blind-play turn
    itself. If the most recent clue was given on the immediately preceding
    turn for that player, latency = 0. Aggregate to a distribution over
    {0, 1, 2, 3, 4+}."

    Bins ``{0, 1, 2, 3, 4}`` where ``4`` is the saturating "4+" bucket.
    Blind plays where the actor has never received a clue are bucketed
    into 4 (the maximum bin).
    """
    out: Counter = Counter()
    for ts in turn_states:
        if ts.action_type != "play":
            continue
        slot = ts.action_detail.get("slot")
        if slot is None:
            continue
        history = ts.hand_clue_history[ts.seat][slot - 1]
        if history:
            continue  # not blind
        latency = ts.action_detail.get("latency")
        if latency is None:
            out[4] += 1
        else:
            out[_bucket_latency(int(latency))] += 1
    return out


def compute_m12(turn_states: list[TurnState]) -> float:
    """M12 — Bomb rate (Spec §3 Axis 6 / §4).

    "Number of strikes (misplays) per completed game. Scalar, non-negative."

    For a single-game ``turn_states``, this is the strike count read from the
    final ``post_board['strikes']``. The audit reports the per-bot mean
    across K=500 games (spec §5.1).
    """
    if not turn_states:
        return 0.0
    final = turn_states[-1].post_board
    return float(final.get("strikes", 0))


__all__ = [
    "HAND_SIZE",
    "NUM_PLAYERS",
    "NUM_COLORS",
    "NUM_RANKS",
    "MAX_RANK",
    "DECK_COMPOSITION",
    "ActionType",
    "TurnState",
    "compute_m1",
    "compute_m2",
    "compute_m3",
    "compute_m4",
    "compute_m5",
    "compute_m6",
    "compute_m7",
    "compute_m8",
    "compute_m9",
    "compute_m10",
    "compute_m11",
    "compute_m12",
]
