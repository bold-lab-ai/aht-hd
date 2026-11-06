"""Replay rollouts through jaxmarl into ``TurnState`` sequences (Phase 3).

The Phase 2 rollouts save only ``(actions, deck)`` per game; this module
walks each game through ``jaxmarl`` and produces the per-turn snapshot that
the Phase 3 metric extractors consume.

Conventions (see also ``hgroup_metrics.TurnState`` and ``hgroup_audit_spec.md``):

* **Slot indexing.** Spec uses 1=leftmost=newest, 5=rightmost=oldest.
  jaxmarl uses 0=oldest, 4=newest (verified empirically — newly-drawn cards
  land at jaxmarl slot 4, plays/discards shift higher slots down).
  Conversion: ``spec_slot = 5 - jaxmarl_slot``  (and back the same way).
* **Action encoding (jaxmarl, 21 actions, 2p).**
    ``0–4``: discard jaxmarl slot 0..4
    ``5–9``: play jaxmarl slot 0..4
    ``10–14``: color clue (color = action - 10) to next seat
    ``15–19``: rank clue (rank = action - 15 + 1, 1-indexed) to next seat
    ``20``: noop
* **Color encoding.** 0=red, 1=yellow, 2=green, 3=white, 4=blue
  (matching jaxmarl's ``HR/HY/HG/HW/HB`` clue ordering).
* **Rank encoding.** Internal jaxmarl: 0..4. Spec / TurnState: 1..5.
  All ``rank`` values exposed to extractors are **1-indexed**.
* **Empty slots.** When the deck runs out, played/discarded slots are not
  refilled; jaxmarl zeros out the hand entry. The replay treats those as
  ``None`` in ``TurnState.hands``.

Public entry: ``replay_game(actions, deck, num_actions) -> list[TurnState]``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxmarl import make

from ah2ac2.tools.hgroup_metrics import (
    HAND_SIZE,
    NUM_PLAYERS,
    TurnState,
)

ACTION_DIM = 21
NOOP_ID = ACTION_DIM - 1
NUM_COLORS = 5
NUM_RANKS = 5


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def jaxmarl_slot_to_spec(j: int) -> int:
    """jaxmarl slot 0..4  ->  spec slot 1..5 (5 = oldest = jaxmarl 0)."""
    return HAND_SIZE - j  # 5 - j


def spec_slot_to_jaxmarl(s: int) -> int:
    """Inverse."""
    return HAND_SIZE - s  # 5 - s


# ---------------------------------------------------------------------------
# State decoders (one-hot -> Python primitives)
# ---------------------------------------------------------------------------


def _decode_hand(player_hands_seat: np.ndarray) -> list[tuple[int, int] | None]:
    """player_hands[seat] is (5 slots, 5 colors, 5 ranks) one-hot.

    Returns a list indexed by spec slot index (0-based: index 0 = spec slot 1
    = newest = jaxmarl slot 4). Each entry is ``(color, rank_1_indexed)`` or
    ``None`` for an empty slot.
    """
    out: list[tuple[int, int] | None] = [None] * HAND_SIZE
    for j in range(HAND_SIZE):
        one_hot = player_hands_seat[j]  # (5, 5)
        if one_hot.sum() <= 0:
            continue
        idxs = np.argwhere(one_hot)
        if len(idxs) == 0:
            continue
        c, r0 = int(idxs[0, 0]), int(idxs[0, 1])
        spec_idx = (
            HAND_SIZE - 1 - j
        )  # j=4 (newest) -> 0 (spec slot 1); j=0 (oldest) -> 4 (spec slot 5)
        out[spec_idx] = (c, r0 + 1)  # 1-indexed rank
    return out


def _decode_clue_history(
    colors_revealed_seat: np.ndarray,
    ranks_revealed_seat: np.ndarray,
) -> list[set[tuple[str, int]] | None]:
    """colors/ranks_revealed[seat] are each (5 slots, 5 colors|ranks).

    Returns a list indexed by spec slot index. Each entry is the set of
    clue specifiers ``{("color", c), ("rank", r_1_indexed)}`` ever applied
    to the card currently in that slot (jaxmarl resets these when a card
    leaves the slot).
    """
    out: list[set[tuple[str, int]]] = [set() for _ in range(HAND_SIZE)]
    for j in range(HAND_SIZE):
        spec_idx = HAND_SIZE - 1 - j
        for c in range(NUM_COLORS):
            if colors_revealed_seat[j, c] > 0:
                out[spec_idx].add(("color", c))
        for r0 in range(NUM_RANKS):
            if ranks_revealed_seat[j, r0] > 0:
                out[spec_idx].add(("rank", r0 + 1))
    return out


def _decode_fireworks(fireworks: np.ndarray) -> dict[int, int]:
    """fireworks is (5 colors, 5 ranks) one-hot per color (highest played rank).

    Returns ``{color: highest_rank_played}`` where 0 means no card of that
    color has been played; rank values are 1-indexed (so a 5 means the
    color is complete).
    """
    out: dict[int, int] = {}
    for c in range(NUM_COLORS):
        col = fireworks[c]
        # "highest played" = index of the last 1; if all zeros, 0.
        nz = np.argwhere(col)
        out[c] = int(nz.max()) + 1 if len(nz) > 0 else 0
    return out


def _decode_discards(discard_pile: np.ndarray) -> list[tuple[int, int]]:
    """discard_pile is (50 positions, 5 colors, 5 ranks) one-hot.

    Returns the list of ``(color, rank_1_indexed)`` cards discarded so far
    (in deal order — jaxmarl appends new discards monotonically).
    """
    out: list[tuple[int, int]] = []
    for p in range(discard_pile.shape[0]):
        slice_ = discard_pile[p]
        if slice_.sum() <= 0:
            continue
        idxs = np.argwhere(slice_)
        if len(idxs) == 0:
            continue
        c, r0 = int(idxs[0, 0]), int(idxs[0, 1])
        out.append((c, r0 + 1))
    return out


# ---------------------------------------------------------------------------
# Action decoder (jaxmarl int -> action_type + detail)
# ---------------------------------------------------------------------------


def _decode_action(action_int: int, actor_seat: int) -> dict[str, Any]:
    """Decode a single jaxmarl action int into a TurnState-compatible detail.

    Returns a dict with at least ``action_type``. For play/discard adds
    ``slot`` (spec, 1-indexed). For clues adds ``target_seat``,
    ``color`` xor ``rank``, ``slots_touched`` filled later by the replay.
    """
    if 0 <= action_int <= 4:
        # discard
        j = action_int
        return {"action_type": "discard", "slot": jaxmarl_slot_to_spec(j)}
    if 5 <= action_int <= 9:
        # play
        j = action_int - 5
        return {"action_type": "play", "slot": jaxmarl_slot_to_spec(j)}
    if 10 <= action_int <= 14:
        # color clue
        c = action_int - 10
        target = 1 - actor_seat  # 2-player: clue goes to the other seat
        return {
            "action_type": "clue_color",
            "target_seat": target,
            "color": c,
        }
    if 15 <= action_int <= 19:
        # rank clue (action 15 = H1 = rank 1 = rank index 0; expose as 1-indexed)
        r1 = action_int - 15 + 1
        target = 1 - actor_seat
        return {
            "action_type": "clue_rank",
            "target_seat": target,
            "rank": r1,
        }
    if action_int == NOOP_ID:
        return {"action_type": "noop"}
    raise ValueError(f"Unknown action int: {action_int}")


# ---------------------------------------------------------------------------
# H-Group focus rule (computed at clue time)
# ---------------------------------------------------------------------------


def _compute_focus(
    receiver_pre_clue_history: list[set[tuple[str, int]]],
    receiver_pre_hand: list[tuple[int, int] | None],
    slots_touched_spec: list[int],
) -> tuple[int | None, list[int], list[int], int | None]:
    """Apply the H-Group focus rule on ``receiver`` at clue time.

    Inputs are pre-clue (the clue specifier has not been added yet).

    Returns ``(focus_slot, newly_touched, re_touched, pre_chop_slot)`` where:

    * ``focus_slot``: spec slot of focus card, or None if the clue is
      degenerate (touches no slots that have a card).
    * ``newly_touched``: spec slots that were unclued before this clue.
    * ``re_touched``: spec slots that already had at least one prior clue.
    * ``pre_chop_slot``: rightmost-unclued spec slot in receiver's hand
      (highest spec slot index whose pre-clue history is empty AND the slot
      has a card). ``None`` if hand is locked (no unclued non-empty slots).
    """
    newly: list[int] = []
    re_t: list[int] = []
    for s in slots_touched_spec:
        idx = s - 1
        if idx < 0 or idx >= HAND_SIZE:
            continue
        if receiver_pre_hand[idx] is None:
            continue  # empty slot; clue can't touch
        if not receiver_pre_clue_history[idx]:
            newly.append(s)
        else:
            re_t.append(s)

    # pre_chop_slot
    pre_chop: int | None = None
    for s in range(HAND_SIZE, 0, -1):
        idx = s - 1
        if receiver_pre_hand[idx] is None:
            continue
        if not receiver_pre_clue_history[idx]:
            pre_chop = s
            break

    # Focus rule
    if not newly and not re_t:
        return None, newly, re_t, pre_chop  # degenerate clue (shouldn't happen)
    if not newly:
        # all re-touched -> leftmost re-touched (smallest spec index)
        return min(re_t), newly, re_t, pre_chop
    if len(newly) == 1:
        return newly[0], newly, re_t, pre_chop
    # multiple newly touched
    if pre_chop is not None and pre_chop in newly:
        return pre_chop, newly, re_t, pre_chop
    return min(newly), newly, re_t, pre_chop


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def replay_game(
    actions: np.ndarray,
    deck: np.ndarray,
    num_actions: int | None = None,
    *,
    env=None,
) -> list[TurnState]:
    """Replay one game and return one TurnState per real (non-noop) turn.

    ``actions`` is shape ``(max_T, NUM_PLAYERS)`` int32. Only the entry at
    ``actions[t, cur_seat]`` is meaningful; the other seat is noop. ``deck``
    is shape ``(50, 2)`` int32 with each row ``(color, rank_0_indexed)``.

    ``num_actions`` is the number of real (non-padded) turns; if None, the
    function walks until ``dones["__all__"]`` triggers.
    """
    if env is None:
        env = make("hanabi", num_agents=NUM_PLAYERS)
    agents_tuple = tuple(env.agents)

    obs, state = env.reset_from_deck_of_pairs(jnp.asarray(deck))
    out: list[TurnState] = []

    T = int(actions.shape[0]) if num_actions is None else int(num_actions)
    last_clue_turn_for_seat: dict[int, int] = {}  # seat -> turn index of last clue received

    for t in range(T):
        # Pre-action seat
        cur_seat = int(np.argmax(np.asarray(state.cur_player_idx)))
        action_int = int(actions[t, cur_seat])
        if action_int == NOOP_ID:
            # Game is already over — we should not be advancing further. Stop.
            break
        action_detail = _decode_action(action_int, cur_seat)

        # Pre-action snapshots for both seats.
        ph = np.asarray(state.player_hands)  # (2, 5, 5, 5)
        cr = np.asarray(state.colors_revealed)  # (2, 5, 5)
        rr = np.asarray(state.ranks_revealed)  # (2, 5, 5)
        pre_hands = [_decode_hand(ph[p]) for p in range(NUM_PLAYERS)]
        pre_history = [_decode_clue_history(cr[p], rr[p]) for p in range(NUM_PLAYERS)]

        # If clue, compute slots_touched, focus, etc. on receiver's pre-clue state.
        if action_detail["action_type"] in ("clue_color", "clue_rank"):
            target = action_detail["target_seat"]
            target_hand = pre_hands[target]
            slots_touched: list[int] = []
            if action_detail["action_type"] == "clue_color":
                c = action_detail["color"]
                for s in range(1, HAND_SIZE + 1):
                    card = target_hand[s - 1]
                    if card is None:
                        continue
                    if card[0] == c:
                        slots_touched.append(s)
            else:
                r1 = action_detail["rank"]
                for s in range(1, HAND_SIZE + 1):
                    card = target_hand[s - 1]
                    if card is None:
                        continue
                    if card[1] == r1:
                        slots_touched.append(s)
            action_detail["slots_touched"] = slots_touched
            focus, newly, re_t, pre_chop = _compute_focus(
                pre_history[target], target_hand, slots_touched
            )
            action_detail["focus_slot"] = focus
            action_detail["newly_touched"] = newly
            action_detail["re_touched"] = re_t
            action_detail["pre_chop_slot"] = pre_chop  # spec slot or None (locked)
        else:
            # play / discard — record the (true) color/rank of the affected card
            _j = spec_slot_to_jaxmarl(action_detail["slot"])
            true_card = pre_hands[cur_seat][action_detail["slot"] - 1]
            if true_card is not None:
                action_detail["color"] = true_card[0]
                action_detail["rank"] = true_card[1]

        # For latency (M11) — track when each seat last RECEIVED a clue.
        # latency_for_this_action available BEFORE updating; it's a play action.
        if action_detail["action_type"] == "play":
            last_clue_t = last_clue_turn_for_seat.get(cur_seat, None)
            if last_clue_t is None:
                action_detail["latency"] = None
            else:
                # "If the most recent clue was given on the immediately preceding
                # turn for that player, latency = 0." That immediately preceding
                # turn for seat S is the one that occurred just before S's current
                # turn. So latency = (number of turns by seat S since the clue).
                # Our `last_clue_t` is the absolute turn index when the clue was
                # given. The number of seat-S plays between then and now is
                # roughly (t - last_clue_t - 1) // NUM_PLAYERS — but only the
                # actor's own turns count.
                latency = (t - last_clue_t - 1) // NUM_PLAYERS
                action_detail["latency"] = max(0, latency)

        # Step env
        env_act = {agents_tuple[i]: jnp.int32(int(actions[t, i])) for i in range(NUM_PLAYERS)}
        rng_step = jax.random.PRNGKey(0)
        obs, state, reward, dones, _info = env.step_env(rng_step, state, env_act)

        # Post-action board (for M12 final strikes; for fireworks/discard updates)
        post_fireworks = _decode_fireworks(np.asarray(state.fireworks))
        post_discards = _decode_discards(np.asarray(state.discard_pile))
        # life_tokens starts at 3; jaxmarl decrements on bombs. Strikes = 3 - sum(life_tokens).
        life = float(np.asarray(state.life_tokens).sum())
        strikes = int(round(3 - life))
        post_board = {
            "fireworks": post_fireworks,
            "discards": post_discards,
            "deck_remaining": int(state.num_cards_dealt)
            - sum(1 for h in pre_hands for c in h if c is not None),
            "strikes": strikes,
        }

        ts = TurnState(
            seat=cur_seat,
            action_type=action_detail["action_type"],
            action_detail=action_detail,
            pre_obs=None,
            post_board=post_board,
            hand_clue_history=pre_history,
            hands=pre_hands,
        )
        out.append(ts)

        # Update last_clue_turn_for_seat AFTER processing this turn:
        # this clue (if any) is now the most recent for the receiver.
        if action_detail["action_type"] in ("clue_color", "clue_rank"):
            last_clue_turn_for_seat[action_detail["target_seat"]] = t

        if bool(dones["__all__"]):
            break

    return out


__all__ = [
    "replay_game",
    "jaxmarl_slot_to_spec",
    "spec_slot_to_jaxmarl",
]
