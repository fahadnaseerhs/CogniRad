"""
allocator.py — CogniRad | Decision Engine
==========================================
Energy-aware, fair, minimum-move reallocation engine.

Phase 2: the allocator does not implement any decay logic itself.  It
reads fresh decayed energy values from signal_physics via get_energy_score
(which is decay-aware) and classify_channel_projected (which uses
project_channel_energy, also decay-aware).  This keeps decay as a single
source of truth in signal_physics.py and ensures the allocator always
operates on post-decay state, never on stale accumulated totals.

The AI loop in main.py calls apply_idle_decay() before calling
reallocate_users(), so by the time the allocator runs, all energy values
are already current.

Public API
----------
assign_channel(cms)                    → dict    Assign a new student to least-loaded channel.
check_congestion()                     → list    Return all unhealthy channels.
reallocate_users(source_channel_key)   → list    Move minimum students to fix an overloaded channel.

Design Principles
-----------------
* Selection uses per-student energy contribution, NOT recent join time.
* A round-robin fairness pointer prevents the same student from always
  being the first victim.
* Destination validation uses projected classification — a move is
  accepted only if the destination remains healthy after absorbing the
  decayed energy.
* Minimum-move: after each move, reclassify the source and stop as soon
  as it becomes healthy.
* Admin-forced JAMMED is never silently overwritten.
"""

from __future__ import annotations

import time
from typing import Any

import channels as ch_mod
import classifier
import database
import signal_physics as sp


# ---------------------------------------------------------------------------
# Module-level fairness state
# ---------------------------------------------------------------------------
# One pointer per source channel.  Tracks the index into the energy-sorted
# member list where we last started looking for reallocation candidates.
# This ensures round-robin fairness across reallocation events.
_reallocation_pointer: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _channel_key_to_db_id(channel_key: str) -> int:
    return int(channel_key.split("-")[1])


def _db_id_to_channel_key(db_id: int) -> str:
    return f"CH-{db_id}"


# ---------------------------------------------------------------------------
# 1. assign_channel
# ---------------------------------------------------------------------------

async def assign_channel(cms: str) -> dict[str, Any]:
    """
    Assign student *cms* to the least-loaded, non-JAMMED channel.

    Steps
    -----
    1. Pick channel with fewest users (skips JAMMED).
    2. Add student to in-memory user list.
    3. Persist to DB.
    4. Return assignment summary.
    """
    best = ch_mod.get_least_loaded_channel()
    channel_key: str = best["channel_id"]
    channel_data = ch_mod.CHANNELS[channel_key]

    # In-memory
    if cms not in channel_data["users"]:
        channel_data["users"].append(cms)

    # Initialise energy score if new
    if sp.get_energy_score(cms) == 0.0:
        sp.set_energy_score(cms, 0.0)

    # Classify after join
    result = classifier.classify_channel(channel_key)
    channel_data["status"] = result["status"]

    # Persist
    db_id = _channel_key_to_db_id(channel_key)
    await database.assign_student_to_channel(cms, db_id)
    await database.update_channel_status(
        db_id,
        result["status"],
        result["confidence"],
        is_jammed=(result["status"] == "JAMMED"),
    )

    return {
        "cms": cms,
        "channel_key": channel_key,
        "channel_id": db_id,
        "frequency": channel_data["frequency"],
        "user_count": len(channel_data["users"]),
        "status": result["status"],
    }


# ---------------------------------------------------------------------------
# 2. check_congestion
# ---------------------------------------------------------------------------

async def check_congestion() -> list[dict[str, Any]]:
    """
    Inspect every channel; return those that are CONGESTED or JAMMED.

    Reads status directly (does not reclassify to avoid overwriting
    externally set JAMMED).
    """
    congested: list[dict[str, Any]] = []

    for channel_key, channel_data in ch_mod.CHANNELS.items():
        status = channel_data["status"]

        if status in {"CONGESTED", "JAMMED"}:
            db_id = _channel_key_to_db_id(channel_key)
            await database.update_channel_status(
                db_id,
                status,
                confidence=0.0,
                is_jammed=(status == "JAMMED"),
            )
            congested.append({
                "channel_key": channel_key,
                "status": status,
                "user_count": len(channel_data["users"]),
                "frequency": channel_data["frequency"],
            })

    return congested


# ---------------------------------------------------------------------------
# 3. reallocate_users — the decision engine
# ---------------------------------------------------------------------------

def _find_valid_destination(
    cms: str,
    source_key: str,
    now: float | None = None,
) -> str | None:
    """
    Find a healthy destination channel for *cms*.

    For every candidate channel (excluding source and JAMMED channels),
    compute the projected energy if *cms* moved there with decayed energy.
    Accept only destinations that remain healthy after the move.

    Phase 3: Gathers all valid candidates and deterministically chooses
    the optimal destination based on:
      1. Lowest projected total energy
      2. Lowest projected confidence
      3. Channel key (tie-breaker)

    Returns channel_key or None.
    """
    decayed_energy = sp.get_energy_score(cms, now=now) * 0.5  # relocation carry-decay preview

    valid_destinations = []

    for dest_key, dest_data in ch_mod.CHANNELS.items():
        if dest_key == source_key:
            continue
        if dest_data["status"] == "JAMMED":
            continue

        # Project destination health with this student's decayed energy.
        proj = classifier.classify_channel_projected(dest_key, decayed_energy, now=now)
        if classifier.is_healthy(proj):
            valid_destinations.append({
                "key": dest_key,
                "total_energy": proj["total_energy"],
                "confidence": proj["confidence"]
            })

    if not valid_destinations:
        return None

    # Sort to find the optimal destination
    valid_destinations.sort(key=lambda d: (d["total_energy"], d["confidence"], d["key"]))
    return valid_destinations[0]["key"]


async def reallocate_users(source_key: str, now: float | None = None) -> list[dict[str, Any]]:
    """
    Move the minimum number of students off *source_key* to restore health.

    Algorithm
    ---------
    1. Capture one shared `now` for the entire decision cycle.
    2. Get all members sorted by energy (highest first) using that `now`.
    3. Apply round-robin fairness pointer to rotate the start.
    4. For each candidate:
       a. Find a valid destination (projected-safe, same `now`).
       b. Move the student (in-memory + DB).
       c. Decay their carried energy.
       d. Reclassify source — stop if healthy.
    5. Return list of moves made.

    Admin-forced JAMMED channels evacuate ALL users (pointer resets).

    Snapshot consistency fix: a single `now` is captured at the top and
    threaded through all energy reads, source ranking, and destination
    projections.  This ensures the entire decision cycle uses one
    consistent decay baseline — source ranking and destination validation
    cannot diverge because a tick boundary was crossed mid-loop.
    """
    source_data = ch_mod.CHANNELS.get(source_key)
    if source_data is None or not source_data["users"]:
        return []

    # Capture one shared observation timestamp for the entire decision
    # cycle.  Every energy read, ranking, and projection in this call
    # will use this same instant so decisions are internally consistent.
    if now is None:
        now = time.time()

    admin_jammed = source_data["status"] == "JAMMED"
    source_db_id = _channel_key_to_db_id(source_key)

    # Build energy-sorted member list.
    # Phase 2: get_energy_score is decay-aware.  Pass the shared `now`
    # so ranking uses the same decay baseline as destination projections.
    members = list(source_data["users"])
    members_ranked = sorted(
        members,
        key=lambda cms: sp.get_energy_score(cms, now=now),
        reverse=True,  # highest energy first
    )

    # Apply fairness pointer (rotate start position)
    pointer = _reallocation_pointer.get(source_key, 0) % max(len(members_ranked), 1)
    rotated = members_ranked[pointer:] + members_ranked[:pointer]

    moved: list[dict[str, Any]] = []

    for cms in rotated:
        if cms not in source_data["users"]:
            continue  # already moved by a prior iteration

        # Find a valid destination using the shared now so the projection
        # is consistent with the ranking done above.
        dest_key = _find_valid_destination(cms, source_key, now=now)
        if dest_key is None:
            continue  # no safe destination — skip this student

        dest_data = ch_mod.CHANNELS[dest_key]
        dest_db_id = _channel_key_to_db_id(dest_key)

        # ── Execute the move ────────────────────────────────────
        # In-memory
        if cms in source_data["users"]:
            source_data["users"].remove(cms)
        if cms not in dest_data["users"]:
            dest_data["users"].append(cms)

        # Decay energy on relocation (separate from idle decay)
        decayed = sp.decay_energy_on_reallocation(cms, factor=0.5)

        # DB
        await database.move_student(cms, source_db_id, dest_db_id)

        moved.append({
            "cms": cms,
            "from": source_key,
            "to": dest_key,
            "frequency": dest_data["frequency"],
            "decayed_energy": decayed,
        })

        # ── Check if source is now healthy ──────────────────────
        # Use a fresh wall-clock read here (not the shared `now`) because
        # the move has just happened and we want the true current state.
        if not admin_jammed:
            src_result = classifier.classify_channel(source_key)
            source_data["status"] = src_result["status"]
            if classifier.is_healthy(src_result):
                break  # minimum-move: source recovered, stop

    # Advance fairness pointer
    _reallocation_pointer[source_key] = (pointer + len(moved)) % max(len(members_ranked), 1)

    # Update source status in DB
    if not source_data["users"] and not admin_jammed:
        source_data["status"] = "FREE"
    src_result = classifier.classify_channel(
        source_key,
        admin_jammed=admin_jammed,
    )
    source_data["status"] = src_result["status"]
    await database.update_channel_status(
        source_db_id,
        src_result["status"],
        src_result["confidence"],
        is_jammed=admin_jammed,
    )

    # Update destination statuses
    for move in moved:
        dk = move["to"]
        d_result = classifier.classify_channel(dk)
        ch_mod.CHANNELS[dk]["status"] = d_result["status"]
        await database.update_channel_status(
            _channel_key_to_db_id(dk),
            d_result["status"],
            d_result["confidence"],
            is_jammed=(d_result["status"] == "JAMMED"),
        )

    return moved
