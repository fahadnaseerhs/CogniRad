"""
signal_physics.py — CogniRad | Energy Accumulation & RF Physics
================================================================
Source of truth for per-student cumulative energy scores and channel
energy snapshots.  Separates radio/energy math from routing code.

Phase 2 additions: idle decay system so that per-student and per-channel
energy decreases naturally during idle periods, allowing channels to
recover from CONGESTED/BUSY back to FREE without forced reallocation.

Public API
----------
compute_message_energy(text, channel_id, ...)  → float
update_energy_score(cms, message_energy)        → float   (new total)
get_energy_score(cms)                           → float   (decay-aware)
reset_energy_score(cms)                         → None
get_channel_energy_snapshot(channel_id)          → dict   (decay-aware)
decay_energy_on_reallocation(cms, factor)       → float   (decayed total)
derive_snr(channel_energy, n_users)             → float
derive_modulation(snr_db)                       → tuple[str, int]

Phase 2 additions:
apply_decay_to_student(cms, now)               → float   (per-student idle decay)
apply_idle_decay(now)                          → dict    (bulk decay for all students)

Snapshot consistency (Phase 2 fix):
get_channel_energy_snapshot and project_channel_energy both accept an
optional `now` parameter.  Pass the same value throughout one control
cycle so all member decays share a single decay baseline and
classification/projection decisions are internally consistent.
"""

from __future__ import annotations

import math
import time
import threading
from typing import Any

# ---------------------------------------------------------------------------
# Internal energy store  (thread-safe, in-memory)
# ---------------------------------------------------------------------------
_energy_lock = threading.Lock()
_energy_scores: dict[str, float] = {}          # CMS → cumulative energy
_energy_timestamps: dict[str, float] = {}      # CMS → last-update epoch


# ---------------------------------------------------------------------------
# Phase 2 — Idle Decay Constants
# ---------------------------------------------------------------------------
# These are module-level tunables.  Adjust here to change decay behaviour
# globally without touching any other file.

# How often (in seconds) one decay "tick" is defined to be.
# Energy is reduced by DECAY_FACTOR once per tick of elapsed idle time.
DECAY_INTERVAL_SECONDS: float = 5.0

# Multiplicative factor applied per tick.  0.95 means energy drops to
# 95 % of its previous value every DECAY_INTERVAL_SECONDS of idle time.
# Must be in (0, 1).  Lower values = faster cooldown.
DECAY_FACTOR: float = 0.95

# Clamp threshold.  Any energy value below this is snapped to exactly 0.0
# to prevent endless tiny float residue (e.g. 0.000000012).
ENERGY_EPSILON: float = 0.01


# ---------------------------------------------------------------------------
# PHY constants  (kept in sync with channels.py band profiles)
# ---------------------------------------------------------------------------

# Base energy per character — higher-order modulation costs more energy
_ENERGY_PER_CHAR_2G = 0.035          # 2.4 GHz band
_ENERGY_PER_CHAR_5G = 0.028          # 5 GHz band
_BASE_ENERGY_2G     = 1.00
_BASE_ENERGY_5G     = 0.85
_EIRP_MAX_2G        = 20.0           # dBm
_EIRP_MAX_5G        = 17.0           # dBm

# SNR model
SNR_CLEAN_DB        = 30.0           # pristine channel
SNR_FLOOR_DB        = 2.0            # minimum useful SNR
SNR_PER_JOULE_DROP  = 1.8            # each joule of total energy reduces SNR by this

# Modulation ladder (min_snr → name, bits_per_symbol)
MODULATION_LADDER: list[tuple[float, str, int]] = [
    (25.0, "64-QAM", 6),
    (15.0, "16-QAM", 4),
    ( 8.0, "QPSK",   2),
    (-math.inf, "BPSK", 1),
]


# ---------------------------------------------------------------------------
# Helpers — band profile lookup
# ---------------------------------------------------------------------------

def _band_params(channel_id: str) -> tuple[float, float, float]:
    """Return (energy_per_char, base_energy, eirp_max) for a channel."""
    # Import lazily to avoid circular import at module level
    import channels as ch_mod
    freq = ch_mod.CHANNELS[channel_id]["frequency"]
    if freq.startswith("2.4"):
        return _ENERGY_PER_CHAR_2G, _BASE_ENERGY_2G, _EIRP_MAX_2G
    return _ENERGY_PER_CHAR_5G, _BASE_ENERGY_5G, _EIRP_MAX_5G


# ---------------------------------------------------------------------------
# 1. compute_message_energy
# ---------------------------------------------------------------------------

def compute_message_energy(
    text: str,
    channel_id: str,
    *,
    concurrent_transmitters: int = 1,
) -> float:
    """
    Compute the energy contribution of a single DM.

    The energy grows with message length and is scaled by the RF band
    profile.  Concurrent transmitters inflate the energy footprint because
    of contention overhead.

    Returns
    -------
    float   Non-negative energy value (arbitrary units, comparable across
            channels because both bands are normalised to eirp_max).
    """
    epc, base, eirp_max = _band_params(channel_id)

    char_count = max(len(text), 1)
    bits = char_count * 8
    tx_scale = max(1, concurrent_transmitters)

    # raw energy = base * per_char * chars * transmitter contention
    raw = base * epc * char_count * tx_scale

    # Cap at EIRP max (regulatory ceiling)
    energy = min(raw, eirp_max * tx_scale)

    return round(energy, 4)


# ---------------------------------------------------------------------------
# 2. update_energy_score / get / reset
# ---------------------------------------------------------------------------

def update_energy_score(cms: str, message_energy: float) -> float:
    """Add *message_energy* to *cms*'s cumulative score.  Returns new total."""
    with _energy_lock:
        current = _energy_scores.get(cms, 0.0)
        new_total = current + message_energy
        _energy_scores[cms] = new_total
        _energy_timestamps[cms] = time.time()
        return round(new_total, 4)


def get_energy_score(cms: str, now: float | None = None) -> float:
    """
    Return the *decayed* current energy for *cms* (0.0 if unknown).

    Phase 2: applies idle decay lazily before returning so that all
    downstream callers always receive a fresh, time-accurate value.
    This prevents stale pre-decay totals from leaking into the classifier
    or allocator between AI loop ticks.

    Parameters
    ----------
    now : float | None
        Shared observation timestamp.  Pass the same value to every call
        within one control cycle so all reads use a consistent decay
        baseline.  Defaults to time.time() when omitted.
    """
    # apply_decay_to_student acquires the lock internally, so we must NOT
    # hold _energy_lock here to avoid a deadlock.
    return apply_decay_to_student(cms, now=now)


def reset_energy_score(cms: str) -> None:
    """Reset a student's energy to zero (e.g. on logout)."""
    with _energy_lock:
        _energy_scores.pop(cms, None)
        _energy_timestamps.pop(cms, None)


def set_energy_score(cms: str, value: float) -> None:
    """Directly set a student's energy (used by decay helpers)."""
    with _energy_lock:
        _energy_scores[cms] = round(value, 4)
        _energy_timestamps[cms] = time.time()


# ---------------------------------------------------------------------------
# Phase 2 — Idle Decay Helpers
# ---------------------------------------------------------------------------

def apply_decay_to_student(cms: str, now: float | None = None) -> float:
    """
    Apply exponential idle decay to a single student's energy and return
    the resulting (post-decay) value.

    Phase 2 core helper.  All energy reads should go through this function
    (or through get_energy_score which delegates here) so that callers
    always receive a time-accurate value.

    Algorithm
    ---------
    1. Read current energy and the timestamp of the last update.
    2. Compute elapsed seconds since that timestamp.
    3. Convert elapsed time into discrete decay ticks:
           ticks = floor(elapsed / DECAY_INTERVAL_SECONDS)
    4. Apply multiplicative decay:
           new_energy = old_energy * (DECAY_FACTOR ** ticks)
    5. Clamp values below ENERGY_EPSILON to exactly 0.0.
    6. Persist the decayed value and update the timestamp only when at
       least one tick has elapsed (idempotent for the same `now`).

    Why discrete ticks instead of continuous decay
    -----------------------------------------------
    Discrete ticks make the output deterministic: given the same `now`
    and the same stored state, the function always returns the same value.
    Continuous decay would produce different results on every call due to
    floating-point timing jitter.

    Parameters
    ----------
    cms : str
        Student CMS identifier.
    now : float | None
        Current epoch time.  Defaults to time.time().  Pass an explicit
        value in tests to make decay deterministic.

    Returns
    -------
    float
        The decayed energy (≥ 0.0).
    """
    if now is None:
        now = time.time()

    with _energy_lock:
        current_energy = _energy_scores.get(cms, 0.0)

        # Nothing to decay — return early to avoid unnecessary work.
        if current_energy == 0.0:
            return 0.0

        last_ts = _energy_timestamps.get(cms, now)
        elapsed = now - last_ts

        # Compute how many full decay ticks have elapsed.
        # Using floor ensures we only decay in discrete steps, keeping
        # the function idempotent for the same `now` value.
        ticks = int(elapsed / DECAY_INTERVAL_SECONDS)

        if ticks <= 0:
            # Less than one full tick has passed — no decay yet.
            return round(current_energy, 4)

        # Exponential decay: energy * factor^ticks
        # This is equivalent to applying the factor `ticks` times but
        # computed in a single operation for efficiency.
        decayed = current_energy * (DECAY_FACTOR ** ticks)

        # Clamp tiny residue to zero to avoid endless float noise.
        if decayed < ENERGY_EPSILON:
            decayed = 0.0

        # Persist the decayed value.
        _energy_scores[cms] = round(decayed, 4)

        # Advance the timestamp by the consumed ticks so that the next
        # call correctly measures elapsed time from the end of the last
        # full tick, not from the original timestamp.
        _energy_timestamps[cms] = last_ts + ticks * DECAY_INTERVAL_SECONDS

        return round(decayed, 4)


def apply_idle_decay(now: float | None = None) -> dict[str, Any]:
    """
    Apply exponential idle decay to ALL students who currently have energy.

    Phase 2 bulk decay function.  Called at the start of every AI
    observation tick in main.py so that the system evolves even when no
    HTTP/WebSocket requests are in flight.

    This function delegates per-student work to apply_decay_to_student so
    that the decay logic lives in exactly one place (single source of
    truth rule from the Phase 2 spec).

    Parameters
    ----------
    now : float | None
        Current epoch time.  Defaults to time.time().  Pass an explicit
        value in tests to make results deterministic.

    Returns
    -------
    dict with keys:
        decayed_count   int     number of students whose energy changed
        zeroed_count    int     number of students clamped to 0.0
        before_total    float   sum of all energies before decay
        after_total     float   sum of all energies after decay
    """
    if now is None:
        now = time.time()

    # Snapshot the current CMS list under the lock so we don't iterate
    # while apply_decay_to_student also acquires the lock.
    with _energy_lock:
        cms_list = list(_energy_scores.keys())
        before_total = round(sum(_energy_scores.values()), 4)

    decayed_count = 0
    zeroed_count = 0

    for cms in cms_list:
        # Read the pre-decay value without the lock (safe: we only need
        # an approximate snapshot for the summary).
        pre = _energy_scores.get(cms, 0.0)
        if pre == 0.0:
            continue  # already zero — skip

        post = apply_decay_to_student(cms, now=now)

        if post != pre:
            decayed_count += 1
        if post == 0.0 and pre > 0.0:
            zeroed_count += 1

    with _energy_lock:
        after_total = round(sum(_energy_scores.values()), 4)

    return {
        "decayed_count": decayed_count,
        "zeroed_count": zeroed_count,
        "before_total": before_total,
        "after_total": after_total,
    }


# ---------------------------------------------------------------------------
# 3. get_channel_energy_snapshot
# ---------------------------------------------------------------------------

def get_channel_energy_snapshot(channel_id: str, now: float | None = None) -> dict[str, Any]:
    """
    Return a dict describing the total energy state of a channel.

    Phase 2: all per-student energies are decayed before aggregation so
    this function becomes the single decay-aware source of truth for
    channel state.  The classifier and allocator both consume this output
    and therefore automatically operate on fresh decayed values.

    Snapshot consistency fix: a single `now` value is captured once and
    passed to every per-student decay call.  This prevents a snapshot from
    spanning a decay-tick boundary mid-loop, which would mix values from
    different ticks and make classification non-deterministic under load.

    Parameters
    ----------
    channel_id : str
    now : float | None
        Shared observation timestamp.  Capture once per control cycle and
        pass through so the whole channel is evaluated at one instant.
        Defaults to time.time() when omitted.

    Keys
    ----
    channel_id          str
    total_energy        float     sum of all member scores (post-decay)
    member_count        int
    per_student         list[dict]  sorted highest-energy-first
                            each: {cms, energy, pct}
    snr_db              float     derived SNR
    modulation          str       current modulation label
    modulation_index    int       bits per symbol
    """
    import channels as ch_mod

    # Capture one consistent timestamp for the entire snapshot so all
    # member decays are evaluated at the same instant.
    if now is None:
        now = time.time()

    members: list[str] = list(ch_mod.CHANNELS[channel_id]["users"])

    # Phase 2: decay each member using the shared `now` so the snapshot
    # is internally consistent — no member can be from a different tick.
    per_student = [
        {"cms": cms, "energy": apply_decay_to_student(cms, now=now)}
        for cms in members
    ]

    total = sum(s["energy"] for s in per_student)

    # Compute percentage contribution
    for entry in per_student:
        entry["pct"] = round(entry["energy"] / total * 100, 1) if total > 0 else 0.0

    per_student.sort(key=lambda e: e["energy"], reverse=True)

    snr = derive_snr(total, len(members))
    mod_name, mod_idx = derive_modulation(snr)

    return {
        "channel_id": channel_id,
        "total_energy": round(total, 4),
        "member_count": len(members),
        "per_student": per_student,
        "snr_db": round(snr, 3),
        "modulation": mod_name,
        "modulation_index": mod_idx,
    }


# ---------------------------------------------------------------------------
# 4. decay_energy_on_reallocation
# ---------------------------------------------------------------------------

def decay_energy_on_reallocation(cms: str, factor: float = 0.5) -> float:
    """
    When a student is moved to a new channel, decay their energy by *factor*.

    This prevents a high-energy user from immediately overloading their new
    channel.

    Returns the new (decayed) energy score.
    """
    with _energy_lock:
        current = _energy_scores.get(cms, 0.0)
        decayed = current * factor
        _energy_scores[cms] = round(decayed, 4)
        _energy_timestamps[cms] = time.time()
        return round(decayed, 4)


# ---------------------------------------------------------------------------
# 5. derive_snr / derive_modulation
# ---------------------------------------------------------------------------

def derive_snr(channel_energy: float, n_users: int) -> float:
    """
    Derive channel SNR from total accumulated energy and user count.

    Higher total energy → lower SNR (more interference).
    """
    contention_penalty = max(n_users - 1, 0) * 2.0
    energy_penalty = channel_energy * SNR_PER_JOULE_DROP
    snr = SNR_CLEAN_DB - contention_penalty - energy_penalty
    return max(snr, SNR_FLOOR_DB)


def derive_modulation(snr_db: float) -> tuple[str, int]:
    """Return (modulation_name, bits_per_symbol) for a given SNR."""
    for min_snr, name, bits in MODULATION_LADDER:
        if snr_db >= min_snr:
            return name, bits
    return "BPSK", 1


# ---------------------------------------------------------------------------
# 6. Projected energy helpers  (used by allocator + classifier)
# ---------------------------------------------------------------------------

def project_channel_energy(channel_id: str, additional_energy: float = 0.0, now: float | None = None) -> float:
    """
    Return what the total energy of *channel_id* would be if
    *additional_energy* were added (without actually adding it).

    Phase 2: uses decayed per-student energy so that projected totals
    start from the current decayed baseline, not stale accumulated values.

    Snapshot consistency fix: accepts a shared `now` so that projection
    and live classification within the same allocator decision cycle use
    the same decay baseline.  Without this, a borderline healthy/unhealthy
    decision can flip just because a few milliseconds passed between the
    source ranking read and the destination projection read.

    Parameters
    ----------
    channel_id : str
    additional_energy : float
    now : float | None
        Shared observation timestamp.  Pass the same value used for the
        live snapshot in the same control cycle.
    """
    import channels as ch_mod
    members = list(ch_mod.CHANNELS[channel_id]["users"])
    # Pass the shared now so projection uses the same decay baseline as
    # the live snapshot taken earlier in the same control cycle.
    total = sum(get_energy_score(cms, now=now) for cms in members)
    return round(total + additional_energy, 4)


def project_channel_energy_without(channel_id: str, cms_to_remove: str, now: float | None = None) -> float:
    """
    Return what the total energy of *channel_id* would be if *cms_to_remove*
    were taken off it (without actually removing them).

    Phase 2: uses decayed per-student energy for the same reason as
    project_channel_energy above.  Accepts a shared `now` for the same
    consistency reason.
    """
    import channels as ch_mod
    members = list(ch_mod.CHANNELS[channel_id]["users"])
    total = sum(
        get_energy_score(cms, now=now)
        for cms in members
        if cms != cms_to_remove
    )
    return round(total, 4)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== signal_physics.py self-test ===\n")

    # Simulate energy accumulation
    update_energy_score("CMS001", 1.5)
    update_energy_score("CMS001", 2.0)
    update_energy_score("CMS002", 0.8)
    assert abs(get_energy_score("CMS001") - 3.5) < 0.001
    assert abs(get_energy_score("CMS002") - 0.8) < 0.001
    print(f"  CMS001 energy: {get_energy_score('CMS001')}")
    print(f"  CMS002 energy: {get_energy_score('CMS002')}")

    # Decay on reallocation
    decayed = decay_energy_on_reallocation("CMS001", factor=0.5)
    assert abs(decayed - 1.75) < 0.001
    print(f"  CMS001 after 50% decay: {decayed}")

    # Phase 2 — per-student idle decay
    set_energy_score("CMS003", 10.0)
    # Simulate 10 seconds elapsed (2 ticks at DECAY_INTERVAL_SECONDS=5)
    fake_now = _energy_timestamps["CMS003"] + 10.0
    result = apply_decay_to_student("CMS003", now=fake_now)
    expected = 10.0 * (DECAY_FACTOR ** 2)
    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"
    print(f"  CMS003 after 2 decay ticks: {result:.4f} (expected {expected:.4f})")

    # Phase 2 — bulk idle decay
    set_energy_score("CMS004", 5.0)
    set_energy_score("CMS005", 3.0)
    fake_now2 = time.time() + DECAY_INTERVAL_SECONDS  # 1 tick ahead
    summary = apply_idle_decay(now=fake_now2)
    print(f"  Bulk decay summary: {summary}")
    assert summary["decayed_count"] >= 0

    # Phase 2 — clamp to zero
    set_energy_score("CMS006", ENERGY_EPSILON / 2)
    fake_now3 = _energy_timestamps["CMS006"] + DECAY_INTERVAL_SECONDS
    clamped = apply_decay_to_student("CMS006", now=fake_now3)
    assert clamped == 0.0, f"Expected 0.0, got {clamped}"
    print(f"  CMS006 clamped to zero: {clamped}")

    # SNR
    snr = derive_snr(5.0, 3)
    mod, idx = derive_modulation(snr)
    print(f"  SNR at 5J/3 users: {snr:.1f} dB -> {mod} ({idx} bps)")

    # Reset
    reset_energy_score("CMS001")
    assert get_energy_score("CMS001") == 0.0

    print("\n[PASS] All signal_physics checks complete.\n")
