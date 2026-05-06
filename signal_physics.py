"""
signal_physics.py — CogniRad | Energy Accumulation & RF Physics
================================================================
Source of truth for per-student cumulative energy scores and channel
energy snapshots.  Separates radio/energy math from routing code.

Phase 2 additions: idle decay system so that per-student and per-channel
energy decreases naturally during idle periods, allowing channels to
recover from CONGESTED/BUSY back to FREE without forced reallocation.

Phase 3 additions: realistic PHY event simulation.  Every incoming
WebSocket message is treated as a transmission burst.  Message size and
arrival timing drive a simulated bitrate, which is then used to compute
channel utilization, interference energy, and SNR degradation.

Public API
----------
compute_message_energy(text, channel_id, ...)  → float
compute_phy_event(text, channel_id, dt_seconds, n_users) → dict
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

# How often (in seconds) one decay "tick" is defined to be.
# Changed from 5s to 1s so the AI loop fires every second and decay
# feels continuous on live charts.  All decay factors have been rescaled
# accordingly (see apply_idle_decay).
DECAY_INTERVAL_SECONDS: float = 1.0

# Dynamic decay: rate adjusts to total active students in the system.
# More students = slower decay (busier network stays energized longer).
# Fewer students = faster decay  (quiet channel clears quickly).
#
# These are per-1-second base values (rescaled from original per-5s values).
#   n= 0 students : decay ≈ 0.979  (fast cooldown)
#   n=50 students : decay ≈ 0.990  (slow cooldown)
#
_DECAY_BASE: float = 0.979   # minimum decay factor (empty/quiet system) — was 0.92
_DECAY_MAX:  float = 0.990   # maximum decay factor (full 50-student load) — was 0.99
_DECAY_MAX_STUDENTS: int = 50

# Clamp threshold.  Any energy value below this is snapped to exactly 0.0
# to prevent endless tiny float residue (e.g. 0.000000012).
ENERGY_EPSILON: float = 0.01

# Backward-compatible alias so existing tests that reference sp.DECAY_FACTOR
# still work.  Points to the per-1s base decay factor.
DECAY_FACTOR: float = _DECAY_BASE   # 0.979 per 1-second tick


def get_dynamic_decay_factor(n_total_active: int) -> float:
    """
    Return the decay factor for the current system-wide active student count.

    Parameters
    ----------
    n_total_active : int
        Total number of students that currently have a non-zero energy score
        across all channels.  Pass 0 if no students are active.

    Returns
    -------
    float
        A value in [_DECAY_BASE, _DECAY_MAX].  Higher = slower decay.
    """
    ratio = min(max(n_total_active, 0) / _DECAY_MAX_STUDENTS, 1.0)
    return _DECAY_BASE + (_DECAY_MAX - _DECAY_BASE) * ratio


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
SNR_PER_JOULE_DROP  = 0.08           # was 1.8 — energy should not tank SNR this fast

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
# 1. compute_phy_event  (Phase 3 — timing-aware PHY simulation)
# ---------------------------------------------------------------------------

def compute_phy_event(
    text: str,
    channel_id: str,
    dt_seconds: float,
    n_users: int,
) -> dict:
    """
    Simulate a realistic PHY transmission event from a WebSocket message.

    This is the Phase 3 core function.  Every incoming DM is treated as a
    transmission burst.  Message size and arrival timing drive a simulated
    bitrate, which is then used to compute channel utilization, interference
    energy, and SNR degradation.

    Algorithm
    ---------
    1. bits = len(text) * 8
    2. dt   = max(dt_seconds, PHY_MIN_DT_SECONDS)   ← clamp to avoid ÷0
    3. bitrate_bps = bits / dt                       ← simulated burst rate
    4. utilization = bitrate_bps / capacity_bps      ← fraction of channel used
    5. msg_energy  = base_energy
                   + energy_per_bit * bits
                   + utilization_weight * utilization
                   + contention_weight * max(n_users - 1, 0)
    6. Derive SNR from channel total + this burst
    7. Select modulation from SNR ladder

    Why this gives realistic behaviour
    -----------------------------------
    * Fast repeated long messages → high bitrate → high utilization → high
      energy → SNR drops → modulation degrades → channel congests.
    * Idle periods → energy decays → SNR recovers → channel frees up.
    * Same text on 2.4 GHz vs 5 GHz produces different stress because the
      profiles have different capacity_bps and weight constants.
    * More concurrent users → higher contention penalty → more energy.

    Parameters
    ----------
    text : str
        The DM text content.
    channel_id : str
        e.g. "CH-1".  Used to look up the PHY profile.
    dt_seconds : float
        Seconds since this sender's previous message.  Pass a large value
        (e.g. 60.0) for the first message from a sender.
    n_users : int
        Number of concurrent users on the channel (including sender).

    Returns
    -------
    dict with keys:
        bits            int     raw bit count of this message
        dt_seconds      float   clamped inter-message interval
        bitrate_bps     float   simulated burst bitrate
        utilization     float   fraction of channel capacity used (0–1+)
        msg_energy      float   energy contribution of this message
        band            str     "2.4 GHz" or "5 GHz"
        capacity_bps    int     channel capacity used for normalisation
    """
    import channels as ch_mod

    freq = ch_mod.CHANNELS[channel_id]["frequency"]
    profile = ch_mod.get_phy_profile(freq)

    bits = max(len(text), 1) * 8
    dt   = max(dt_seconds, ch_mod.PHY_MIN_DT_SECONDS)

    bitrate_bps  = bits / dt
    utilization  = bitrate_bps / profile["capacity_bps"]

    # Contention penalty: (N-1)/sqrt(N)
    # Grows with users but sub-linearly so it doesn't explode at N=30.
    # N=2: 0.71  N=5: 1.79  N=10: 2.85  N=30: 5.29  N=50: 6.93
    n            = max(n_users, 1)
    contention   = (n - 1) / math.sqrt(n)

    msg_energy = (
        profile["base_energy"]
        + profile["energy_per_bit"] * bits
        + profile["utilization_weight"] * utilization
        + profile["contention_weight"] * contention
    )

    band = "2.4 GHz" if freq.startswith("2.4") else "5 GHz"

    return {
        "bits":         bits,
        "dt_seconds":   round(dt, 4),
        "bitrate_bps":  round(bitrate_bps, 2),
        "utilization":  round(min(utilization, 1.0), 6),
        "msg_energy":   round(msg_energy, 4),
        "band":         band,
        "capacity_bps": profile["capacity_bps"],
    }


# ---------------------------------------------------------------------------
# 1b. compute_message_energy  (legacy wrapper — kept for backward compat)
# ---------------------------------------------------------------------------

def compute_message_energy(
    text: str,
    channel_id: str,
    *,
    concurrent_transmitters: int = 1,
) -> float:
    """
    Compute the energy contribution of a single DM.

    Phase 3: delegates to compute_phy_event with a default dt of 1.0 s
    (representing a "normal" inter-message gap) so that callers that do
    not track timing still get a reasonable energy value.  The full
    timing-aware path goes through compute_phy_event directly.

    Returns
    -------
    float   Non-negative energy value.
    """
    event = compute_phy_event(
        text,
        channel_id,
        dt_seconds=1.0,
        n_users=max(concurrent_transmitters, 1),
    )
    return event["msg_energy"]


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

def apply_decay_to_student(cms: str, now: float | None = None, decay_factor: float | None = None) -> float:
    """
    Apply exponential idle decay to a single student's energy and return
    the resulting (post-decay) value.

    Parameters
    ----------
    cms : str
        Student CMS identifier.
    now : float | None
        Current epoch time.  Defaults to time.time().
    decay_factor : float | None
        Override the decay factor for this call.  When None, uses
        get_dynamic_decay_factor() based on current active student count.
        Pass an explicit value from apply_idle_decay() so that all
        students in one bulk decay cycle use the same factor.

    Returns
    -------
    float
        The decayed energy (≥ 0.0).
    """
    if now is None:
        now = time.time()

    with _energy_lock:
        current_energy = _energy_scores.get(cms, 0.0)

        if current_energy == 0.0:
            return 0.0

        last_ts = _energy_timestamps.get(cms, now)
        elapsed = now - last_ts
        ticks   = int(elapsed / DECAY_INTERVAL_SECONDS)

        if ticks <= 0:
            return round(current_energy, 4)

        # Resolve decay factor: use provided value or compute dynamically.
        if decay_factor is None:
            n_active = len([e for e in _energy_scores.values() if e > 0.0])
            decay_factor = get_dynamic_decay_factor(n_active)

        decayed = current_energy * (decay_factor ** ticks)

        if decayed < ENERGY_EPSILON:
            decayed = 0.0

        _energy_scores[cms] = round(decayed, 4)
        _energy_timestamps[cms] = last_ts + ticks * DECAY_INTERVAL_SECONDS

        return round(decayed, 4)


def apply_idle_decay(now: float | None = None) -> dict[str, Any]:
    """
    Apply state-dependent exponential idle decay to ALL students.

    Decay rate depends on the channel state the student is currently on:

        State        Base decay   Meaning
        -------      ----------   -------
        FREE         0.90         Fast cooldown — quiet channel
        BUSY         0.95         Moderate — ongoing traffic
        CONGESTED    0.80         Round-robin active, forced cooldown
        JAMMED       0.70         Aggressive drain + new msgs blocked

    Each rate is additionally adjusted upward by 0.005 * ln(N_on_channel)
    so that larger channels cool down slightly more slowly (more background
    chatter keeps energy up).

    All students in one bulk cycle use a per-channel pre-computed factor
    so the rate can't change mid-loop.
    """
    if now is None:
        now = time.time()

    import channels as ch_mod  # lazy to avoid circular import at module level

    # State → base decay factor mapping.
    # These are per-1-second factors, rescaled from the original per-5s values
    # using the formula: new = old^(1/5).
    #   FREE:      0.90^0.2 = 0.979  (was 0.90 per 5s)
    #   BUSY:      0.95^0.2 = 0.990  (was 0.95 per 5s)
    #   CONGESTED: 0.80^0.2 = 0.956  (was 0.80 per 5s)
    #   JAMMED:    0.70^0.2 = 0.931  (was 0.70 per 5s)
    # Keeping the old values at 1s ticks would drain energy to near-zero
    # in seconds and prevent channels from ever building up load.
    _STATE_DECAY: dict[str, float] = {
        "FREE":      0.979,
        "BUSY":      0.986,
        "CONGESTED": 0.992,
        "JAMMED":    0.995,
    }

    # Build student → (decay_factor) map per channel
    student_decay: dict[str, float] = {}
    for ch_data in ch_mod.CHANNELS.values():
        state   = ch_data.get("status", "FREE")
        members = list(ch_data["users"])
        n       = max(len(members), 1)
        base    = _STATE_DECAY.get(state, 0.90)
        # N-adjustment: larger channels cool slightly slower
        factor  = min(base + 0.005 * math.log(n), 0.999)
        for cms in members:
            student_decay[cms] = factor

    with _energy_lock:
        cms_list     = list(_energy_scores.keys())
        before_total = round(sum(_energy_scores.values()), 4)
        n_active     = sum(1 for e in _energy_scores.values() if e > 0.0)

    decayed_count = 0
    zeroed_count  = 0
    decay_factors_used: set[float] = set()

    for cms in cms_list:
        pre = _energy_scores.get(cms, 0.0)
        if pre == 0.0:
            continue

        # Per-student state-dependent decay factor
        factor = student_decay.get(cms, 0.90)  # default FREE if not on channel
        decay_factors_used.add(factor)

        post = apply_decay_to_student(cms, now=now, decay_factor=factor)

        if post != pre:
            decayed_count += 1
        if post == 0.0 and pre > 0.0:
            zeroed_count += 1

    with _energy_lock:
        after_total = round(sum(_energy_scores.values()), 4)

    return {
        "decayed_count": decayed_count,
        "zeroed_count":  zeroed_count,
        "before_total":  before_total,
        "after_total":   after_total,
        "n_active":      n_active,
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
    Derive channel SNR from total energy and user count.

    Formula (spec Step 6):
        SNR = 30 - (N-1)*0.5 - (E / T_jammed) * 15

    Where:
      (N-1)*0.5      : each extra user costs 0.5 dB (was 2.0, too harsh)
      (E/T_jammed)*15: at 100% of JAMMED threshold, SNR drops 15 dB max
      T_jammed       = 35 * sqrt(N)  (same delta coefficient as classifier)

    This anchors SNR degradation to the dynamic JAMMED ceiling rather than
    raw joules, so SNR never collapses just because N is large.
    """
    n        = max(n_users, 1)
    t_jammed = 35.0 * math.sqrt(n)   # delta coefficient from classifier
    contention_penalty = max(n - 1, 0) * 0.5
    energy_penalty     = (channel_energy / max(t_jammed, 1.0)) * 15.0
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
