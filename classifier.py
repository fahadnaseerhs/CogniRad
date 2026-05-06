"""
classifier.py — CogniRad | Channel Health Classifier
=====================================================
Observation layer only.  Classifies current and projected channel states
using cumulative energy, never decides movement directly.

Phase 2: no direct decay logic here.  The classifier consumes decayed
energy snapshots from signal_physics.get_channel_energy_snapshot, which
applies idle decay before aggregation.  This keeps decay logic in a
single source of truth (signal_physics.py) and prevents duplication.

Public API
----------
classify_channel(channel_id)                          → ClassificationResult
classify_channel_projected(channel_id, extra_energy)  → ClassificationResult
is_healthy(result)                                    → bool

ClassificationResult is a TypedDict:
    status        : str     FREE | BUSY | CONGESTED | JAMMED
    confidence    : float   0.0–1.0
    total_energy  : float   (post-decay in Phase 2)
    snr_db        : float
    modulation    : str
    per_student   : list    sorted contributor breakdown
"""

from __future__ import annotations

import math
from typing import Any, TypedDict

import signal_physics as sp


# ---------------------------------------------------------------------------
# Dynamic Threshold Engine
# ---------------------------------------------------------------------------
# Thresholds scale with the number of active students on the channel.
# Design goal: works correctly for n = 1 to n = 50 students per channel.
#
# Formula:  threshold = coefficient × sqrt(N)
#
# Why sqrt(N)?
#   In real WiFi (CSMA/CA), the MAC layer handles collisions so effective
#   load grows sub-linearly with users. sqrt(N) is a standard approximation.
#
# Coefficients (tuned for realistic chat traffic with current PHY model):
#   α =  3  → FREE      ceiling
#   β = 10  → BUSY      ceiling
#   γ = 20  → CONGESTED ceiling
#   δ = 35  → JAMMED    ceiling
#
# Scaling table:
#   N= 1:  FREE<  3.0  BUSY< 10.0  CONGESTED< 20.0  JAMMED< 35.0
#   N= 5:  FREE<  6.7  BUSY< 22.4  CONGESTED< 44.7  JAMMED< 78.3
#   N=10:  FREE<  9.5  BUSY< 31.6  CONGESTED< 63.2  JAMMED<110.7
#   N=30:  FREE< 16.4  BUSY< 54.8  CONGESTED<109.5  JAMMED<191.7
#   N=50:  FREE< 21.2  BUSY< 70.7  CONGESTED<141.4  JAMMED<247.5

_ALPHA: float = 3.0    # FREE      coefficient
_BETA:  float = 10.0   # BUSY      coefficient
_GAMMA: float = 20.0   # CONGESTED coefficient
_DELTA: float = 35.0   # JAMMED    coefficient  (used in SNR model too)


def dynamic_thresholds(n: int) -> tuple[float, float, float, float]:
    """
    Compute (FREE_MAX, BUSY_MAX, CONGESTED_MAX, JAMMED_MAX) for *n* students.

    Formula: threshold = coefficient * sqrt(n)

    Returns a 4-tuple in Joules: (free_max, busy_max, congested_max, jammed_max).

    Parameters
    ----------
    n : int
        Number of students currently on the channel (0 treated as 1).
    """
    scale = math.sqrt(max(n, 1))
    return (
        round(_ALPHA * scale, 4),
        round(_BETA  * scale, 4),
        round(_GAMMA * scale, 4),
        round(_DELTA * scale, 4),
    )


# ---------------------------------------------------------------------------
# SNR thresholds  (physics-based, NOT load-based — stay static)
# ---------------------------------------------------------------------------
SNR_JAMMED_CEIL    = 5.0     # below this SNR → definitely JAMMED
SNR_CONGESTED_CEIL = 10.0    # below this SNR → CONGESTED

# Confidence calculation weights
W_ENERGY = 0.60
W_SNR    = 0.25
W_MOD    = 0.15


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class ClassificationResult(TypedDict):
    status: str
    confidence: float
    total_energy: float
    snr_db: float
    modulation: str
    modulation_index: int
    member_count: int
    per_student: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------

def _classify_snapshot(snapshot: dict[str, Any], admin_jammed: bool = False) -> ClassificationResult:
    """
    Internal: produce a classification from an energy snapshot dict.

    All 4 thresholds (FREE/BUSY/CONGESTED/JAMMED) are computed dynamically
    from member_count so the system self-calibrates for 1–50 students.
    SNR is normalised against the dynamic JAMMED ceiling, not raw joules.
    """
    total   = snapshot["total_energy"]
    snr     = snapshot["snr_db"]
    mod_idx = snapshot["modulation_index"]
    n       = snapshot["member_count"]

    # ── Dynamic thresholds (all 4 bands scale with sqrt(N)) ──────
    free_max, busy_max, congested_max, jammed_max = dynamic_thresholds(n)

    # ── Energy ratio (normalised to JAMMED ceiling for confidence) 
    energy_ratio = min(total / max(jammed_max, 1.0), 1.0)

    # ── SNR penalty ──────────────────────────────────────────────
    snr_penalty = max((15.0 - snr) / 20.0, 0.0)

    # ── Modulation penalty ───────────────────────────────────────
    mod_penalty = max((4.0 - mod_idx) / 4.0, 0.0)

    # ── Composite confidence score ───────────────────────────────
    score = min(
        W_ENERGY * energy_ratio + W_SNR * snr_penalty + W_MOD * mod_penalty,
        1.0,
    )

    # ── Status determination ─────────────────────────────────────
    if admin_jammed:
        status = "JAMMED"
        score  = max(score, 0.85)
    elif total > jammed_max or snr < SNR_JAMMED_CEIL:
        status = "JAMMED"
    elif total > congested_max or snr < SNR_CONGESTED_CEIL:
        status = "CONGESTED"
    elif total > busy_max:
        status = "BUSY"
    elif total > free_max:
        status = "BUSY"   # same label — free_max is soft lower bound
    else:
        status = "FREE"

    return ClassificationResult(
        status=status,
        confidence=round(score, 3),
        total_energy=snapshot["total_energy"],
        snr_db=snapshot["snr_db"],
        modulation=snapshot["modulation"],
        modulation_index=snapshot["modulation_index"],
        member_count=n,
        per_student=snapshot["per_student"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_channel(channel_id: str, *, admin_jammed: bool = False, now: float | None = None) -> ClassificationResult:
    """
    Classify the *current real state* of a channel using cumulative energy.

    This is the primary observation function.  Call it from the AI loop
    and after every message send to decide channel health.

    Phase 2: get_channel_energy_snapshot now applies idle decay before
    aggregating per-student totals, so this function automatically
    classifies based on decayed (time-accurate) energy without any extra
    work here.  Decay logic stays in signal_physics.py only.

    Snapshot consistency fix: accepts a shared `now` so that all member
    decays within the snapshot use the same timestamp.  Pass the same
    `now` to every classify call in one control cycle.
    """
    snapshot = sp.get_channel_energy_snapshot(channel_id, now=now)
    return _classify_snapshot(snapshot, admin_jammed=admin_jammed)


def classify_channel_projected(
    channel_id: str,
    additional_energy: float,
    now: float | None = None,
) -> ClassificationResult:
    """
    Classify what *would* happen if *additional_energy* were added to
    *channel_id* without actually mutating state.

    Used by the allocator to test whether a destination can absorb a
    student before committing a move.

    Phase 2: project_channel_energy now reads decayed per-student scores
    via get_energy_score (decay-aware), so the projected total starts from
    the current decayed baseline and then adds the incoming energy.
    Formula: projected_total = decayed_current_total + additional_energy
    No decay logic is duplicated here — it all lives in signal_physics.py.

    Snapshot consistency fix: accepts a shared `now` so that the
    projection uses the same decay baseline as the live snapshot taken
    earlier in the same allocator decision cycle.  This prevents a
    borderline healthy/unhealthy decision from flipping just because a
    few milliseconds passed between the source ranking read and the
    destination projection read.
    """
    import channels as ch_mod

    members = list(ch_mod.CHANNELS[channel_id]["users"])
    # Pass the shared now so projection and live classification share the
    # same decay baseline within one control cycle.
    projected_total = sp.project_channel_energy(channel_id, additional_energy, now=now)
    n_users = len(members) + 1  # +1 because the student hasn't arrived yet

    snr = sp.derive_snr(projected_total, n_users)
    mod_name, mod_idx = sp.derive_modulation(snr)

    # Build a synthetic snapshot for classification
    snapshot: dict[str, Any] = {
        "channel_id": channel_id,
        "total_energy": projected_total,
        "member_count": n_users,
        "per_student": [],  # breakdown not meaningful for projection
        "snr_db": round(snr, 3),
        "modulation": mod_name,
        "modulation_index": mod_idx,
    }

    return _classify_snapshot(snapshot)


def is_healthy(result: ClassificationResult) -> bool:
    """Return True if the classification indicates the channel is safe."""
    return result["status"] in {"FREE", "BUSY"}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== classifier.py self-test ===\n")

    # Prep: put some users on CH-1 with energy
    import channels as ch_mod

    ch_mod.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    sp.set_energy_score("CMS001", 0.5)
    sp.set_energy_score("CMS002", 0.3)

    result = classify_channel("CH-1")
    print(f"  CH-1: status={result['status']}, confidence={result['confidence']}, "
          f"energy={result['total_energy']}, snr={result['snr_db']}")
    assert result["status"] == "FREE"

    # Simulate high energy → should escalate
    sp.set_energy_score("CMS001", 10.0)
    sp.set_energy_score("CMS002", 8.0)
    result2 = classify_channel("CH-1")
    print(f"  CH-1 high-energy: status={result2['status']}, energy={result2['total_energy']}")
    assert result2["status"] in {"CONGESTED", "JAMMED"}

    # Projected test
    sp.set_energy_score("CMS001", 1.0)
    sp.set_energy_score("CMS002", 1.0)
    proj = classify_channel_projected("CH-1", additional_energy=20.0)
    print(f"  CH-1 projected +20J: status={proj['status']}")
    assert proj["status"] in {"CONGESTED", "JAMMED"}

    # Clean up
    ch_mod.CHANNELS["CH-1"]["users"] = []
    sp.reset_energy_score("CMS001")
    sp.reset_energy_score("CMS002")

    print("\n[PASS] All classifier checks complete.\n")
