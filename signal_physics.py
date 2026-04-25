"""
signal_physics.py — CogniRad | Energy Accumulation & RF Physics
================================================================
Source of truth for per-student cumulative energy scores and channel
energy snapshots.  Separates radio/energy math from routing code.

Public API
----------
compute_message_energy(text, channel_id, ...)  → float
update_energy_score(cms, message_energy)        → float   (new total)
get_energy_score(cms)                           → float
reset_energy_score(cms)                         → None
get_channel_energy_snapshot(channel_id)          → dict
decay_energy_on_reallocation(cms, factor)       → float   (decayed total)
derive_snr(channel_energy, n_users)             → float
derive_modulation(snr_db)                       → tuple[str, int]
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


def get_energy_score(cms: str) -> float:
    """Return current cumulative energy for *cms* (0.0 if unknown)."""
    with _energy_lock:
        return _energy_scores.get(cms, 0.0)


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
# 3. get_channel_energy_snapshot
# ---------------------------------------------------------------------------

def get_channel_energy_snapshot(channel_id: str) -> dict[str, Any]:
    """
    Return a dict describing the total energy state of a channel.

    Keys
    ----
    channel_id          str
    total_energy        float     sum of all member scores
    member_count        int
    per_student         list[dict]  sorted highest-energy-first
                            each: {cms, energy, pct}
    snr_db              float     derived SNR
    modulation          str       current modulation label
    modulation_index    int       bits per symbol
    """
    import channels as ch_mod

    members: list[str] = list(ch_mod.CHANNELS[channel_id]["users"])

    with _energy_lock:
        per_student = [
            {"cms": cms, "energy": _energy_scores.get(cms, 0.0)}
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

def project_channel_energy(channel_id: str, additional_energy: float = 0.0) -> float:
    """
    Return what the total energy of *channel_id* would be if
    *additional_energy* were added (without actually adding it).
    """
    import channels as ch_mod
    members = list(ch_mod.CHANNELS[channel_id]["users"])
    with _energy_lock:
        total = sum(_energy_scores.get(cms, 0.0) for cms in members)
    return round(total + additional_energy, 4)


def project_channel_energy_without(channel_id: str, cms_to_remove: str) -> float:
    """
    Return what the total energy of *channel_id* would be if *cms_to_remove*
    were taken off it (without actually removing them).
    """
    import channels as ch_mod
    members = list(ch_mod.CHANNELS[channel_id]["users"])
    with _energy_lock:
        total = sum(
            _energy_scores.get(cms, 0.0)
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

    # Decay
    decayed = decay_energy_on_reallocation("CMS001", factor=0.5)
    assert abs(decayed - 1.75) < 0.001
    print(f"  CMS001 after 50% decay: {decayed}")

    # SNR
    snr = derive_snr(5.0, 3)
    mod, idx = derive_modulation(snr)
    print(f"  SNR at 5J/3 users: {snr:.1f} dB -> {mod} ({idx} bps)")

    # Reset
    reset_energy_score("CMS001")
    assert get_energy_score("CMS001") == 0.0

    print("\n[PASS] All signal_physics checks complete.\n")
