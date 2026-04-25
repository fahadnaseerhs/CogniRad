"""
channels.py — CogniRad | Channel Registry
=========================================
Defines the 5 IEEE 802.11 Wi-Fi channels used by the CogniRad spectrum
management system.  A channel is a **shared frequency band**, not a group
chat room.  Multiple students can occupy the same channel; they communicate
via point-to-point DMs over that band.  The AI watches cumulative band
energy and reallocates students when a channel becomes overloaded.

Developer Guide
---------------
WHAT THIS FILE IS:
    Single source of truth for all channel definitions.  Every other module
    (allocator.py, classifier.py, signal_physics.py, main.py) imports from
    here.  It holds the in-memory channel state; it never connects to a
    database directly.

HOW THE DATA IS STRUCTURED:
    CHANNELS is a plain Python dict.  Key = channel ID string (e.g. "CH-1").
    Value = dict with these fields:

    Field                  Type        Description
    ---------------------  ----------  -----------------------------------
    frequency              str         Human-readable GHz label shown in UI
    users                  list[str]   CMS IDs currently on this frequency
    message_rate           int         DMs routed in the last 60 seconds
    status                 str         FREE | BUSY | CONGESTED | JAMMED
    congestion_threshold   int         Max users before status → CONGESTED
    message_rate_threshold int         Max msg/min before status → CONGESTED

WHY THESE EXACT FREQUENCIES:
    The five channels map to real non-overlapping IEEE 802.11 allocations:
      2.4 GHz band  →  CH-1 (1), CH-2 (6), CH-3 (11)
      5 GHz band    →  CH-36 and CH-48
    We label them CH-1 through CH-5 internally for simplicity.

STATUS LIFECYCLE:
    FREE       → low total energy, few users
    BUSY       → energy present but within healthy limits
    CONGESTED  → cumulative energy exceeds safe threshold
    JAMMED     → admin-forced or extreme energy overload;
                 all students are reallocated automatically

MODULE INTERACTIONS:
    allocator.py      — calls get_least_loaded_channel() for new joins;
                        mutates channel["users"] and channel["status"]
    classifier.py     — reads members to compute cumulative energy and
                        classify channel health
    signal_physics.py — reads channel frequency for band-specific energy
                        calculations
    main.py           — calls get_channel_status() for the REST API;
                        forces status="JAMMED" on admin command
"""

from __future__ import annotations

import math
import random

# ---------------------------------------------------------------------------
# PHY + classifier tuning constants
# ---------------------------------------------------------------------------

MODULATION_BY_SNR = (
    (25.0, "64-QAM", 6),
    (15.0, "16-QAM", 4),
    (8.0, "QPSK", 2),
    (-math.inf, "BPSK", 1),
)

MODULATION_INDEX = {"BPSK": 1, "QPSK": 2, "16-QAM": 4, "64-QAM": 6}

SNR_CLEAN_DB = 30.0
SNR_FLOOR_DB = 2.0
SNR_JAMMED_FLOOR_DB = -5.0

FREEZE_THRESHOLD = 0.95
JAMMED_THRESHOLD = 0.70
BUSY_THRESHOLD = 0.45

MAX_LATENCY_MS = 2500
JAM_WINDOW_LIMIT = 4.0
JAM_WINDOW_DECAY = 0.85


def _band_profile(frequency: str) -> dict[str, float | str]:
    """
    Return the regulatory and baseline PHY profile for a channel frequency.
    """
    if frequency.startswith("2.4"):
        return {
            "band": "2.4 GHz",
            "eirp_max_dbm": 20.0,
            "base_energy": 1.00,
            "energy_per_symbol": 0.030,
        }

    return {
        "band": "5 GHz",
        "eirp_max_dbm": 17.0,
        "base_energy": 0.85,
        "energy_per_symbol": 0.024,
    }


# ---------------------------------------------------------------------------
# Channel registry
# ---------------------------------------------------------------------------

CHANNELS: dict[str, dict] = {
    "CH-1": {
        "frequency": "2.412 GHz",
        "users": [],
        "message_rate": 0,
        "status": "FREE",
        "congestion_threshold": 8,
        "message_rate_threshold": 20,
        "rolling_jammed_score": 0.0,
        "last_signal": {},
        "transmit_frozen": False,
    },
    "CH-2": {
        "frequency": "2.437 GHz",
        "users": [],
        "message_rate": 0,
        "status": "FREE",
        "congestion_threshold": 8,
        "message_rate_threshold": 20,
        "rolling_jammed_score": 0.0,
        "last_signal": {},
        "transmit_frozen": False,
    },
    "CH-3": {
        "frequency": "2.462 GHz",
        "users": [],
        "message_rate": 0,
        "status": "FREE",
        "congestion_threshold": 8,
        "message_rate_threshold": 20,
        "rolling_jammed_score": 0.0,
        "last_signal": {},
        "transmit_frozen": False,
    },
    "CH-4": {
        "frequency": "5.180 GHz",
        "users": [],
        "message_rate": 0,
        "status": "FREE",
        "congestion_threshold": 8,
        "message_rate_threshold": 20,
        "rolling_jammed_score": 0.0,
        "last_signal": {},
        "transmit_frozen": False,
    },
    "CH-5": {
        "frequency": "5.240 GHz",
        "users": [],
        "message_rate": 0,
        "status": "FREE",
        "congestion_threshold": 8,
        "message_rate_threshold": 20,
        "rolling_jammed_score": 0.0,
        "last_signal": {},
        "transmit_frozen": False,
    },
}

# ---------------------------------------------------------------------------
# Helper: get the channel with the fewest active users
# ---------------------------------------------------------------------------

def get_least_loaded_channel() -> dict:
    """
    Return the channel dict (including its ID) that has the fewest users.

    Behaviour:
      - Skips channels whose status is JAMMED (they accept no new users).
      - Among remaining channels, picks the one with min(len(users)).
      - If multiple channels are tied, the one that appears first in the
        CHANNELS dict (i.e. CH-1 before CH-2, etc.) wins — deterministic.
      - If ALL channels are JAMMED, raises RuntimeError so the caller can
        surface a meaningful error to the student trying to join.

    Returns:
        dict with keys: "channel_id" + all fields from CHANNELS[channel_id]

    Example:
        >>> ch = get_least_loaded_channel()
        >>> ch["channel_id"]
        'CH-1'
        >>> ch["frequency"]
        '2.412 GHz'
    """
    available = {
        ch_id: ch_data
        for ch_id, ch_data in CHANNELS.items()
        if ch_data["status"] != "JAMMED"
    }

    if not available:
        raise RuntimeError(
            "All channels are currently JAMMED. "
            "No channel is available for assignment."
        )

    least_loaded_id = min(available, key=lambda ch_id: len(available[ch_id]["users"]))

    return {"channel_id": least_loaded_id, **CHANNELS[least_loaded_id]}


# ---------------------------------------------------------------------------
# Helper: get full info for one channel by ID
# ---------------------------------------------------------------------------

def get_channel_status(channel_id: str) -> dict:
    """
    Return a copy of the full state dict for a single channel.

    Args:
        channel_id: e.g. "CH-1", "CH-3". Case-sensitive.

    Returns:
        dict with keys: channel_id, frequency, users, message_rate,
        status, congestion_threshold, message_rate_threshold,
        user_count (convenience field added here).

    Raises:
        KeyError: if channel_id does not exist in CHANNELS.

    Example:
        >>> info = get_channel_status("CH-4")
        >>> info["frequency"]
        '5.180 GHz'
        >>> info["user_count"]
        0
    """
    if channel_id not in CHANNELS:
        raise KeyError(
            f"Channel '{channel_id}' does not exist. "
            f"Valid IDs are: {list(CHANNELS.keys())}"
        )

    ch = CHANNELS[channel_id]

    return {
        "channel_id": channel_id,
        "frequency": ch["frequency"],
        "users": list(ch["users"]),           # return a copy, not a reference
        "message_rate": ch["message_rate"],
        "status": ch["status"],
        "congestion_threshold": ch["congestion_threshold"],
        "message_rate_threshold": ch["message_rate_threshold"],
        "user_count": len(ch["users"]),        # convenience field for the UI
        "transmit_frozen": ch.get("transmit_frozen", False),
        "rolling_jammed_score": ch.get("rolling_jammed_score", 0.0),
        "last_signal": dict(ch.get("last_signal", {})),
    }


# ---------------------------------------------------------------------------
# PHY feature synthesis + local status inference
# ---------------------------------------------------------------------------

def _select_modulation(snr_db: float) -> tuple[str, int]:
    for min_snr, name, bits in MODULATION_BY_SNR:
        if snr_db >= min_snr:
            return name, bits
    return "BPSK", 1


def build_signal_features(
    channel_id: str,
    message_text: str,
    *,
    concurrent_transmitters: int = 1,
    admin_jammed: bool = False,
) -> dict:
    """
    Build a per-message synthetic PHY descriptor: [energy, modulation, snr].
    """
    if channel_id not in CHANNELS:
        raise KeyError(f"Unknown channel: {channel_id}")

    ch = CHANNELS[channel_id]
    profile = _band_profile(ch["frequency"])
    char_count = max(len(message_text), 1)
    bits = char_count * 8
    contention = max(len(ch["users"]) - 1, 0) + max(ch["message_rate"] / 15.0, 0.0)

    resting_snr = max(SNR_CLEAN_DB - (2.0 * contention), SNR_FLOOR_DB)
    modulation, bits_per_symbol = _select_modulation(resting_snr)

    symbol_count = math.ceil(bits / bits_per_symbol)
    tx_scale = max(1, concurrent_transmitters)
    raw_energy = (
        profile["base_energy"]
        * profile["energy_per_symbol"]
        * symbol_count
        * tx_scale
    )
    eirp_limit = profile["eirp_max_dbm"] * tx_scale
    energy = min(raw_energy, eirp_limit)

    snr_drop = 0.06 * symbol_count
    effective_snr = max(resting_snr - snr_drop, SNR_FLOOR_DB)
    if admin_jammed or ch["status"] == "JAMMED":
        effective_snr = min(effective_snr, SNR_JAMMED_FLOOR_DB)

    adjusted_modulation, adjusted_bits = _select_modulation(effective_snr)
    if adjusted_bits != bits_per_symbol:
        symbol_count = math.ceil(bits / adjusted_bits)
        raw_energy = (
            profile["base_energy"]
            * profile["energy_per_symbol"]
            * symbol_count
            * tx_scale
        )
        energy = min(raw_energy, eirp_limit)
        modulation = adjusted_modulation
        bits_per_symbol = adjusted_bits

    return {
        "channel_id": channel_id,
        "band": profile["band"],
        "energy": round(energy, 3),
        "snr_db": round(effective_snr, 3),
        "modulation": modulation,
        "modulation_index": MODULATION_INDEX[modulation],
        "symbol_count": symbol_count,
        "char_count": char_count,
        "eirp_max_dbm": profile["eirp_max_dbm"],
    }


def classify_signal_features(features: dict) -> dict:
    """
    Convert signal features into FREE/BUSY/CONGESTED/JAMMED + confidence.
    """
    energy = features["energy"]
    snr_db = features["snr_db"]
    mod_idx = features["modulation_index"]
    eirp_max = max(features["eirp_max_dbm"], 1.0)

    energy_ratio = min(energy / eirp_max, 1.0)
    snr_penalty = max((15.0 - snr_db) / 20.0, 0.0)
    mod_penalty = max((4.0 - mod_idx) / 4.0, 0.0)

    jam_score = min((0.65 * energy_ratio) + (0.25 * snr_penalty) + (0.10 * mod_penalty), 1.0)

    if jam_score >= JAMMED_THRESHOLD:
        status = "JAMMED"
    elif jam_score >= BUSY_THRESHOLD:
        status = "CONGESTED"
    elif energy_ratio > 0.08:
        status = "BUSY"
    else:
        status = "FREE"

    return {"status": status, "confidence": round(jam_score, 3)}


def evaluate_message_transmission(
    channel_id: str,
    message_text: str,
    *,
    concurrent_transmitters: int = 1,
    admin_jammed: bool = False,
) -> dict:
    """
    End-to-end message admission decision using PHY-derived features.
    """
    features = build_signal_features(
        channel_id,
        message_text,
        concurrent_transmitters=concurrent_transmitters,
        admin_jammed=admin_jammed,
    )
    prediction = classify_signal_features(features)
    ch = CHANNELS[channel_id]

    confidence = prediction["confidence"]
    status = prediction["status"]
    hard_freeze = status == "JAMMED" and confidence >= FREEZE_THRESHOLD

    if hard_freeze:
        drop_probability = 1.0
    elif status == "JAMMED":
        drop_probability = min(0.2 + ((confidence - JAMMED_THRESHOLD) / 0.25) * 0.8, 0.95)
    elif status == "CONGESTED":
        drop_probability = min(0.1 + confidence * 0.4, 0.6)
    else:
        drop_probability = 0.0

    latency_ms = int(MAX_LATENCY_MS * min(confidence, 1.0))
    accepted = random.random() >= drop_probability
    if hard_freeze:
        accepted = False

    # Rolling persistence signal for escalation logic in the control plane.
    ch["rolling_jammed_score"] = (ch["rolling_jammed_score"] * JAM_WINDOW_DECAY) + (
        1.0 if status == "JAMMED" else 0.0
    )
    ch["transmit_frozen"] = hard_freeze
    ch["status"] = status
    ch["last_signal"] = {**features, **prediction}

    return {
        "accepted": accepted,
        "drop_probability": round(drop_probability, 3),
        "latency_ms": latency_ms,
        "status": status,
        "confidence": confidence,
        "freeze_sender": hard_freeze,
        "persistent_jammed": ch["rolling_jammed_score"] >= JAM_WINDOW_LIMIT,
        "error": (
            "Channel saturated — Spectrum AI managing reallocation."
            if hard_freeze
            else None
        ),
        "features": features,
    }


def refresh_channel_status(channel_id: str) -> None:
    """
    Backward-compatible status refresh.
    If recent PHY classification exists, honor it. Otherwise, use a lightweight
    fallback based on user presence to keep older call sites safe.
    """
    ch = CHANNELS[channel_id]

    if ch["status"] == "JAMMED":
        return  # admin-controlled; do not auto-override

    last_signal = ch.get("last_signal", {})
    if last_signal:
        ch["status"] = last_signal.get("status", ch["status"])
        ch["transmit_frozen"] = (
            last_signal.get("status") == "JAMMED"
            and last_signal.get("confidence", 0.0) >= FREEZE_THRESHOLD
        )
        return

    if len(ch["users"]) > 0:
        ch["status"] = "BUSY"
    else:
        ch["status"] = "FREE"


# ---------------------------------------------------------------------------
# DM-aware membership helpers
# ---------------------------------------------------------------------------

def get_channel_members(channel_id: str) -> list[str]:
    """
    Return a copy of the CMS list for students on *channel_id*.
    """
    if channel_id not in CHANNELS:
        raise KeyError(f"Channel '{channel_id}' does not exist.")
    return list(CHANNELS[channel_id]["users"])


def are_on_same_channel(cms_a: str, cms_b: str) -> str | None:
    """
    If *cms_a* and *cms_b* share a channel, return that channel key.
    Otherwise return None.
    """
    for ch_id, ch_data in CHANNELS.items():
        users = ch_data["users"]
        if cms_a in users and cms_b in users:
            return ch_id
    return None


def find_student_channel(cms: str) -> str | None:
    """
    Return the channel key that *cms* is currently on, or None.
    """
    for ch_id, ch_data in CHANNELS.items():
        if cms in ch_data["users"]:
            return ch_id
    return None


# ---------------------------------------------------------------------------
# Self-test — run: python channels.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import copy, pprint

    PASS = "\033[92m[PASS]\033[0m"
    FAIL = "\033[91m[FAIL]\033[0m"
    HEAD = "\033[96m"
    END  = "\033[0m"

    errors = 0

    def check(condition: bool, label: str):
        global errors
        if condition:
            print(f"  {PASS} {label}")
        else:
            print(f"  {FAIL} {label}")
            errors += 1

    # ── 1. Initial state ────────────────────────────────────────────────
    print(f"\n{HEAD}=== 1. Initial channel registry ==={END}")
    pprint.pprint(CHANNELS, width=60)

    check(len(CHANNELS) == 5, "Exactly 5 channels defined")
    check(CHANNELS["CH-1"]["frequency"] == "2.412 GHz", "CH-1 frequency correct")
    check(CHANNELS["CH-2"]["frequency"] == "2.437 GHz", "CH-2 frequency correct")
    check(CHANNELS["CH-3"]["frequency"] == "2.462 GHz", "CH-3 frequency correct")
    check(CHANNELS["CH-4"]["frequency"] == "5.180 GHz", "CH-4 frequency correct")
    check(CHANNELS["CH-5"]["frequency"] == "5.240 GHz", "CH-5 frequency correct")

    for ch_id, ch in CHANNELS.items():
        check(ch["status"] == "FREE",                 f"{ch_id} starts FREE")
        check(ch["congestion_threshold"] == 8,        f"{ch_id} congestion_threshold = 8")
        check(ch["message_rate_threshold"] == 20,     f"{ch_id} message_rate_threshold = 20")
        check(ch["users"] == [],                      f"{ch_id} starts with empty user list")

    # ── 2. get_least_loaded_channel — all empty ──────────────────────────
    print(f"\n{HEAD}=== 2. get_least_loaded_channel (all empty) ==={END}")
    least = get_least_loaded_channel()
    print(f"  Returned: {least['channel_id']} ({least['frequency']})")
    check(least["channel_id"] == "CH-1", "Returns CH-1 when all channels tied (first wins)")

    # ── 3. get_least_loaded_channel — after adding users to CH-1 & CH-2 ─
    print(f"\n{HEAD}=== 3. get_least_loaded_channel (CH-1 has 3 users, CH-2 has 1) ==={END}")
    CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
    CHANNELS["CH-2"]["users"] = ["CMS004"]
    least = get_least_loaded_channel()
    print(f"  Returned: {least['channel_id']} ({least['frequency']})")
    check(least["channel_id"] == "CH-3", "Returns CH-3 (0 users) when CH-1 and CH-2 are busier")

    # ── 4. refresh_channel_status fallback ───────────────────────────────
    print(f"\n{HEAD}=== 4. refresh_channel_status (legacy fallback) ==={END}")
    refresh_channel_status("CH-1")
    check(CHANNELS["CH-1"]["status"] == "BUSY", "CH-1 becomes BUSY with 3 users")

    # Legacy fallback does not infer congestion from headcount anymore.
    CHANNELS["CH-3"]["users"] = [f"CMS10{i}" for i in range(8)]
    refresh_channel_status("CH-3")
    check(CHANNELS["CH-3"]["status"] == "BUSY", "CH-3 remains BUSY without PHY classification")

    # Evaluate a short transmission and ensure channel stays non-jammed.
    short_tx = evaluate_message_transmission("CH-4", "ok")
    check(short_tx["status"] in {"FREE", "BUSY"}, "Short transmission remains FREE/BUSY")

    # Evaluate a long transmission and ensure classifier can hit congested/jammed.
    long_tx = evaluate_message_transmission("CH-4", "x" * 600)
    check(long_tx["status"] in {"CONGESTED", "JAMMED"}, "Long transmission escalates congestion")

    # JAMMED should not be overridden
    CHANNELS["CH-5"]["status"] = "JAMMED"
    CHANNELS["CH-5"]["users"] = []
    refresh_channel_status("CH-5")
    check(CHANNELS["CH-5"]["status"] == "JAMMED", "JAMMED status is never auto-overridden")

    # ── 5. get_least_loaded_channel — skips JAMMED ───────────────────────
    print(f"\n{HEAD}=== 5. get_least_loaded_channel skips JAMMED channel ==={END}")
    # CH-5 is jammed; CH-4 has 0 users and may be CONGESTED but not jammed.
    # CH-2 has 1 user; CH-1 has 3; CH-3 has 8; CH-4 has 0 users
    least = get_least_loaded_channel()
    print(f"  Returned: {least['channel_id']} (CH-5 is JAMMED, should skip it)")
    check(least["channel_id"] != "CH-5", "JAMMED channels are skipped for new assignment")

    # ── 6. get_channel_status ────────────────────────────────────────────
    print(f"\n{HEAD}=== 6. get_channel_status ==={END}")
    for ch_id in CHANNELS:
        info = get_channel_status(ch_id)
        print(f"  {ch_id}: status={info['status']}, users={info['user_count']}, freq={info['frequency']}")
        check("channel_id" in info and info["channel_id"] == ch_id, f"{ch_id} channel_id field correct")
        check("user_count" in info,  f"{ch_id} has convenience user_count field")

    # Unknown channel
    print(f"\n{HEAD}=== 7. get_channel_status — unknown ID ==={END}")
    try:
        get_channel_status("CH-99")
        check(False, "Should have raised KeyError for CH-99")
    except KeyError as e:
        print(f"  Caught KeyError: {e}")
        check(True, "KeyError raised correctly for unknown channel ID")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    if errors == 0:
        print(f"{PASS} All checks complete. channels.py is working correctly.\n")
    else:
        print(f"{FAIL} {errors} check(s) failed. Review output above.\n")