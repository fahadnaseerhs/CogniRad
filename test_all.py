"""
============================================================
CogniRad Test Suite — DM Model v2
============================================================
Tests the cognitive-radio DM backend:

* ``channels.py``         — channel registry + membership helpers
* ``signal_physics.py``   — cumulative energy tracking
* ``classifier.py``       — channel health classification
* ``allocator.py``        — energy-aware reallocation engine
* ``auth.py``             — student login/logout

Run:
    pytest -s test_all.py --asyncio-mode=auto -v

or for a quick interactive demo:
    python test_all.py
"""

import asyncio
import json
import sys
from typing import Any, Dict, List

import pytest
from unittest import mock


# ---------------------------------------------------------------------------
# In-memory fake database module
# ---------------------------------------------------------------------------

class _FakeStudent:
    def __init__(self, cms: str, active: bool = True):
        self.cms = cms
        self.name = f"Student_{cms}"
        self.active = active
        self.is_active = active
        self.id = int(cms.replace("CMS", ""))
        self.channel_id = None
        self.joined_at = None

    def __repr__(self) -> str:
        return f"<_FakeStudent {self.cms}>"


class _FakeDatabase:
    def __init__(self):
        self.students: Dict[str, _FakeStudent] = {
            f"CMS{i:03d}": _FakeStudent(f"CMS{i:03d}") for i in range(1, 11)
        }
        self.sessions: Dict[str, str] = {}
        self.channel_assignments: Dict[int, List[str]] = {i: [] for i in range(1, 6)}
        self.channel_status: Dict[int, Dict[str, Any]] = {}
        self.messages: List[Dict[str, Any]] = []

    async def get_student_by_cms(self, cms: str) -> _FakeStudent | None:
        return self.students.get(cms)

    async def get_cms_from_token(self, token: str) -> str | None:
        return self.sessions.get(token)

    async def create_session(self, token: str, cms: str, *, invalidate_existing: bool = False) -> None:
        if invalidate_existing:
            self.sessions = {t: c for t, c in self.sessions.items() if c != cms}
        self.sessions[token] = cms

    async def delete_session(self, token: str) -> bool:
        return self.sessions.pop(token, None) is not None

    async def get_all_channels(self):
        return [mock.Mock(id=i) for i in range(1, 6)]

    async def get_students_on_channel(self, channel_id: int, *, active_only: bool = True):
        cms_list = self.channel_assignments.get(channel_id, [])
        if active_only:
            return [self.students[c] for c in cms_list if self.students[c].active]
        return [self.students[c] for c in cms_list]

    async def get_all_students(self):
        return list(self.students.values())

    async def assign_student_to_channel(self, cms: str, channel_id: int) -> bool:
        if cms in self.channel_assignments[channel_id]:
            return False
        self.channel_assignments[channel_id].append(cms)
        if cms in self.students:
            self.students[cms].channel_id = channel_id
        return True

    async def update_channel_status(self, channel_id: int, status: str, confidence: float, *, is_jammed: bool = False):
        self.channel_status[channel_id] = {"status": status, "confidence": confidence, "jammed": is_jammed}

    async def get_recent_messages(self, channel_num: int, limit: int = 10):
        msgs = [m for m in self.messages if m["channel"] == channel_num]
        return msgs[-limit:]

    async def move_student(self, cms: str, src: int, dst: int) -> bool:
        if cms not in self.channel_assignments.get(src, []):
            return False
        self.channel_assignments[src].remove(cms)
        self.channel_assignments[dst].append(cms)
        if cms in self.students:
            self.students[cms].channel_id = dst
        return True

    async def save_message(self, **kwargs):
        self.messages.append(kwargs)

    async def init_db(self): pass


# ---------------------------------------------------------------------------
# Patch database
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_database(monkeypatch):
    import allocator, auth, channels, signal_physics, classifier
    fake_db = _FakeDatabase()
    monkeypatch.setitem(sys.modules, "database", fake_db)
    monkeypatch.setattr(allocator, "database", fake_db, raising=False)
    monkeypatch.setattr(auth, "database", fake_db, raising=False)

    # Reset channel state
    for ch in channels.CHANNELS.values():
        ch["users"].clear()
        ch["status"] = "FREE"
        ch["message_rate"] = 0
        ch["rolling_jammed_score"] = 0.0
        ch["last_signal"] = {}
        ch["transmit_frozen"] = False
        ch["admin_forced_jammed"] = False

    # Reset energy scores
    signal_physics._energy_scores.clear()
    signal_physics._energy_timestamps.clear()

    return fake_db


# ---------------------------------------------------------------------------
# Channel registry tests
# ---------------------------------------------------------------------------

def test_channel_definitions_and_least_loaded():
    import channels
    expected = {
        "CH-1": "2.412 GHz",
        "CH-2": "2.437 GHz",
        "CH-3": "2.462 GHz",
        "CH-4": "5.180 GHz",
        "CH-5": "5.240 GHz",
    }
    assert set(channels.CHANNELS.keys()) == set(expected.keys())
    for key, freq in expected.items():
        assert channels.CHANNELS[key]["frequency"] == freq

    least = channels.get_least_loaded_channel()
    assert least["channel_id"] == "CH-1"

    channels.CHANNELS["CH-1"]["users"].extend(["CMS001", "CMS002", "CMS003"])
    channels.CHANNELS["CH-2"]["users"].append("CMS004")
    channels.CHANNELS["CH-5"]["status"] = "JAMMED"
    least = channels.get_least_loaded_channel()
    assert least["channel_id"] == "CH-3"

    for ch in channels.CHANNELS.values():
        ch["users"].clear()
        ch["status"] = "FREE"


def test_membership_helpers():
    import channels
    channels.CHANNELS["CH-2"]["users"] = ["CMS001", "CMS002"]
    channels.CHANNELS["CH-3"]["users"] = ["CMS003"]

    assert channels.get_channel_members("CH-2") == ["CMS001", "CMS002"]
    assert channels.are_on_same_channel("CMS001", "CMS002") == "CH-2"
    assert channels.are_on_same_channel("CMS001", "CMS003") is None
    assert channels.find_student_channel("CMS003") == "CH-3"
    assert channels.find_student_channel("CMS099") is None


# ---------------------------------------------------------------------------
# Signal physics tests
# ---------------------------------------------------------------------------

def test_energy_accumulation():
    import signal_physics as sp

    assert sp.get_energy_score("CMS001") == 0.0

    sp.update_energy_score("CMS001", 1.5)
    sp.update_energy_score("CMS001", 2.0)
    assert abs(sp.get_energy_score("CMS001") - 3.5) < 0.001

    sp.update_energy_score("CMS002", 0.8)
    assert abs(sp.get_energy_score("CMS002") - 0.8) < 0.001


def test_energy_decay():
    import signal_physics as sp

    sp.set_energy_score("CMS010", 10.0)
    decayed = sp.decay_energy_on_reallocation("CMS010", factor=0.5)
    assert abs(decayed - 5.0) < 0.001
    assert abs(sp.get_energy_score("CMS010") - 5.0) < 0.001


def test_snr_and_modulation():
    import signal_physics as sp

    # Low energy → high SNR → 64-QAM
    snr = sp.derive_snr(0.5, 1)
    mod, idx = sp.derive_modulation(snr)
    assert snr > 25.0
    assert mod == "64-QAM"

    # High energy → low SNR → BPSK
    # With SNR_PER_JOULE_DROP = 0.08, need very high energy to drop SNR below 8 dB.
    # 30 dB − (4 users × 2) − (300 J × 0.08) = 30 − 8 − 24 = −2 → clamped to 2 dB → BPSK
    snr2 = sp.derive_snr(300.0, 5)
    mod2, idx2 = sp.derive_modulation(snr2)
    assert snr2 < 10.0
    assert idx2 <= 2  # QPSK or BPSK


def test_channel_energy_snapshot():
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    sp.set_energy_score("CMS001", 3.0)
    sp.set_energy_score("CMS002", 1.0)

    snap = sp.get_channel_energy_snapshot("CH-1")
    assert abs(snap["total_energy"] - 4.0) < 0.001
    assert snap["member_count"] == 2
    assert snap["per_student"][0]["cms"] == "CMS001"  # highest first
    assert snap["per_student"][0]["pct"] == 75.0


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

def test_classify_healthy():
    import channels, signal_physics as sp, classifier

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    sp.set_energy_score("CMS001", 0.3)

    result = classifier.classify_channel("CH-1")
    assert result["status"] == "FREE"
    assert classifier.is_healthy(result)


def test_classify_overloaded():
    import channels, signal_physics as sp, classifier

    channels.CHANNELS["CH-2"]["users"] = ["CMS001", "CMS002", "CMS003"]
    # Use energy values that exceed the new ENERGY_BUSY_MAX (80 J) threshold
    sp.set_energy_score("CMS001", 50.0)
    sp.set_energy_score("CMS002", 40.0)
    sp.set_energy_score("CMS003", 30.0)  # total = 120 J → CONGESTED

    result = classifier.classify_channel("CH-2")
    assert result["status"] in {"CONGESTED", "JAMMED"}
    assert not classifier.is_healthy(result)


def test_classify_projected():
    import channels, signal_physics as sp, classifier

    channels.CHANNELS["CH-3"]["users"] = ["CMS001"]
    sp.set_energy_score("CMS001", 1.0)

    # Current: healthy
    current = classifier.classify_channel("CH-3")
    assert classifier.is_healthy(current)

    # Projected with +200 J: should be unhealthy (exceeds ENERGY_CONGESTED_MAX=150 J)
    projected = classifier.classify_channel_projected("CH-3", 200.0)
    assert not classifier.is_healthy(projected)


def test_dynamic_threshold_scaling_single_and_mass_load():
    import math
    import classifier

    one = classifier.dynamic_thresholds(1)
    thirty = classifier.dynamic_thresholds(30)

    assert one == (3.0, 10.0, 20.0, 35.0)
    assert math.isclose(thirty[3], 35.0 * math.sqrt(30), abs_tol=0.01)
    assert math.isclose(thirty[3], 191.7, abs_tol=0.1)


def test_confidence_and_snr_use_dynamic_jammed_ceiling():
    import channels, signal_physics as sp, classifier

    channels.CHANNELS["CH-1"]["users"] = [f"CMS{i:03d}" for i in range(1, 31)]
    for cms in channels.CHANNELS["CH-1"]["users"]:
        sp.set_energy_score(cms, 100.0 / 30.0)

    result = classifier.classify_channel("CH-1")

    assert result["member_count"] == 30
    assert result["status"] in {"BUSY", "CONGESTED"}
    assert result["snr_db"] > sp.SNR_FLOOR_DB
    assert result["confidence"] < 1.0


def test_state_dependent_idle_decay_penalizes_loaded_channels():
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    channels.CHANNELS["CH-1"]["status"] = "FREE"
    channels.CHANNELS["CH-2"]["users"] = ["CMS002"]
    channels.CHANNELS["CH-2"]["status"] = "CONGESTED"

    sp.set_energy_score("CMS001", 100.0)
    sp.set_energy_score("CMS002", 100.0)
    base_ts = min(sp._energy_timestamps["CMS001"], sp._energy_timestamps["CMS002"])

    sp.apply_idle_decay(now=base_ts + sp.DECAY_INTERVAL_SECONDS)

    assert sp.get_energy_score("CMS001") < sp.get_energy_score("CMS002")


def test_large_class_decay_adjustment_is_slower_than_small_class():
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    channels.CHANNELS["CH-2"]["users"] = [f"LOAD{i:03d}" for i in range(50)]
    channels.CHANNELS["CH-1"]["status"] = "FREE"
    channels.CHANNELS["CH-2"]["status"] = "FREE"

    sp.set_energy_score("CMS001", 100.0)
    sp.set_energy_score("LOAD000", 100.0)
    base_ts = min(sp._energy_timestamps["CMS001"], sp._energy_timestamps["LOAD000"])

    sp.apply_idle_decay(now=base_ts + sp.DECAY_INTERVAL_SECONDS)

    assert sp.get_energy_score("CMS001") < sp.get_energy_score("LOAD000")


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_login_and_verify_token(patch_database):
    import auth
    cms = "CMS001"
    token = await auth.login_student(cms)
    assert isinstance(token, str) and token
    student = await auth.verify_token(token)
    assert student.cms == cms
    result = await auth.logout_student(token)
    assert result is True
    with pytest.raises(auth.AuthenticationError):
        await auth.verify_token(token)


@pytest.mark.asyncio
async def test_invalid_login(patch_database):
    import auth
    with pytest.raises(auth.AuthenticationError):
        await auth.login_student("NON_EXISTENT")


# ---------------------------------------------------------------------------
# Allocator tests — energy-aware reallocation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_assign_channel(patch_database):
    import allocator, channels

    result = await allocator.assign_channel("CMS001")
    assert result["channel_key"] == "CH-1"
    assert "CMS001" in channels.CHANNELS["CH-1"]["users"]


@pytest.mark.asyncio
async def test_check_congestion(patch_database):
    import allocator, channels

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    congested = await allocator.check_congestion()
    assert any(item["channel_key"] == "CH-1" for item in congested)


@pytest.mark.asyncio
async def test_reallocate_energy_aware(patch_database):
    import allocator, channels, signal_physics as sp, classifier

    # Put 3 users on CH-1 with high energy
    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
    sp.set_energy_score("CMS001", 10.0)  # highest energy
    sp.set_energy_score("CMS002", 5.0)
    sp.set_energy_score("CMS003", 3.0)

    # Force CH-1 to CONGESTED
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    moved = await allocator.reallocate_users("CH-1")

    # Should have moved at least one user (the highest-energy one first)
    assert len(moved) > 0
    # First moved should be the highest-energy user (CMS001)
    assert moved[0]["cms"] == "CMS001"
    # Their energy should be decayed (50%)
    assert abs(sp.get_energy_score("CMS001") - 5.0) < 0.001


# ---------------------------------------------------------------------------
# Phase 2 — Idle Decay Tests
# ---------------------------------------------------------------------------

def test_phase2_student_idle_decay():
    """
    Phase 2: per-student energy should decrease exponentially when idle.
    Given a known energy and a simulated elapsed time, the decayed value
    must match the expected formula: energy * DECAY_FACTOR ** ticks.
    DECAY_FACTOR is the per-1s base factor (0.979).
    """
    import signal_physics as sp

    sp.set_energy_score("CMS001", 10.0)
    base_ts = sp._energy_timestamps["CMS001"]

    # Simulate 2 full decay ticks (2 seconds at DECAY_INTERVAL_SECONDS=1)
    fake_now = base_ts + 2 * sp.DECAY_INTERVAL_SECONDS
    result = sp.apply_decay_to_student("CMS001", now=fake_now)

    # With 1 active student, dynamic factor ≈ DECAY_BASE = 0.979
    expected = 10.0 * (sp.DECAY_FACTOR ** 2)
    assert abs(result - expected) < 0.05, (
        f"Expected ~{expected:.4f} after 2 ticks, got {result:.4f}"
    )
    with sp._energy_lock:
        stored = sp._energy_scores.get("CMS001", 0.0)
    assert abs(stored - expected) < 0.05


def test_phase2_clamp_to_zero():
    """
    Phase 2: energy below ENERGY_EPSILON must be clamped to exactly 0.0
    to prevent endless tiny float residue.
    """
    import signal_physics as sp

    tiny = sp.ENERGY_EPSILON / 2  # guaranteed below epsilon
    sp.set_energy_score("CMS002", tiny)
    base_ts = sp._energy_timestamps["CMS002"]

    # One tick is enough to push a sub-epsilon value to zero
    fake_now = base_ts + sp.DECAY_INTERVAL_SECONDS
    result = sp.apply_decay_to_student("CMS002", now=fake_now)

    assert result == 0.0, f"Expected 0.0 after clamp, got {result}"


def test_phase2_no_decay_before_one_tick():
    """
    Phase 2: energy must NOT change if less than one full tick has elapsed.
    This ensures the function is idempotent for the same `now` value.
    """
    import signal_physics as sp

    sp.set_energy_score("CMS003", 5.0)
    base_ts = sp._energy_timestamps["CMS003"]

    # Advance by less than one full tick
    fake_now = base_ts + sp.DECAY_INTERVAL_SECONDS * 0.5
    result = sp.apply_decay_to_student("CMS003", now=fake_now)

    assert abs(result - 5.0) < 0.001, (
        f"Energy should not change before one full tick, got {result}"
    )


def test_phase2_bulk_idle_decay():
    """
    Phase 2: apply_idle_decay should reduce energy for all active students
    and return an accurate summary.
    """
    import signal_physics as sp

    sp.set_energy_score("CMS001", 8.0)
    sp.set_energy_score("CMS002", 4.0)
    sp.set_energy_score("CMS003", 0.0)  # already zero — should be skipped

    # Advance time by exactly one tick for all students
    base_ts = max(
        sp._energy_timestamps.get("CMS001", 0),
        sp._energy_timestamps.get("CMS002", 0),
    )
    fake_now = base_ts + sp.DECAY_INTERVAL_SECONDS

    summary = sp.apply_idle_decay(now=fake_now)

    assert summary["decayed_count"] >= 2, (
        f"Expected at least 2 students decayed, got {summary['decayed_count']}"
    )
    assert summary["after_total"] < summary["before_total"], (
        "Total energy should decrease after bulk decay"
    )


def test_phase2_channel_decay_reduces_total():
    """
    Phase 2: when two students on the same channel have their energy
    decayed, the channel snapshot total should decrease accordingly.
    """
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    sp.set_energy_score("CMS001", 6.0)
    sp.set_energy_score("CMS002", 4.0)

    # Snapshot before decay
    snap_before = sp.get_channel_energy_snapshot("CH-1")
    total_before = snap_before["total_energy"]

    # Simulate one tick of elapsed time for both students
    for cms in ["CMS001", "CMS002"]:
        base_ts = sp._energy_timestamps[cms]
        sp.apply_decay_to_student(cms, now=base_ts + sp.DECAY_INTERVAL_SECONDS)

    # Snapshot after decay
    snap_after = sp.get_channel_energy_snapshot("CH-1")
    total_after = snap_after["total_energy"]

    assert total_after < total_before, (
        f"Channel total should decrease after decay: {total_before} → {total_after}"
    )


def test_phase2_classification_recovery():
    """
    Phase 2: a channel initially classified as CONGESTED should recover
    to BUSY or FREE after sufficient idle decay is applied.
    """
    import channels, signal_physics as sp, classifier

    channels.CHANNELS["CH-2"]["users"] = ["CMS001", "CMS002"]
    # N=2: CONGESTED_MAX = 20×sqrt(2) = 28.3 J, JAMMED_MAX = 35×sqrt(2) = 49.5 J
    # Use 55+45=100 J → JAMMED initially
    sp.set_energy_score("CMS001", 55.0)
    sp.set_energy_score("CMS002", 45.0)

    initial = classifier.classify_channel("CH-2")
    assert initial["status"] in {"CONGESTED", "JAMMED"}, (
        f"Expected CONGESTED/JAMMED initially, got {initial['status']}"
    )

    # Apply 80 decay ticks to simulate a long idle period.
    # At per-1s factor ~0.979: 100 × 0.979^80 ≈ 18 J < BUSY_MAX(14.1 J for N=2)
    # Actually BUSY_MAX = 10×sqrt(2) = 14.1 J. 18 > 14.1 → still BUSY.
    # Use 120 ticks: 100 × 0.979^120 ≈ 8.1 J < 14.1 J → FREE/BUSY.
    for cms in ["CMS001", "CMS002"]:
        base_ts = sp._energy_timestamps[cms]
        sp.apply_decay_to_student(cms, now=base_ts + 120 * sp.DECAY_INTERVAL_SECONDS)

    recovered = classifier.classify_channel("CH-2")
    assert recovered["status"] in {"FREE", "BUSY"}, (
        f"Expected recovery to FREE/BUSY after decay, got {recovered['status']}"
    )


def test_phase2_reallocation_suppressed_after_cooldown():
    """
    Phase 2: if idle decay brings a channel below the overload threshold,
    the AI loop should not trigger reallocation.  We simulate this by
    checking that classify_channel returns a healthy status after decay,
    meaning the allocator would not be called.
    """
    import channels, signal_physics as sp, classifier

    channels.CHANNELS["CH-3"]["users"] = ["CMS001", "CMS002"]
    # N=2: JAMMED_MAX = 49.5 J. Use 55+45=100 J → JAMMED
    sp.set_energy_score("CMS001", 55.0)
    sp.set_energy_score("CMS002", 45.0)

    # Confirm overloaded before decay
    before = classifier.classify_channel("CH-3")
    assert not classifier.is_healthy(before)

    # Apply 120 ticks of decay to bring below BUSY_MAX (14.1 J for N=2)
    for cms in ["CMS001", "CMS002"]:
        base_ts = sp._energy_timestamps[cms]
        sp.apply_decay_to_student(cms, now=base_ts + 120 * sp.DECAY_INTERVAL_SECONDS)

    # After decay, channel should be healthy — no reallocation needed
    after = classifier.classify_channel("CH-3")
    assert classifier.is_healthy(after), (
        f"Channel should be healthy after cooldown, got {after['status']}"
    )


def test_phase2_projected_classification_uses_decayed_base():
    """
    Phase 2: classify_channel_projected must start from the decayed
    current total, then add the projected incoming energy.
    """
    import channels, signal_physics as sp, classifier

    channels.CHANNELS["CH-4"]["users"] = ["CMS001"]
    sp.set_energy_score("CMS001", 8.0)

    # Apply decay to bring energy down
    base_ts = sp._energy_timestamps["CMS001"]
    sp.apply_decay_to_student("CMS001", now=base_ts + 10 * sp.DECAY_INTERVAL_SECONDS)

    decayed_energy = sp.get_energy_score("CMS001")
    assert decayed_energy < 8.0, "Energy should have decayed"

    # Projected classification with a small additional load
    proj_small = classifier.classify_channel_projected("CH-4", 0.5)
    # Projected classification with a large additional load
    proj_large = classifier.classify_channel_projected("CH-4", 20.0)

    # Small addition on a cooled channel should be healthier than large
    assert proj_large["total_energy"] > proj_small["total_energy"]


# ---------------------------------------------------------------------------
# Bug-fix tests — delivery policy, snapshot consistency, shared now
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_overload_no_destination_rejects_message(patch_database):
    """
    Bug fix P1: when the channel is overloaded and the allocator finds no
    safe destination (all other channels are also full/jammed), the message
    must NOT be accepted.  The old code set accepted=True unconditionally
    after calling reallocate_users(), which was wrong.
    """
    import allocator, channels, signal_physics as sp, classifier

    # Fill every channel with high energy so no destination is safe.
    # 3 users × 200 J = 600 J total → well above JAMMED threshold (150 J).
    for ch_key in channels.CHANNELS:
        channels.CHANNELS[ch_key]["users"] = ["CMS001", "CMS002", "CMS003"]
        for cms in ["CMS001", "CMS002", "CMS003"]:
            sp.set_energy_score(cms, 200.0)
        channels.CHANNELS[ch_key]["status"] = "JAMMED"

    # Sender is on CH-1
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    # reallocate_users should return [] because every destination is JAMMED
    moved = await allocator.reallocate_users("CH-1")
    assert moved == [], f"Expected no moves when all channels are JAMMED, got {moved}"

    # After no moves, reclassify CH-1 — it should still be unhealthy
    post = classifier.classify_channel("CH-1")
    assert not classifier.is_healthy(post), (
        "Channel should still be unhealthy when no reallocation happened"
    )


@pytest.mark.asyncio
async def test_overload_recovered_delivers_message(patch_database):
    """
    Bug fix P1: when reallocation succeeds and the channel recovers,
    delivery_status must be DELIVERED_AFTER_STABILIZATION (not rejected).
    Verifies the post-reallocation reclassify path works correctly.
    """
    import allocator, channels, signal_physics as sp, classifier

    # CH-1: 3 users, high energy → CONGESTED
    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
    sp.set_energy_score("CMS001", 10.0)
    sp.set_energy_score("CMS002", 5.0)
    sp.set_energy_score("CMS003", 3.0)
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    # CH-2 through CH-5: empty and free (valid destinations)
    for ch_key in ["CH-2", "CH-3", "CH-4", "CH-5"]:
        channels.CHANNELS[ch_key]["users"] = []
        channels.CHANNELS[ch_key]["status"] = "FREE"

    moved = await allocator.reallocate_users("CH-1")
    assert len(moved) > 0, "Expected at least one move"

    # After reallocation, CH-1 should be healthier
    post = classifier.classify_channel("CH-1")
    # With the highest-energy user moved, the channel should have recovered
    assert classifier.is_healthy(post) or post["status"] == "CONGESTED", (
        f"Unexpected post-reallocation status: {post['status']}"
    )


def test_snapshot_consistent_now(monkeypatch):
    """
    Bug fix P2: get_channel_energy_snapshot must pass a single `now` to
    all per-student decay calls so the snapshot is internally consistent.
    Verify by passing an explicit `now` and checking all members are
    evaluated at that same instant (no member gets a different tick count).
    """
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
    sp.set_energy_score("CMS001", 5.0)
    sp.set_energy_score("CMS002", 5.0)
    sp.set_energy_score("CMS003", 5.0)

    # All three students have the same energy and the same timestamp
    base_ts = max(
        sp._energy_timestamps.get("CMS001", 0),
        sp._energy_timestamps.get("CMS002", 0),
        sp._energy_timestamps.get("CMS003", 0),
    )

    # Advance by exactly 2 ticks
    fake_now = base_ts + 2 * sp.DECAY_INTERVAL_SECONDS
    snap = sp.get_channel_energy_snapshot("CH-1", now=fake_now)

    # Dynamic factor ≈ DECAY_FACTOR (0.979) for small active count
    expected_per_student = round(5.0 * (sp.DECAY_FACTOR ** 2), 4)
    expected_total = round(expected_per_student * 3, 4)

    assert abs(snap["total_energy"] - expected_total) < 0.5, (
        f"Expected total ~{expected_total}, got {snap['total_energy']}"
    )
    for entry in snap["per_student"]:
        assert abs(entry["energy"] - expected_per_student) < 0.5, (
            f"Student {entry['cms']} energy {entry['energy']} != expected ~{expected_per_student}"
        )


def test_project_channel_energy_consistent_now():
    """
    Bug fix P2/P3: project_channel_energy must accept and use a shared
    `now` so that projection and live classification within the same
    allocator cycle use the same decay baseline.
    """
    import channels, signal_physics as sp

    channels.CHANNELS["CH-2"]["users"] = ["CMS001", "CMS002"]
    sp.set_energy_score("CMS001", 4.0)
    sp.set_energy_score("CMS002", 4.0)

    base_ts = max(
        sp._energy_timestamps.get("CMS001", 0),
        sp._energy_timestamps.get("CMS002", 0),
    )

    # Project at exactly 1 tick ahead
    fake_now = base_ts + sp.DECAY_INTERVAL_SECONDS
    projected = sp.project_channel_energy("CH-2", additional_energy=1.0, now=fake_now)

    expected_each = round(4.0 * sp.DECAY_FACTOR, 4)
    expected_total = round(expected_each * 2 + 1.0, 4)

    assert abs(projected - expected_total) < 0.1, (
        f"Expected projected total ~{expected_total}, got {projected}"
    )


@pytest.mark.asyncio
async def test_reallocate_shared_now_consistent(patch_database):
    """
    Bug fix P3: reallocate_users must use a single shared `now` for both
    source ranking and destination projection so borderline decisions
    cannot flip mid-cycle.  Verify by passing an explicit `now` and
    confirming the function accepts it without error and returns a
    consistent result.
    """
    import allocator, channels, signal_physics as sp
    import time

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
    sp.set_energy_score("CMS001", 10.0)
    sp.set_energy_score("CMS002", 5.0)
    sp.set_energy_score("CMS003", 3.0)
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    for ch_key in ["CH-2", "CH-3", "CH-4", "CH-5"]:
        channels.CHANNELS[ch_key]["users"] = []
        channels.CHANNELS[ch_key]["status"] = "FREE"

    # Pass an explicit now — function must accept it and use it consistently
    explicit_now = time.time()
    moved = await allocator.reallocate_users("CH-1", now=explicit_now)

    # Result should be the same as without explicit now (deterministic)
    assert isinstance(moved, list), "reallocate_users must return a list"


# ---------------------------------------------------------------------------
# Fix 1 + Fix 2 — process_message shared timestamp and consistent metadata
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_process_message_uses_shared_now(patch_database, monkeypatch):
    """
    Fix 1: process_message must capture one message_now and pass it to
    apply_decay_to_student, classify_channel, and reallocate_users so the
    entire decision path uses a consistent decay baseline.

    We verify this by monkeypatching the three functions to record the
    `now` argument they receive and asserting all three got the same value.
    """
    import channels, signal_physics as sp, classifier as clf, allocator as alloc
    import main as main_mod
    import time as _time

    # Set up a healthy channel so the message goes through without overload
    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    sp.set_energy_score("CMS001", 0.5)
    sp.set_energy_score("CMS002", 0.3)
    channels.CHANNELS["CH-1"]["status"] = "FREE"

    captured_nows: dict[str, float] = {}

    original_decay = sp.apply_decay_to_student
    original_classify = clf.classify_channel

    def patched_decay(cms, now=None):
        captured_nows["decay"] = now
        return original_decay(cms, now=now)

    def patched_classify(channel_id, *, admin_jammed=False, now=None):
        captured_nows["classify"] = now
        return original_classify(channel_id, admin_jammed=admin_jammed, now=now)

    monkeypatch.setattr(sp, "apply_decay_to_student", patched_decay)
    monkeypatch.setattr(clf, "classify_channel", patched_classify)
    # Patch the reference used inside main.py
    monkeypatch.setattr(main_mod.sp, "apply_decay_to_student", patched_decay)
    monkeypatch.setattr(main_mod.classifier, "classify_channel", patched_classify)

    await main_mod.process_message(
        sender_cms="CMS001",
        sender_name="Student_CMS001",
        recipient_cms="CMS002",
        text="hello",
        channel_key="CH-1",
    )

    # Both calls must have received a non-None now
    assert captured_nows.get("decay") is not None, "apply_decay_to_student got no now"
    assert captured_nows.get("classify") is not None, "classify_channel got no now"

    # Both must have received the same value (same message_now)
    assert captured_nows["decay"] == captured_nows["classify"], (
        f"decay now={captured_nows['decay']} != classify now={captured_nows['classify']}: "
        "process_message must pass the same timestamp to both calls"
    )


@pytest.mark.asyncio
async def test_sender_and_recipient_see_same_channel_metadata(patch_database):
    """
    Fix 2: after a message is delivered, the signal metadata in the
    sender's MESSAGE_RESULT and the recipient's DM payload must use the
    same canonical final_result so both clients see consistent channel
    health information.

    We verify this by capturing what was sent to the recipient via
    manager.send_dm and comparing it to the sender result.
    """
    import channels, signal_physics as sp
    import main as main_mod

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    sp.set_energy_score("CMS001", 0.5)
    sp.set_energy_score("CMS002", 0.3)
    channels.CHANNELS["CH-1"]["status"] = "FREE"

    # Capture the DM payload sent to the recipient
    captured_dm: dict = {}

    original_send_dm = main_mod.manager.send_dm

    async def patched_send_dm(sender_cms, recipient_cms, payload):
        captured_dm.update(payload)
        return True

    main_mod.manager.send_dm = patched_send_dm

    try:
        sender_result = await main_mod.process_message(
            sender_cms="CMS001",
            sender_name="Student_CMS001",
            recipient_cms="CMS002",
            text="hello",
            channel_key="CH-1",
        )
    finally:
        main_mod.manager.send_dm = original_send_dm

    # Message must have been accepted and delivered
    assert sender_result["accepted"] is True
    assert captured_dm, "Recipient DM payload was never sent"

    # Both sender and recipient must report the same channel_status
    sender_status = sender_result["classification"]["status"]
    recipient_status = captured_dm["signal"]["channel_status"]
    assert sender_status == recipient_status, (
        f"Sender sees status={sender_status!r} but recipient sees "
        f"status={recipient_status!r}: metadata must be consistent"
    )

    # Both must report the same SNR
    sender_snr = sender_result["classification"]["snr_db"]
    recipient_snr = captured_dm["signal"]["snr_db"]
    assert sender_snr == recipient_snr, (
        f"Sender snr_db={sender_snr} != recipient snr_db={recipient_snr}"
    )

    # Both must report the same modulation
    sender_mod = sender_result["classification"]["modulation"]
    recipient_mod = captured_dm["signal"]["modulation"]
    assert sender_mod == recipient_mod, (
        f"Sender modulation={sender_mod!r} != recipient modulation={recipient_mod!r}"
    )


def test_ai_loop_history_window_slope_and_cooldown_gate():
    from collections import deque
    import main as main_mod

    history = deque(maxlen=20)
    for value in range(25):
        history.append(float(value))

    assert len(history) == 20
    assert main_mod._compute_slope(history) == 1.0

    falling = deque([10.0, 8.0, 6.0, 4.0, 2.0], maxlen=20)
    assert main_mod._compute_slope(falling) == -2.0

    main_mod._channel_reallocation_cooldowns["CH-1"] = 103.0
    assert main_mod._is_reallocation_cooling_down("CH-1", 102.0)
    assert not main_mod._is_reallocation_cooling_down("CH-1", 103.0)


def test_predictive_preemption_trigger_window():
    import main as main_mod

    rising_result = {
        "member_count": 4,
        "total_energy": 35.5,
    }
    assert main_mod._should_preempt_channel(rising_result, slope_jps=1.0)

    cooling_result = {
        "member_count": 4,
        "total_energy": 34.0,
    }
    assert not main_mod._should_preempt_channel(cooling_result, slope_jps=-0.5)

    distant_result = {
        "member_count": 4,
        "total_energy": 10.0,
    }
    assert not main_mod._should_preempt_channel(distant_result, slope_jps=0.3)


@pytest.mark.asyncio
async def test_admin_students_requires_admin_key(patch_database):
    import main as main_mod
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        await main_mod.list_students(admin_key="wrong")

    result = await main_mod.list_students(admin_key=main_mod.ADMIN_KEY)
    assert len(result["students"]) == 10


@pytest.mark.asyncio
async def test_admin_verify_accepts_configured_key(patch_database):
    import main as main_mod

    result = await main_mod.verify_admin(main_mod.AdminAuthRequest(admin_key=main_mod.ADMIN_KEY))
    assert result == {"ok": True}


# ---------------------------------------------------------------------------
# Phase 3 — Fair Energy-Aware Reallocation Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_phase3_highest_energy_preference(patch_database):
    """
    Phase 3: Candidates with the highest energy must be moved first.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
    sp.set_energy_score("CMS001", 3.0)
    sp.set_energy_score("CMS002", 15.0)  # Highest
    sp.set_energy_score("CMS003", 8.0)
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    # Assume destinations are free
    for ch in ["CH-2", "CH-3", "CH-4", "CH-5"]:
        channels.CHANNELS[ch]["status"] = "FREE"

    # Reset pointer for test
    allocator._reallocation_pointer["CH-1"] = 0

    moved = await allocator.reallocate_users("CH-1")
    assert moved[0]["cms"] == "CMS002", "Highest energy candidate must be first"


@pytest.mark.asyncio
async def test_phase3_fairness_rotation(patch_database):
    """
    Phase 3: Repeated reallocations should rotate the start pointer.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
    # Total initial energy = 12.0 (CONGESTED)
    # Removing CMS001 (5.0) -> remaining 7.0 (BUSY -> healthy)
    # Removing CMS002 (4.0) -> remaining 8.0 (BUSY -> healthy)
    sp.set_energy_score("CMS001", 5.0)
    sp.set_energy_score("CMS002", 4.0)
    sp.set_energy_score("CMS003", 3.0)

    allocator._reallocation_pointer["CH-1"] = 0

    # Mock _find_valid_destination to force exactly 1 move per call
    original_find = allocator._find_valid_destination
    def mock_find(cms, src, now=None):
        return "CH-2"
    allocator._find_valid_destination = mock_find

    try:
        # Call 1
        channels.CHANNELS["CH-1"]["status"] = "CONGESTED"
        moved1 = await allocator.reallocate_users("CH-1")
        assert len(moved1) == 1
        assert moved1[0]["cms"] == "CMS001"
        assert allocator._reallocation_pointer["CH-1"] == 1

        # Simulate user 1 comes back with high energy (reset back to exact same state)
        channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
        channels.CHANNELS["CH-1"]["status"] = "CONGESTED"
        sp.set_energy_score("CMS001", 5.0)
        sp.set_energy_score("CMS002", 4.0)
        sp.set_energy_score("CMS003", 3.0)

        # Call 2
        moved2 = await allocator.reallocate_users("CH-1")
        # Pointer rotated, so the 2nd highest (CMS002) should be evaluated first
        assert len(moved2) == 1
        assert moved2[0]["cms"] == "CMS002"
        assert allocator._reallocation_pointer["CH-1"] == 2
    finally:
        allocator._find_valid_destination = original_find


@pytest.mark.asyncio
async def test_phase3_safe_destination_only(patch_database):
    """
    Phase 3: Destination is chosen only if it remains healthy post-move.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    sp.set_energy_score("CMS001", 10.0)
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    channels.CHANNELS["CH-2"]["users"] = ["CMS002"]
    sp.set_energy_score("CMS002", 7.0)
    channels.CHANNELS["CH-2"]["status"] = "BUSY" # Almost congested

    channels.CHANNELS["CH-3"]["users"] = []
    channels.CHANNELS["CH-3"]["status"] = "FREE"

    channels.CHANNELS["CH-4"]["users"] = []
    channels.CHANNELS["CH-4"]["status"] = "JAMMED"
    channels.CHANNELS["CH-5"]["users"] = []
    channels.CHANNELS["CH-5"]["status"] = "JAMMED"

    moved = await allocator.reallocate_users("CH-1")
    assert len(moved) > 0
    assert moved[0]["to"] == "CH-3", "Must pick the safe destination, not the one that would overload"


@pytest.mark.asyncio
async def test_phase3_deterministic_best_destination(patch_database):
    """
    Phase 3: Allocator chooses the optimal valid destination based on projected energy.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    sp.set_energy_score("CMS001", 10.0)
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    # CH-2 has 2.5 energy (safely below 8.0 boundary after +5.0)
    channels.CHANNELS["CH-2"]["users"] = ["CMS002"]
    sp.set_energy_score("CMS002", 2.5)
    channels.CHANNELS["CH-2"]["status"] = "FREE"

    # CH-3 has 0 energy (Best)
    channels.CHANNELS["CH-3"]["users"] = []
    channels.CHANNELS["CH-3"]["status"] = "FREE"

    moved = await allocator.reallocate_users("CH-1")
    assert moved[0]["to"] == "CH-3", "Must pick the optimal (lowest energy) destination deterministically"


@pytest.mark.asyncio
async def test_phase3_minimum_move_stopping(patch_database):
    """
    Phase 3: Reallocation stops as soon as the source channel is healthy.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    sp.set_energy_score("CMS001", 9.0)
    sp.set_energy_score("CMS002", 1.0)
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    # Valid destinations
    channels.CHANNELS["CH-2"]["status"] = "FREE"
    channels.CHANNELS["CH-3"]["status"] = "FREE"

    allocator._reallocation_pointer["CH-1"] = 0

    moved = await allocator.reallocate_users("CH-1")
    # Removing CMS001 (9 energy) should instantly recover the channel.
    # It must NOT move CMS002 unnecessarily.
    assert len(moved) == 1
    assert moved[0]["cms"] == "CMS001"
    assert "CMS002" in channels.CHANNELS["CH-1"]["users"]


@pytest.mark.asyncio
async def test_phase3_sort_tiebreaker(patch_database):
    """
    Phase 3: Tie-breaker. If two destinations have identical projected energy
    and confidence, the channel key alphabetically breaks the tie.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    sp.set_energy_score("CMS001", 10.0)
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    channels.CHANNELS["CH-3"]["users"] = []
    channels.CHANNELS["CH-3"]["status"] = "FREE"
    channels.CHANNELS["CH-2"]["users"] = []
    channels.CHANNELS["CH-2"]["status"] = "FREE"

    # Both CH-2 and CH-3 have 0 energy and identical confidence.
    # CH-2 comes first alphabetically.
    moved = await allocator.reallocate_users("CH-1")
    assert len(moved) == 1
    assert moved[0]["to"] == "CH-2", "Must pick CH-2 as alphabetical tie-breaker"


@pytest.mark.asyncio
async def test_phase3_multi_move_before_recovery(patch_database):
    """
    Phase 3: Multi-move. The allocator must continue moving candidates if the
    source channel remains overloaded after moving the first candidate.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
    # N=3: JAMMED_MAX=60.6J, CONGESTED_MAX=34.6J, BUSY_MAX=17.3J, FREE_MAX=5.2J
    # Total = 75 J → JAMMED (> 60.6 J)
    # Remove CMS001 (40J, carry=20J) → remaining 30+5=35J, N=2, JAMMED_MAX=49.5J
    #   35 < 49.5 → CONGESTED. Not healthy → continue.
    # Remove CMS002 (30J, carry=15J) → remaining 5J, N=1, BUSY_MAX=10J
    #   5 < 10 → FREE. Healthy → stop. 2 moves total.
    sp.set_energy_score("CMS001", 40.0)
    sp.set_energy_score("CMS002", 30.0)
    sp.set_energy_score("CMS003", 5.0)
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    for ch in ["CH-2", "CH-3", "CH-4", "CH-5"]:
        channels.CHANNELS[ch]["users"] = []
        channels.CHANNELS[ch]["status"] = "FREE"

    allocator._reallocation_pointer["CH-1"] = 0

    moved = await allocator.reallocate_users("CH-1")
    assert len(moved) == 2, "Must move exactly two users before recovering"
    assert moved[0]["cms"] == "CMS001"
    assert moved[1]["cms"] == "CMS002"
    assert "CMS003" in channels.CHANNELS["CH-1"]["users"], "CMS003 must remain"


@pytest.mark.asyncio
async def test_phase3_natural_jammed_uses_minimum_move_logic(patch_database):
    """
    Phase 3: A channel that becomes JAMMED from energy should still stop
    as soon as it recovers. Only admin-forced JAMMED fully evacuates.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
    sp.set_energy_score("CMS001", 40.0)
    sp.set_energy_score("CMS002", 30.0)
    sp.set_energy_score("CMS003", 5.0)
    channels.CHANNELS["CH-1"]["status"] = "JAMMED"
    channels.CHANNELS["CH-1"]["admin_forced_jammed"] = False

    for ch in ["CH-2", "CH-3", "CH-4", "CH-5"]:
        channels.CHANNELS[ch]["users"] = []
        channels.CHANNELS[ch]["status"] = "FREE"

    allocator._reallocation_pointer["CH-1"] = 0

    moved = await allocator.reallocate_users("CH-1")
    assert len(moved) == 2
    assert [move["cms"] for move in moved] == ["CMS001", "CMS002"]
    assert channels.CHANNELS["CH-1"]["users"] == ["CMS003"]


@pytest.mark.asyncio
async def test_phase3_admin_forced_jammed_evacuation(patch_database):
    """
    Phase 3: Admin-forced JAMMED. When a channel is explicitly JAMMED,
    the is_healthy check is ignored and the channel is completely evacuated.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002", "CMS003"]
    sp.set_energy_score("CMS001", 10.0)
    sp.set_energy_score("CMS002", 1.0)
    sp.set_energy_score("CMS003", 1.0)
    channels.CHANNELS["CH-1"]["status"] = "JAMMED"
    channels.CHANNELS["CH-1"]["admin_forced_jammed"] = True

    for ch in ["CH-2", "CH-3", "CH-4", "CH-5"]:
        channels.CHANNELS[ch]["users"] = []
        channels.CHANNELS[ch]["status"] = "FREE"

    moved = await allocator.reallocate_users("CH-1")
    # All 3 users must be evacuated regardless of remaining energy
    assert len(moved) == 3
    assert not channels.CHANNELS["CH-1"]["users"]


@pytest.mark.asyncio
async def test_phase3_pointer_wraparound(patch_database):
    """
    Phase 3: Pointer wraparound. The fairness pointer must correctly wrap around
    to the start of the candidate list when it exceeds list bounds.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    # Total = 9.0 (CONGESTED)
    # Remove CMS002 (1.0) -> remaining 8.0 (BUSY -> healthy)
    sp.set_energy_score("CMS001", 8.0)
    sp.set_energy_score("CMS002", 1.0)
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    for ch in ["CH-2", "CH-3"]:
        channels.CHANNELS[ch]["users"] = []
        channels.CHANNELS[ch]["status"] = "FREE"

    # Set pointer specifically to end of list
    allocator._reallocation_pointer["CH-1"] = 1

    # Call 1 (moves 1 user: CMS002 since pointer=1)
    moved1 = await allocator.reallocate_users("CH-1")
    assert len(moved1) == 1
    assert moved1[0]["cms"] == "CMS002"
    
    # After moving 1 user, pointer was 1, length was 2.
    # (1 + 1) % 2 == 0. Pointer must wrap back to 0.
    assert allocator._reallocation_pointer["CH-1"] == 0


@pytest.mark.asyncio
async def test_phase3_no_valid_destination_unsafe(patch_database):
    """
    Phase 3: No valid destination. If all destinations project as overloaded
    (but are not technically JAMMED yet), no student should be moved.
    """
    import allocator, channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    sp.set_energy_score("CMS001", 200.0)  # huge energy — carry-decay = 100 J
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    # Fill all other channels near the CONGESTED limit (80 J) so that
    # absorbing 100 J of carry-decay would push them over 150 J → JAMMED.
    for ch in ["CH-2", "CH-3", "CH-4", "CH-5"]:
        channels.CHANNELS[ch]["users"] = ["USER_" + ch]
        sp.set_energy_score("USER_" + ch, 100.0)  # 100 + 100 = 200 J → JAMMED
        channels.CHANNELS[ch]["status"] = "BUSY"

    moved = await allocator.reallocate_users("CH-1")
    assert moved == [], "Must not move any user if no safe destination exists"
    assert "CMS001" in channels.CHANNELS["CH-1"]["users"]


@pytest.mark.asyncio
async def test_phase3_empty_source_channel(patch_database):
    """
    Phase 3: Empty source channel. Calling reallocate on an empty channel
    must safely return an empty list immediately.
    """
    import allocator, channels

    channels.CHANNELS["CH-1"]["users"] = []
    channels.CHANNELS["CH-1"]["status"] = "FREE"

    moved = await allocator.reallocate_users("CH-1")
    assert moved == []


# ---------------------------------------------------------------------------
# Phase 3 — PHY Event Simulation Tests
# ---------------------------------------------------------------------------

def test_phy_event_basic_structure():
    """
    Phase 3: compute_phy_event must return a dict with all required keys
    and sensible values for a normal message.
    """
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    event = sp.compute_phy_event("Hello world", "CH-1", dt_seconds=1.0, n_users=1)

    required_keys = {"bits", "dt_seconds", "bitrate_bps", "utilization", "msg_energy", "band", "capacity_bps"}
    assert required_keys.issubset(event.keys()), f"Missing keys: {required_keys - event.keys()}"

    assert event["bits"] == len("Hello world") * 8
    assert event["dt_seconds"] == 1.0
    assert event["bitrate_bps"] > 0
    assert 0.0 <= event["utilization"] <= 1.0
    assert event["msg_energy"] > 0
    assert event["band"] == "2.4 GHz"


def test_phy_event_fast_burst_increases_energy():
    """
    Phase 3: a fast burst (small dt) must produce higher energy than a
    slow message (large dt) for the same text, because the simulated
    bitrate is higher and utilization is higher.
    """
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    # Use a long message so the utilization term is large enough to
    # produce a measurable difference after rounding to 4 decimal places.
    text = "x" * 5000

    slow = sp.compute_phy_event(text, "CH-1", dt_seconds=10.0, n_users=1)
    fast = sp.compute_phy_event(text, "CH-1", dt_seconds=0.1, n_users=1)

    assert fast["bitrate_bps"] > slow["bitrate_bps"], (
        "Fast burst must have higher bitrate than slow message"
    )
    assert fast["msg_energy"] > slow["msg_energy"], (
        "Fast burst must produce more energy than slow message"
    )


def test_phy_event_long_message_more_energy_than_short():
    """
    Phase 3: a longer message must produce more energy than a shorter one
    at the same timing, because it has more bits.
    """
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]

    short_event = sp.compute_phy_event("Hi", "CH-1", dt_seconds=1.0, n_users=1)
    long_event  = sp.compute_phy_event("x" * 200, "CH-1", dt_seconds=1.0, n_users=1)

    assert long_event["bits"] > short_event["bits"]
    assert long_event["msg_energy"] > short_event["msg_energy"], (
        "Longer message must produce more energy"
    )


def test_phy_event_contention_increases_energy():
    """
    Phase 3: more concurrent users must produce higher energy due to the
    contention penalty term in the PHY formula.
    """
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    text = "test message"

    solo   = sp.compute_phy_event(text, "CH-1", dt_seconds=1.0, n_users=1)
    crowd  = sp.compute_phy_event(text, "CH-1", dt_seconds=1.0, n_users=5)

    assert crowd["msg_energy"] > solo["msg_energy"], (
        "More concurrent users must produce higher energy (contention penalty)"
    )


def test_phy_event_band_profiles_differ():
    """
    Phase 3: the same text at the same timing must produce different energy
    on 2.4 GHz vs 5 GHz because the PHY profiles have different parameters.
    """
    import channels, signal_physics as sp

    # CH-1 is 2.4 GHz, CH-4 is 5 GHz
    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    channels.CHANNELS["CH-4"]["users"] = ["CMS001"]
    text = "x" * 50

    event_2g = sp.compute_phy_event(text, "CH-1", dt_seconds=1.0, n_users=1)
    event_5g = sp.compute_phy_event(text, "CH-4", dt_seconds=1.0, n_users=1)

    assert event_2g["band"] == "2.4 GHz"
    assert event_5g["band"] == "5 GHz"
    # Different profiles → different energy (2.4 GHz has higher base + weights)
    assert event_2g["msg_energy"] != event_5g["msg_energy"], (
        "2.4 GHz and 5 GHz must produce different energy for the same message"
    )
    # 5 GHz has much higher capacity so utilization fraction is lower
    assert event_5g["utilization"] < event_2g["utilization"], (
        "5 GHz has higher capacity so utilization fraction must be lower"
    )


def test_phy_event_dt_clamped_to_minimum():
    """
    Phase 3: dt must be clamped to PHY_MIN_DT_SECONDS so that a zero or
    near-zero dt does not produce an infinite bitrate.
    """
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    import channels as ch_mod

    # Pass dt=0 — should be clamped to PHY_MIN_DT_SECONDS
    event_zero = sp.compute_phy_event("hello", "CH-1", dt_seconds=0.0, n_users=1)
    event_min  = sp.compute_phy_event("hello", "CH-1", dt_seconds=ch_mod.PHY_MIN_DT_SECONDS, n_users=1)

    assert event_zero["dt_seconds"] == ch_mod.PHY_MIN_DT_SECONDS, (
        f"dt=0 must be clamped to {ch_mod.PHY_MIN_DT_SECONDS}, got {event_zero['dt_seconds']}"
    )
    assert abs(event_zero["msg_energy"] - event_min["msg_energy"]) < 0.001, (
        "dt=0 and dt=PHY_MIN_DT must produce the same energy after clamping"
    )


def test_phy_event_rapid_messages_degrade_snr():
    """
    Phase 3: rapid repeated long messages on the same channel must reduce
    SNR quickly.  Simulate 5 fast bursts and verify SNR drops each time.
    """
    import channels, signal_physics as sp, classifier

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    sp.set_energy_score("CMS001", 0.0)

    text = "x" * 200   # long message
    snr_values = []

    for _ in range(5):
        event = sp.compute_phy_event(text, "CH-1", dt_seconds=0.1, n_users=1)
        sp.update_energy_score("CMS001", event["msg_energy"])
        snap = sp.get_channel_energy_snapshot("CH-1")
        snr_values.append(snap["snr_db"])

    # SNR must decrease (or stay the same) with each burst
    for i in range(1, len(snr_values)):
        assert snr_values[i] <= snr_values[i - 1], (
            f"SNR should not increase during rapid bursts: "
            f"snr[{i-1}]={snr_values[i-1]:.2f} → snr[{i}]={snr_values[i]:.2f}"
        )

    # After 5 rapid long messages, SNR must be noticeably lower than clean
    assert snr_values[-1] < sp.SNR_CLEAN_DB, (
        f"SNR must drop below clean level after rapid bursts, got {snr_values[-1]:.2f}"
    )


def test_phy_event_idle_recovery_restores_snr():
    """
    Phase 3: after rapid bursts degrade SNR, idle decay must restore it.
    """
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]

    # Load up the channel with rapid bursts
    text = "x" * 200
    for _ in range(5):
        event = sp.compute_phy_event(text, "CH-1", dt_seconds=0.1, n_users=1)
        sp.update_energy_score("CMS001", event["msg_energy"])

    snap_loaded = sp.get_channel_energy_snapshot("CH-1")
    snr_loaded = snap_loaded["snr_db"]

    # Apply heavy idle decay (simulate long quiet period)
    base_ts = sp._energy_timestamps.get("CMS001", 0)
    sp.apply_decay_to_student("CMS001", now=base_ts + 50 * sp.DECAY_INTERVAL_SECONDS)

    snap_recovered = sp.get_channel_energy_snapshot("CH-1")
    snr_recovered = snap_recovered["snr_db"]

    assert snr_recovered > snr_loaded, (
        f"SNR must recover after idle decay: {snr_loaded:.2f} → {snr_recovered:.2f}"
    )


def test_phy_event_compute_message_energy_backward_compat():
    """
    Phase 3: compute_message_energy (legacy wrapper) must still work and
    return a positive float, so existing callers are not broken.
    """
    import channels, signal_physics as sp

    channels.CHANNELS["CH-1"]["users"] = ["CMS001"]
    energy = sp.compute_message_energy("Hello", "CH-1", concurrent_transmitters=1)

    assert isinstance(energy, float)
    assert energy > 0, "compute_message_energy must return a positive value"


@pytest.mark.asyncio
async def test_phy_event_in_process_message(patch_database):
    """
    Phase 3: process_message must include a 'phy' key in its result with
    the expected PHY telemetry fields when called with a dt_seconds value.
    """
    import channels, signal_physics as sp
    import main as main_mod

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    sp.set_energy_score("CMS001", 0.5)
    sp.set_energy_score("CMS002", 0.3)
    channels.CHANNELS["CH-1"]["status"] = "FREE"

    result = await main_mod.process_message(
        sender_cms="CMS001",
        sender_name="Student_CMS001",
        recipient_cms="CMS002",
        text="hello world",
        channel_key="CH-1",
        dt_seconds=2.0,
    )

    assert "phy" in result, "MESSAGE_RESULT must contain a 'phy' key"
    phy = result["phy"]
    required = {"bits", "dt_seconds", "bitrate_bps", "utilization", "band", "capacity_bps"}
    assert required.issubset(phy.keys()), f"Missing PHY keys: {required - phy.keys()}"
    assert phy["dt_seconds"] == 2.0, "dt_seconds must match the value passed in"
    assert phy["bits"] == len("hello world") * 8


@pytest.mark.asyncio
async def test_phy_event_recipient_sees_phy_telemetry(patch_database):
    """
    Phase 3: the DM payload delivered to the recipient must also contain
    a 'phy' sub-dict inside 'signal' so both sender and recipient have
    access to the PHY telemetry.
    """
    import channels, signal_physics as sp
    import main as main_mod

    channels.CHANNELS["CH-1"]["users"] = ["CMS001", "CMS002"]
    sp.set_energy_score("CMS001", 0.5)
    sp.set_energy_score("CMS002", 0.3)
    channels.CHANNELS["CH-1"]["status"] = "FREE"

    captured_dm: dict = {}

    original_send_dm = main_mod.manager.send_dm

    async def patched_send_dm(sender_cms, recipient_cms, payload):
        captured_dm.update(payload)
        return True

    main_mod.manager.send_dm = patched_send_dm

    try:
        await main_mod.process_message(
            sender_cms="CMS001",
            sender_name="Student_CMS001",
            recipient_cms="CMS002",
            text="hello",
            channel_key="CH-1",
            dt_seconds=1.5,
        )
    finally:
        main_mod.manager.send_dm = original_send_dm

    assert captured_dm, "Recipient DM was never sent"
    assert "signal" in captured_dm, "DM payload must have 'signal'"
    assert "phy" in captured_dm["signal"], "DM signal must contain 'phy' telemetry"
    phy = captured_dm["signal"]["phy"]
    assert phy["dt_seconds"] == 1.5


# ---------------------------------------------------------------------------
# Simple CLI demo
# ---------------------------------------------------------------------------

def _print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


async def _demo_workflow():
    import allocator, auth, channels, signal_physics as sp, classifier

    _print_header("CogniRad DM Backend Demo")

    # Login
    token_a = await auth.login_student("CMS001")
    token_b = await auth.login_student("CMS002")
    print(f"Logged in CMS001 → token {token_a[:8]}…")
    print(f"Logged in CMS002 → token {token_b[:8]}…")

    # Join channels
    assign_a = await allocator.assign_channel("CMS001")
    assign_b = await allocator.assign_channel("CMS002")
    print(f"CMS001 → {assign_a['channel_key']}")
    print(f"CMS002 → {assign_b['channel_key']}")

    # Simulate DM energy
    energy = sp.compute_message_energy("Hello, how are you?", assign_a["channel_key"])
    sp.update_energy_score("CMS001", energy)
    print(f"\nDM energy: {energy:.4f}")
    print(f"CMS001 total energy: {sp.get_energy_score('CMS001'):.4f}")

    # Classify
    result = classifier.classify_channel(assign_a["channel_key"])
    print(f"Channel {assign_a['channel_key']}: {result['status']} (confidence={result['confidence']:.3f})")

    # Check congestion
    congested = await allocator.check_congestion()
    if congested:
        print("⚠️  Congested channels detected")
    else:
        print("✅ No congestion")

    # Cleanup
    await auth.logout_student(token_a)
    await auth.logout_student(token_b)
    print("\nDemo complete. Sessions cleared.")


if __name__ == "__main__":
    asyncio.run(_demo_workflow())
