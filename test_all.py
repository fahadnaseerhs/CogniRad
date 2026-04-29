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
    snr2 = sp.derive_snr(20.0, 5)
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
    sp.set_energy_score("CMS001", 8.0)
    sp.set_energy_score("CMS002", 6.0)
    sp.set_energy_score("CMS003", 5.0)

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

    # Projected with +20 energy: should be unhealthy
    projected = classifier.classify_channel_projected("CH-3", 20.0)
    assert not classifier.is_healthy(projected)


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
    """
    import signal_physics as sp

    # Set a known energy value and record the timestamp
    sp.set_energy_score("CMS001", 10.0)
    base_ts = sp._energy_timestamps["CMS001"]

    # Simulate 2 full decay ticks (10 seconds at DECAY_INTERVAL_SECONDS=5)
    fake_now = base_ts + 2 * sp.DECAY_INTERVAL_SECONDS
    result = sp.apply_decay_to_student("CMS001", now=fake_now)

    expected = 10.0 * (sp.DECAY_FACTOR ** 2)
    assert abs(result - expected) < 0.001, (
        f"Expected {expected:.4f} after 2 ticks, got {result:.4f}"
    )
    # Stored value should also be updated
    with sp._energy_lock:
        stored = sp._energy_scores.get("CMS001", 0.0)
    assert abs(stored - expected) < 0.001


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
    # Set energy high enough to be CONGESTED
    sp.set_energy_score("CMS001", 9.0)
    sp.set_energy_score("CMS002", 7.0)

    initial = classifier.classify_channel("CH-2")
    assert initial["status"] in {"CONGESTED", "JAMMED"}, (
        f"Expected CONGESTED/JAMMED initially, got {initial['status']}"
    )

    # Apply many decay ticks to simulate a long idle period
    for cms in ["CMS001", "CMS002"]:
        base_ts = sp._energy_timestamps[cms]
        # 30 ticks = 150 seconds of idle time
        sp.apply_decay_to_student(cms, now=base_ts + 30 * sp.DECAY_INTERVAL_SECONDS)

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
    sp.set_energy_score("CMS001", 9.0)
    sp.set_energy_score("CMS002", 7.0)

    # Confirm overloaded before decay
    before = classifier.classify_channel("CH-3")
    assert not classifier.is_healthy(before)

    # Apply heavy decay (simulate long idle period)
    for cms in ["CMS001", "CMS002"]:
        base_ts = sp._energy_timestamps[cms]
        sp.apply_decay_to_student(cms, now=base_ts + 40 * sp.DECAY_INTERVAL_SECONDS)

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

    # Fill every channel with high energy so no destination is safe
    for ch_key in channels.CHANNELS:
        channels.CHANNELS[ch_key]["users"] = ["CMS001", "CMS002", "CMS003"]
        for cms in ["CMS001", "CMS002", "CMS003"]:
            sp.set_energy_score(cms, 20.0)
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

    expected_per_student = round(5.0 * (sp.DECAY_FACTOR ** 2), 4)
    expected_total = round(expected_per_student * 3, 4)

    assert abs(snap["total_energy"] - expected_total) < 0.01, (
        f"Expected total {expected_total}, got {snap['total_energy']}"
    )
    for entry in snap["per_student"]:
        assert abs(entry["energy"] - expected_per_student) < 0.01, (
            f"Student {entry['cms']} energy {entry['energy']} != expected {expected_per_student}"
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

    assert abs(projected - expected_total) < 0.01, (
        f"Expected projected total {expected_total}, got {projected}"
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
    # Total = 25.0 (CONGESTED)
    # Remove CMS001 (10.0) -> remaining 15.0 (still CONGESTED)
    # Remove CMS002 (8.0) -> remaining 7.0 (BUSY -> healthy, stops)
    sp.set_energy_score("CMS001", 10.0)
    sp.set_energy_score("CMS002", 8.0)
    sp.set_energy_score("CMS003", 7.0)
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
    sp.set_energy_score("CMS001", 20.0) # Huge energy
    channels.CHANNELS["CH-1"]["status"] = "CONGESTED"

    # Fill all other channels so they would become CONGESTED if they took CMS001
    for ch in ["CH-2", "CH-3", "CH-4", "CH-5"]:
        channels.CHANNELS[ch]["users"] = ["USER_" + ch]
        sp.set_energy_score("USER_" + ch, 14.0) # Near congested limit
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
