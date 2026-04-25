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
