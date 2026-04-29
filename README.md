# CogniRad — Cognitive Radio Spectrum Manager

An AI-driven Wi-Fi channel allocation backend with real-time DM routing,
cumulative energy tracking, idle decay, and fair minimum-move reallocation.

---

## What It Does

CogniRad models a classroom Wi-Fi environment as a cognitive-radio system.
Students are assigned to shared spectrum channels (2.4 GHz / 5 GHz bands).
Every message they send accumulates energy on their channel.
An AI loop watches channel health every 5 seconds and moves students when
a channel becomes overloaded — fairly, minimally, and deterministically.

---

## Architecture

```
main.py               FastAPI app, WebSocket hub, DM pipeline
signal_physics.py     Energy math, idle decay, channel snapshots
classifier.py         Channel health classification (FREE/BUSY/CONGESTED/JAMMED)
allocator.py          Decision engine — fair minimum-move reallocation
channels.py           In-memory channel registry
auth.py               Token-based student authentication
database.py           SQLite persistence (students, sessions, messages)
terminal_dashboard.py Live ASCII admin portal in the terminal
```

---

## Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/static/index.html` in a browser.
The terminal shows a live dashboard that refreshes every 4 seconds.

---

## Today's Work — Full Session Log

### Phase 2 — Idle Energy Decay

**Problem:** Energy only ever increased. Long-lived sessions became
permanently stressed. Channels could never recover without forced
reallocation.

**What was built:**

- `DECAY_INTERVAL_SECONDS = 5`, `DECAY_FACTOR = 0.95`, `ENERGY_EPSILON = 0.01`
  added as tunable constants in `signal_physics.py`.

- `apply_decay_to_student(cms, now)` — exponential decay in discrete ticks
  (`energy × factor^ticks`). Idempotent for the same `now` value. Clamps
  sub-epsilon values to exactly `0.0`.

- `apply_idle_decay(now)` — bulk decay for all active students. Called at
  the start of every AI observation tick.

- `get_energy_score(cms, now)` — now decay-aware. Applies lazy decay before
  returning so all callers always get a time-accurate value.

- `get_channel_energy_snapshot(channel_id, now)` — accepts a shared `now`
  so all member decays within one snapshot use the same instant (no
  tick-boundary splits mid-snapshot).

- `project_channel_energy(channel_id, additional_energy, now)` — same
  shared-`now` fix for projections.

- AI loop in `main.py` updated: decay → classify → reallocate (only if
  still overloaded after decay).

- `process_message()` decays the sender before adding new message energy.

**Tests added:** 8 Phase 2 tests covering decay math, epsilon clamp,
no-decay-before-one-tick, bulk decay, channel total reduction,
classification recovery, reallocation suppression after cooldown,
projected classification using decayed base.

---

### Bug Fixes — Delivery Policy and Snapshot Consistency

**Bug 1 — Overloaded sends always accepted**

`process_message()` set `accepted = True` unconditionally after calling
`reallocate_users()`, even when the allocator found no safe destination
and stabilisation never happened.

Fix: reclassify the sender's channel after reallocation. Apply a
three-way delivery policy on the post-decision state:

| Post-decision status | Delivery outcome |
|---|---|
| FREE / BUSY | `DELIVERED_AFTER_STABILIZATION` |
| CONGESTED | `DELIVERED_CHANNEL_DEGRADED` |
| JAMMED | `REJECTED_CHANNEL_JAMMED` |

**Bug 2 — Snapshot timing inconsistency**

`get_channel_energy_snapshot` and `project_channel_energy` each called
`time.time()` independently per member, so a snapshot could span a
decay-tick boundary mid-loop and mix values from different ticks.

Fix: both functions now accept `now: float | None = None`. One timestamp
is captured at the top and passed to every per-student decay call.

**Bug 3 — Allocator used different decay baselines for ranking vs projection**

Source ranking and destination projection each called `get_energy_score`
with independent wall-clock reads. A tick boundary mid-loop could flip a
borderline healthy/unhealthy decision.

Fix: `reallocate_users(source_key, now)` and `_find_valid_destination`
both accept `now`. One timestamp is captured at the top of
`reallocate_users` and threaded through all reads, rankings, and
projections in that decision cycle.

**Tests added:** 4 bug-fix tests covering no-destination rejection,
post-recovery delivery, snapshot consistency, and shared-now projection.

---

### Fix — Shared Timestamp Through `process_message`

`process_message()` was calling decay, classify, and reallocate with
separate implicit `time.time()` calls that could cross a tick boundary
mid-request.

Fix: `message_now = time.time()` captured once at the top and passed to:
- `sp.apply_decay_to_student(sender_cms, now=message_now)`
- `classifier.classify_channel(channel_key, now=message_now)`
- `allocator.reallocate_users(channel_key, now=message_now)`

The post-reallocation reclassify intentionally uses a fresh `post_now`
because the channel topology has actually changed (users moved).

**Fix — Recipient DM metadata consistency**

The recipient's `signal` dict used the pre-reallocation `result` while
the sender's `MESSAGE_RESULT` used `final_result`. Both clients could see
different channel health metadata for the same delivered message.

Fix: both sender and recipient now use `final_result` (the canonical
post-decision classification). `modulation` also added to the sender's
`classification` dict.

**Tests added:** 2 tests — shared-now verification via monkeypatching,
and sender/recipient metadata consistency check.

---

### Phase 3 — Deterministic Optimal Destination Selection

**What was built:**

`_find_valid_destination` now collects all valid (healthy-after-move)
destination channels and sorts them deterministically:

1. Lowest projected total energy
2. Lowest projected confidence score
3. Channel key alphabetically (tie-breaker)

Previously it returned the first channel that passed the health check,
which was dict-iteration order — non-deterministic.

**Tests added (original 5):**
- `test_phase3_highest_energy_preference`
- `test_phase3_fairness_rotation`
- `test_phase3_safe_destination_only`
- `test_phase3_deterministic_best_destination`
- `test_phase3_minimum_move_stopping`

**Test improvements (6 gaps closed):**

| Gap | Fix |
|---|---|
| Fragile fairness rotation test | Reset energy scores to exact values before call 2 |
| Sort tie-breaker untested | New test: two empty channels, asserts alphabetical winner |
| Multi-move scenario missing | New test: total=25J, one move insufficient, asserts `len(moved)==2` |
| Admin JAMMED evacuation untested | New test: JAMMED channel, asserts all users evacuated |
| Pointer wraparound untested | New test: pointer=1 on 2-member list, asserts wraps to 0 |
| No valid destination (unsafe) untested | New test: destinations near limit, asserts `moved==[]` |
| Empty source channel untested | New test: empty users list, asserts immediate `[]` |
| Exact boundary in deterministic test | CH-2 energy changed 3.0 → 2.5 to avoid `== ENERGY_BUSY_MAX` |
| Implicit users in safe-destination test | Explicit `users=[]` for CH-4 and CH-5 |

---

### Terminal Dashboard

A live ASCII admin portal renders in the terminal every 4 seconds.

**Features:**
- ASCII art banner with server URL in **red**
- Per-channel energy progress bars coloured by status
  (green=FREE, yellow=BUSY, red=CONGESTED, magenta=JAMMED)
- Per-active-student energy bars (green→yellow→red as energy rises)
- Live message feed: last 8 messages with timestamp, sender→recipient,
  channel, energy, status, delivery outcome — all colour-coded
- Footer with docs and API links

**Implementation:** `terminal_dashboard.py` is a standalone module.
`main.py` calls `dashboard.start_dashboard()` in the lifespan and
`dashboard.record_message(...)` at the end of every `process_message()`.
No business logic in the dashboard — pure display.

---

## Test Suite

```bash
pytest test_all.py -v --asyncio-mode=auto
```

**40 tests, all passing.**

| Group | Count |
|---|---|
| Channel registry | 2 |
| Signal physics | 4 |
| Classifier | 3 |
| Auth | 2 |
| Allocator (Phase 1) | 3 |
| Phase 2 — Idle decay | 8 |
| Bug fixes | 6 |
| Phase 3 — Decision engine | 12 |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| POST | `/auth/login` | Student login → token |
| POST | `/logout` | Invalidate token |
| GET | `/channel/state` | All channel states with energy |
| POST | `/channel/join` | Assign student to best channel |
| POST | `/channel/message` | Send a DM (REST fallback) |
| GET | `/channel/{id}/messages` | Recent messages |
| GET | `/channel/{id}/members` | Channel member list |
| POST | `/admin/jam` | Force channel to JAMMED |
| POST | `/admin/unjam` | Clear JAMMED status |
| GET | `/admin/students` | List all students |
| POST | `/admin/reallocate` | Trigger manual reallocation |
| WS | `/ws/{token}` | Authenticated real-time DM hub |

Interactive docs: `http://127.0.0.1:8000/docs`

---

## Energy Model

```
message_energy  = base_cost + (len(text) × per_char_cost) × tx_scale
student_energy += message_energy                    (on every send)
student_energy  = student_energy × 0.95^ticks       (every 5 s idle)
channel_energy  = Σ student_energy of members
```

Channel status thresholds:

| Status | Total energy |
|---|---|
| FREE | < 2.0 J |
| BUSY | 2.0 – 8.0 J |
| CONGESTED | 8.0 – 15.0 J |
| JAMMED | > 15.0 J |

---

## WebSocket Protocol

**Send:**
```json
{ "to": "CMS007", "text": "Hello" }
```

**Receive (DM):**
```json
{
  "type": "DM",
  "from": "CMS001",
  "to": "CMS007",
  "text": "Hello",
  "signal": {
    "energy": 1.05,
    "sender_total_energy": 4.2,
    "channel_status": "BUSY",
    "snr_db": 18.4,
    "modulation": "16-QAM"
  }
}
```

**Receive (reallocation notice):**
```json
{
  "type": "REALLOCATED",
  "from": "CH-1",
  "to": "CH-3",
  "frequency": "2.462 GHz",
  "decayed_energy": 2.1
}
```
