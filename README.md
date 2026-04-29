# CogniRad

**CogniRad** is an advanced cognitive-radio simulation and secure messaging platform. The system operates on a direct-messaging (DM) model, constrained by radio-frequency physics and channel contention algorithms. The backend uses Python and FastAPI to run real-time WebSocket communication, while an autonomous AI loop dynamically reallocates users across frequency bands based on aggregate energy thresholds.

## Features
- **Cognitive Radio Physics Model:** A realistic energy accumulation system where direct messages add energy to shared frequency bands.
- **AI Channel Orchestration:** A background decision engine that monitors band health and seamlessly reallocates nodes to prevent frequency jamming.
- **Secure DM Routing:** Dual-dictionary WebSocket architecture to guarantee fast, private point-to-point delivery.
- **Immersive Frontend UI:** A fully optimized, mobile-responsive "terminal" interface featuring a WebGL 3D wave background, seamless single-page slide transitions, and native-feeling interactions.
- **Spectrum Classification (ML):** A dedicated PyTorch pipeline for training and evaluating dataset models for signal classification.

---

## Project Structure

A professional separation of concerns has been applied to the project repository:

```text
CogniRad/
├── main.py                # FastAPI entry point, API routes, WS Manager
├── database.py            # SQLAlchemy async database configuration
├── auth.py                # Session validation and login logic
├── channels.py            # In-memory channel states and band membership
├── allocator.py           # The Decision Engine for user reallocation
├── classifier.py          # AI health projection and classification logic
├── signal_physics.py      # Core physics math for computing signal energy
│
├── static/                # Frontend Web Application
│   ├── app.html           # Main DM and contacts interface
│   ├── index.html         # Secure access gateway login
│   ├── css/               # Modular CSS (app.css, style.css)
│   └── js/                # Client logic (app.js, auth.js)
│
├── ml/                    # Machine Learning and Data Science pipeline
│   ├── cognirad_training.py # Core PyTorch training loop
│   ├── models/            # Saved weights (.pt files)
│   ├── dataset/           # Training data
│   ├── layer_1/           # Scripts for dataset mapping
│   └── plots/             # Generated analytics and visual plots
│
├── planning/              # Project specs and build sequence docs
├── documentation/         # General project reference guides
├── tests/                 # Integration test suite (test_all.py)
└── requirements.txt       # Python dependencies
```

*Note on Dataset: The `ml/dataset/` directory is intentionally excluded from version control due to its massive size. This project trains on the [DeepSig RadioML Dataset](https://www.deepsig.ai/datasets). To replicate the ML training locally, you must download the dataset directly from DeepSig and place it in the `ml/dataset/` directory.*

---

## Setup & Installation

### Prerequisites
- **Python 3.10+**
- **pip** and **virtualenv** (recommended)

### 1. Environment Setup
Create and activate a virtual environment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Database Initialization
Ensure `students.json` is located in the root directory and populated with valid CMS IDs and names. The system will automatically build `cognirad.db` upon the first run using SQLAlchemy.

### 3. Running the Server
To start the FastAPI server with live reloading:
```bash
uvicorn main:app --reload
```
Alternatively, you can just run:
```bash
python main.py
```

### 4. Accessing the Client
Once the server is running, navigate your browser to:
`http://localhost:8000/static/index.html`

---

## Testing
To run the automated test suite and verify the physics, classification, and allocation logic:
```bash
pytest test_all.py -v --asyncio-mode=auto
```

---

## Maintenance Notes
- **Static Assets:** Do not change file structures within the `static` directory without updating the relative paths in `app.html` and `index.html`.
- **Background Tasks:** The AI allocator loop runs every 5 seconds asynchronously within the FastAPI lifecycle context. Adjust the intervals in `main.py` if testing specific energy decay behaviors.

---

## Changelog

### April 29, 2025

#### Phase 2 — Idle Energy Decay (`signal_physics.py`, `main.py`)

Energy previously only ever increased. Long-lived sessions became permanently stressed and channels could never recover without a forced reallocation. Phase 2 adds a time-based decay system so inactive users and channels cool down naturally.

- Added tunable constants: `DECAY_INTERVAL_SECONDS = 5`, `DECAY_FACTOR = 0.95`, `ENERGY_EPSILON = 0.01`
- `apply_decay_to_student(cms, now)` — exponential decay in discrete ticks (`energy × 0.95^ticks`), idempotent for the same `now`, clamps sub-epsilon values to exactly `0.0`
- `apply_idle_decay(now)` — bulk decay for all active students; called at the start of every AI observation tick
- `get_energy_score(cms, now)` — now decay-aware; applies lazy decay before returning so all callers always get a time-accurate value
- `get_channel_energy_snapshot(channel_id, now)` — accepts a shared `now` so all member decays within one snapshot use the same instant (no tick-boundary splits mid-snapshot)
- `project_channel_energy(channel_id, additional_energy, now)` — same shared-`now` fix for projections used by the allocator
- AI loop order changed to: **decay → classify → reallocate** (only if still overloaded after decay)
- `process_message()` decays the sender before adding new message energy

#### Bug Fix — Delivery Policy (`main.py`)

`process_message()` set `accepted = True` unconditionally after calling `reallocate_users()`, even when the allocator found no safe destination and stabilisation never happened.

**Fix:** reclassify the sender's channel after reallocation and apply a three-way delivery policy on the post-decision state:

| Post-decision status | Delivery outcome |
|---|---|
| FREE / BUSY | `DELIVERED_AFTER_STABILIZATION` |
| CONGESTED | `DELIVERED_CHANNEL_DEGRADED` |
| JAMMED | `REJECTED_CHANNEL_JAMMED` |

#### Bug Fix — Snapshot Timing Consistency (`signal_physics.py`)

`get_channel_energy_snapshot` and `project_channel_energy` each called `time.time()` independently per member, so a snapshot could span a decay-tick boundary mid-loop and mix values from different ticks.

**Fix:** both functions now accept `now: float | None = None`; one timestamp is captured at the top and passed to every per-student decay call.

#### Bug Fix — Shared Timestamp Through `process_message` (`main.py`)

`process_message()` was calling decay, classify, and reallocate with separate implicit `time.time()` calls that could cross a tick boundary mid-request.

**Fix:** `message_now = time.time()` captured once at the top and passed to `apply_decay_to_student`, `classify_channel`, and `reallocate_users`. The post-reallocation reclassify intentionally uses a fresh `post_now` because the channel topology has actually changed.

#### Bug Fix — Recipient DM Metadata Consistency (`main.py`)

The recipient's `signal` dict used the pre-reallocation `result` while the sender's `MESSAGE_RESULT` used `final_result`. Both clients could see different channel health metadata for the same delivered message.

**Fix:** both sender and recipient now use `final_result` (the canonical post-decision classification). `modulation` also added to the sender's `classification` dict.

#### Phase 3 — Deterministic Optimal Destination Selection (`allocator.py`)

`_find_valid_destination` previously returned the first channel that passed the health check, which was dict-iteration order — non-deterministic.

**Fix:** collects all valid (healthy-after-move) destinations and sorts them by `(total_energy, confidence, channel_key)` to always pick the lowest-load channel deterministically.

#### Test Suite — 40 Tests, All Passing (`test_all.py`)

| Group | Tests |
|---|---|
| Channel registry | 2 |
| Signal physics | 4 |
| Classifier | 3 |
| Auth | 2 |
| Allocator Phase 1 | 3 |
| Phase 2 idle decay | 8 |
| Bug fixes | 6 |
| Phase 3 decision engine | 12 |

Six coverage gaps closed in Phase 3 tests: sort tie-breaker, multi-move before recovery, admin-forced JAMMED evacuation, pointer wraparound, no-valid-destination (unsafe channels), and empty source channel.

#### Terminal Dashboard (`terminal_dashboard.py`) — New File

A live ASCII admin portal renders in the terminal every 4 seconds when the server is running.

- ASCII art banner with the server URL printed in **red**
- Per-channel energy progress bars coloured by status (green = FREE, yellow = BUSY, red = CONGESTED, magenta = JAMMED), showing energy, user count, SNR, modulation, and frequency
- Per-active-student energy bars that shift green → yellow → red as energy rises
- Live message feed showing the last 8 messages with timestamp, sender → recipient, channel, energy, status, and delivery outcome — all colour-coded
- Refreshes every 4 seconds via a background asyncio task started in the FastAPI lifespan
- `record_message()` called from `process_message()` after every DM attempt so the feed updates in real time
- Pure display module — no business logic, no state mutation
