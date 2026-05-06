# CogniRad

CogniRad is a cognitive-radio simulation platform with real-time direct messaging, channel health classification, and automatic user reallocation across Wi-Fi bands.

It combines:
- a FastAPI backend,
- WebSocket-based live messaging,
- RF-inspired energy and SNR modeling,
- an AI loop that prevents channel collapse,
- student and admin web interfaces,
- and an async SQLite persistence layer.

---

## What This Project Is

CogniRad models communication in shared spectrum where each message contributes energy load on a channel.  
As load rises, channel quality degrades from `FREE` to `BUSY`, then `CONGESTED`, and finally `JAMMED`.

When overload is detected (or predicted), the allocator moves users to safer channels using a fair, minimum-move strategy.

This is both:
- **an educational system** (network/spectrum behavior simulation), and
- **a full-stack application** (backend + realtime frontend + dashboards + tests).

---

## Core Features

- Authenticated student login/session handling
- Direct messaging (DM) via WebSocket and REST fallback
- 5-channel spectrum model (`CH-1` to `CH-5`)
- Per-student cumulative energy scoring
- Signal-derived channel classification (energy, SNR, modulation)
- Automatic and manual channel reallocation
- Admin force-jam / unjam controls
- Live spectrum telemetry stream for admin dashboard
- Terminal dashboard + browser dashboard
- Async SQLAlchemy + SQLite persistence

---

## Channel Model

Defined in `channels.py`.

| Channel | Frequency |
|---|---|
| CH-1 | 2.412 GHz |
| CH-2 | 2.437 GHz |
| CH-3 | 2.462 GHz |
| CH-4 | 5.180 GHz |
| CH-5 | 5.240 GHz |

### Channel states
- `FREE`: low load
- `BUSY`: active but healthy
- `CONGESTED`: degraded quality; AI may reallocate
- `JAMMED`: severe/forced interference; communication constrained and users moved

---

## How It Works (End-to-End)

1. Student logs in at `static/index.html` (`/auth/login`).
2. Student is assigned a channel (`/channel/join`).
3. App opens authenticated WebSocket (`/ws/{token}`).
4. Each outbound message goes through `process_message()` in `main.py`:
   - PHY event is computed from text and timing,
   - sender energy is updated,
   - source channel is reclassified,
   - if overloaded, allocator attempts recovery,
   - message is delivered/rejected with delivery metadata.
5. Background AI loop runs every second:
   - applies idle decay,
   - classifies all channels,
   - triggers reactive or predictive reallocations,
   - broadcasts `SPECTRUM_TICK` to admin subscribers (`/ws/spectrum`).

---

## Architecture

- `main.py`: FastAPI app, endpoints, websocket hub, AI loop
- `auth.py`: token/session auth
- `database.py`: async ORM + persistence
- `channels.py`: channel registry + membership helpers
- `signal_physics.py`: energy, SNR, modulation, decay
- `classifier.py`: health classification
- `allocator.py`: fair destination-aware reallocation
- `terminal_dashboard.py`: live terminal status output
- `static/`: student/admin frontend

---

## Project Structure

```text
CogniRad/
├── main.py
├── auth.py
├── allocator.py
├── channels.py
├── classifier.py
├── database.py
├── signal_physics.py
├── terminal_dashboard.py
├── students.json
├── simulate_load.py
├── test_all.py
├── requirements.txt
├── static/
│   ├── index.html
│   ├── app.html
│   ├── admin.html
│   ├── spectrum.html
│   ├── css/
│   └── js/
└── ml/
```

---

## Setup

### 1) Create virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Ensure seed file exists

`students.json` must be present at project root and contain CMS-to-name mapping, for example:

```json
{
  "458214": "Daneen Anwar",
  "481862": "Noor ul Harem"
}
```

---

## Running The System

### Standard mode

```bash
python main.py
```

### Simulation browser mode (opens many tabs)

```bash
python main.py --simulate
```

### Optional load bot script

```bash
python simulate_load.py
```

---

## URLs

- Student login: `http://127.0.0.1:8080/static/index.html`
- Student app: `http://127.0.0.1:8080/static/app.html`
- Admin dashboard: `http://127.0.0.1:8080/admin`
- Spectrum viewer: `http://127.0.0.1:8080/static/spectrum.html`
- OpenAPI docs: `http://127.0.0.1:8080/docs`

---

## Configuration

Environment variables:

- `COGNIRAD_ADMIN_PASSWORD` (default: `admin`)
- `COGNIRAD_HOST` (default: `127.0.0.1`)
- `COGNIRAD_PORT` (default: `8080`)

---

## API Reference

### Auth

- `POST /auth/login`
  - body: `{ "cms_id": "..."} ` or `{ "cms": "..." }`
  - returns token + student/channel metadata

- `POST /logout`
  - body: `{ "token": "..." }`

### Channels

- `GET /channel/state`
- `POST /channel/join?token=...`
- `POST /channel/message`
- `GET /channel/{id}/messages`
- `GET /channel/{id}/members`
- `GET /students?token=...`

### Admin

- `POST /admin/verify`
- `POST /admin/jam`
- `POST /admin/unjam`
- `POST /admin/reallocate`
- `POST /admin/simulate_load`
- `GET /admin/students?admin_key=...`

### WebSocket

- `ws://host:port/ws/{token}` (student realtime DM)
- `ws://host:port/ws/spectrum` (admin telemetry stream)

---

## API Examples (curl)

Set a base URL:

```bash
BASE="http://127.0.0.1:8080"
```

### 1) Login

```bash
curl -s -X POST "$BASE/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"cms":"458214"}'
```

Save token from response:

```bash
TOKEN="paste_token_here"
```

### 2) Join channel

```bash
curl -s -X POST "$BASE/channel/join?token=$TOKEN"
```

### 3) List students (authenticated)

```bash
curl -s "$BASE/students?token=$TOKEN"
```

### 4) Get channel state

```bash
curl -s "$BASE/channel/state"
```

### 5) Send DM (REST fallback)

```bash
curl -s -X POST "$BASE/channel/message" \
  -H "Content-Type: application/json" \
  -d "{\"token\":\"$TOKEN\",\"to\":\"481862\",\"text\":\"Hello from curl\"}"
```

### 6) Verify admin access

```bash
curl -s -X POST "$BASE/admin/verify" \
  -H "Content-Type: application/json" \
  -d '{"admin_key":"admin"}'
```

### 7) Force jam a channel

```bash
curl -s -X POST "$BASE/admin/jam" \
  -H "Content-Type: application/json" \
  -d '{"channel_key":"CH-1","admin_key":"admin"}'
```

### 8) Unjam a channel

```bash
curl -s -X POST "$BASE/admin/unjam" \
  -H "Content-Type: application/json" \
  -d '{"channel_key":"CH-1","admin_key":"admin"}'
```

### 9) Trigger manual reallocation

```bash
curl -s -X POST "$BASE/admin/reallocate" \
  -H "Content-Type: application/json" \
  -d '{"channel_key":"CH-2","admin_key":"admin"}'
```

### 10) Simulate load

```bash
curl -s -X POST "$BASE/admin/simulate_load" \
  -H "Content-Type: application/json" \
  -d '{"admin_key":"admin","energy_per_student":8}'
```

---

## Important Event Types

Student websocket receives:

- `CONNECTED`
- `DM`
- `MESSAGE_RESULT`
- `REALLOCATED`
- `SYSTEM` (`CHANNEL_JAMMED`, `CHANNEL_REBALANCED`, etc.)
- `ERROR`

Spectrum websocket receives:

- `SPECTRUM_HISTORY` (on connect)
- `SPECTRUM_TICK` (periodic live updates)

---

## Deployment Guide

### Option A: Uvicorn (simple production-ish start)

```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1
```

Notes:
- Keep workers at `1` unless you redesign shared in-memory state (`channels.py`, live connections, in-memory energy maps) for multi-process coordination.
- If multiple workers are required, use shared state (Redis/pubsub + centralized state) before scaling out.

### Option B: Gunicorn + Uvicorn worker

```bash
gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8080 \
  --workers 1 \
  --timeout 60
```

### Reverse proxy (Nginx)

Use Nginx in front for TLS and stable websocket proxying.

Minimal location example:

```nginx
location / {
    proxy_pass http://127.0.0.1:8080;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

### Environment variables for deployment

```bash
export COGNIRAD_ADMIN_PASSWORD="change-me"
export COGNIRAD_HOST="your-server-hostname"
export COGNIRAD_PORT="8080"
```

On Windows PowerShell:

```powershell
$env:COGNIRAD_ADMIN_PASSWORD="change-me"
$env:COGNIRAD_HOST="your-server-hostname"
$env:COGNIRAD_PORT="8080"
```

### Recommended hardening checklist

- Change default admin password.
- Restrict CORS origins (current backend allows all origins).
- Serve behind HTTPS.
- Add request rate-limiting for auth/admin routes.
- Add centralized logging and process supervision.
- Back up `cognirad.db` if persistence matters in your environment.

---

## Testing

Run complete test suite:

```bash
pytest test_all.py --asyncio-mode=auto -v
```

Interactive test run:

```bash
python test_all.py
```

---

## Troubleshooting

### App stuck on loading screen

- Hard refresh browser (`Ctrl+F5`)
- Verify backend is running and reachable on `127.0.0.1:8080`
- Check browser console/network for:
  - `/students` response,
  - `/channel/state` response,
  - websocket handshake status.

### Login fails

- CMS must exist in `students.json`
- Re-run app so DB seed loads from `students.json`

### Messages rejected

- Sender may be on degraded/jammed channel
- Inspect `delivery_status` and `classification` in message result

### Admin actions fail

- Ensure `admin_key` matches `COGNIRAD_ADMIN_PASSWORD`

---

## Development Notes

- This repo may contain additional planning/docs artifacts under `planning/` and `documentation/`.
- `decompiled_main.py` is not the runtime entrypoint; use `main.py`.
- Frontend session state is kept in `sessionStorage`.
- Runtime DB file is `cognirad.db` in project root.

---

## License

Use and distribution policy depends on your course/team context.  
Add your preferred license file (`MIT`, `Apache-2.0`, etc.) if needed.

