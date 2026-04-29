"""
main.py — CogniRad | FastAPI Orchestration Layer
=================================================
DM-only cognitive-radio backend.  Channels are shared frequency bands;
students DM each other over the band.  The AI watches cumulative channel
energy and triggers fair, minimum-move reallocation when overloaded.

Start with:
    uvicorn main:app --reload

REST Endpoints
--------------
POST   /login                   Student login → token
POST   /logout                  Invalidate token
GET    /channel/state           All channel states
POST   /channel/join            Assign student to best channel
POST   /channel/message         Send a DM (REST fallback)
GET    /channel/{id}/messages   Fetch recent messages for a channel
GET    /channel/{id}/members    List members on a channel
POST   /admin/jam               Force a channel to JAMMED
POST   /admin/unjam             Clear JAMMED status
GET    /admin/students          List all students
POST   /admin/reallocate        Trigger manual reallocation

WebSocket
---------
/ws/{token}     Authenticated WebSocket for real-time DMs.
                Send:   {"to": "CMS007", "text": "Hey"}
                Recv:   {type: DM|MESSAGE_RESULT|REALLOCATED|CHANNEL_JAMMED|...}
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as dt
import json
import logging
import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import allocator
import auth
import channels as ch_mod
import classifier
import database
import signal_physics as sp
import terminal_dashboard as dashboard

logger = logging.getLogger("cognirad")


# ═══════════════════════════════════════════════════════════════════════════
#  App bootstrap
# ═══════════════════════════════════════════════════════════════════════════

@contextlib.asynccontextmanager
async def _lifespan(application: FastAPI):
    await database.init_db()
    # Start the background AI loop
    task = asyncio.create_task(_ai_loop())
    # Start the terminal dashboard (reads host/port from uvicorn config)
    _host = os.environ.get("COGNIRAD_HOST", "127.0.0.1")
    _port = int(os.environ.get("COGNIRAD_PORT", "8000"))
    dashboard.set_server_url(_host, _port)
    dashboard.start_dashboard()
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(
    title="CogniRad Spectrum Management API",
    version="2.0.0",
    description=(
        "AI-driven Wi-Fi channel allocation with DM routing, "
        "cumulative energy tracking, and fair reallocation."
    ),
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


# ═══════════════════════════════════════════════════════════════════════════
#  Connection Manager — DM-aware WebSocket hub
# ═══════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """
    Manages authenticated WebSocket connections.

    Key change from v1: ``send_dm()`` replaces ``broadcast_to_channel()``
    as the primary message delivery path.  ``broadcast_to_channel()`` is
    kept *only* for system notices (REALLOCATED, CHANNEL_JAMMED, etc.).
    """

    def __init__(self) -> None:
        # cms → WebSocket
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, cms: str, ws: WebSocket) -> None:
        await ws.accept()
        # Disconnect any existing socket for this CMS (single-session)
        old = self._connections.pop(cms, None)
        if old:
            with contextlib.suppress(Exception):
                await old.close(code=4001, reason="New session opened")
        self._connections[cms] = ws

    def disconnect(self, cms: str) -> None:
        self._connections.pop(cms, None)

    def is_online(self, cms: str) -> bool:
        return cms in self._connections

    async def send_dm(
        self,
        sender_cms: str,
        recipient_cms: str,
        payload: dict[str, Any],
    ) -> bool:
        """
        Deliver a DM payload to *recipient_cms*.
        Returns True if the recipient has an active WebSocket.
        """
        ws = self._connections.get(recipient_cms)
        if ws is None:
            return False
        try:
            await ws.send_json(payload)
            return True
        except Exception:
            self.disconnect(recipient_cms)
            return False

    async def send_to(self, cms: str, payload: dict[str, Any]) -> bool:
        """Send a payload to a specific user (for confirmations, errors)."""
        ws = self._connections.get(cms)
        if ws is None:
            return False
        try:
            await ws.send_json(payload)
            return True
        except Exception:
            self.disconnect(cms)
            return False

    async def broadcast_to_channel(
        self,
        channel_key: str,
        payload: dict[str, Any],
    ) -> None:
        """
        Broadcast a system notice to everyone on *channel_key*.
        Used ONLY for: REALLOCATED, CHANNEL_JAMMED, CHANNEL_FROZEN.
        NOT for regular DMs.
        """
        members = ch_mod.get_channel_members(channel_key)
        dead: list[str] = []
        for cms in members:
            ws = self._connections.get(cms)
            if ws is None:
                continue
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(cms)
        for cms in dead:
            self.disconnect(cms)

    def get_reachable_peers(self, cms: str) -> list[str]:
        """Return online peers on the same channel as *cms*."""
        ch_key = ch_mod.find_student_channel(cms)
        if ch_key is None:
            return []
        members = ch_mod.get_channel_members(ch_key)
        return [
            m for m in members
            if m != cms and m in self._connections
        ]


manager = ConnectionManager()


# ═══════════════════════════════════════════════════════════════════════════
#  Request / Response schemas
# ═══════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    cms_id: str | None = None
    cms: str | None = None

    def resolved_cms(self) -> str:
        value = (self.cms_id or self.cms or "").strip()
        if not value:
            raise HTTPException(status_code=422, detail="CMS ID is required.")
        return value

class LoginResponse(BaseModel):
    token: str
    cms: str
    student_name: str | None = None
    channel_id: str | None = None
    channel_freq: str | None = None
    channel_status: str | None = None

class LogoutRequest(BaseModel):
    token: str

class SendDMRequest(BaseModel):
    token: str
    to: str
    text: str

class JamRequest(BaseModel):
    channel_key: str
    admin_key: str = "admin"

class ReallocateRequest(BaseModel):
    channel_key: str
    admin_key: str = "admin"

ADMIN_KEY = "admin"

def _check_admin(key: str) -> None:
    if key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")


# ═══════════════════════════════════════════════════════════════════════════
#  DM Processing Core
# ═══════════════════════════════════════════════════════════════════════════

async def process_message(
    sender_cms: str,
    sender_name: str,
    recipient_cms: str,
    text: str,
    channel_key: str,
) -> dict[str, Any]:
    """
    End-to-end DM processing pipeline:

    1. Capture one shared timestamp (message_now) for the whole request.
    2. Compute message energy from text.
    3. Decay sender energy, then add message energy.
    4. Classify channel using message_now as the decay baseline.
    5. If healthy → save DM, deliver to recipient, confirm to sender.
    6. If overloaded:
       a. Attempt reallocation (passes message_now so ranking/projection
          share the same decay baseline as the initial classify).
       b. Reclassify with a fresh post_now — topology has changed so we
          want the real post-move state, not a stale snapshot.
       c. Apply delivery policy on post-move classification.
    7. Build both sender MESSAGE_RESULT and recipient DM from the same
       canonical final_result so both clients see consistent metadata.

    Returns a MESSAGE_RESULT dict for the sender.
    """
    channel_data = ch_mod.CHANNELS[channel_key]
    db_channel_id = int(channel_key.split("-")[1])
    sender_channel_before = ch_mod.find_student_channel(sender_cms)
    recipient_channel_before = ch_mod.find_student_channel(recipient_cms)

    # ── 0. Capture one shared timestamp for the decision path ──────
    # Every decay, classify, and allocator call up to (and including)
    # the reallocation uses this same instant so no step can cross a
    # decay-tick boundary mid-request and produce inconsistent results.
    message_now = time.time()

    # ── 1. Compute message energy ──────────────────────────────────
    n_users = len(channel_data["users"])
    msg_energy = sp.compute_message_energy(
        text, channel_key, concurrent_transmitters=n_users,
    )

    # ── 2. Update sender cumulative energy ─────────────────────────
    # Decay to message_now first so the new energy is added on top of
    # the current decayed baseline, not a stale accumulated total.
    sp.apply_decay_to_student(sender_cms, now=message_now)
    new_total = sp.update_energy_score(sender_cms, msg_energy)

    # ── 3-4. Classify channel (using message_now) ──────────────────
    result = classifier.classify_channel(channel_key, now=message_now)
    channel_data["status"] = result["status"]
    channel_data["message_rate"] += 1

    # Sync status to DB
    await database.update_channel_status(
        db_channel_id,
        result["status"],
        result["confidence"],
        is_jammed=(result["status"] == "JAMMED"),
    )

    # ── 5-6. Decide outcome ────────────────────────────────────────
    is_ok = classifier.is_healthy(result)
    accepted = is_ok
    warning = None
    reallocation_info = None
    delivery_status = "DELIVERED"
    # final_result starts as the initial classification; overwritten below
    # if overload handling runs so both sender and recipient always see
    # the same canonical post-decision channel metadata.
    final_result = result

    if not is_ok:
        # Channel is overloaded.  Pass message_now to the allocator so
        # source ranking and destination projections share the same decay
        # baseline as the initial classify above.
        #
        # Delivery policy:
        #   FREE / BUSY  after decision → DELIVERED_AFTER_STABILIZATION
        #   CONGESTED    after decision → DELIVERED_CHANNEL_DEGRADED
        #   JAMMED       after decision → REJECTED_CHANNEL_JAMMED
        warning = (
            f"Channel {channel_key} is {result['status']} "
            f"(confidence={result['confidence']:.2f}). "
            "Attempting reallocation."
        )
        moved = await allocator.reallocate_users(channel_key, now=message_now)
        if moved:
            reallocation_info = moved
            # Notify affected users of their new channel assignment
            for move in moved:
                await manager.broadcast_to_channel(
                    move["to"],
                    {
                        "type": "SYSTEM",
                        "subtype": "NEW_MEMBER",
                        "cms": move["cms"],
                        "channel_key": move["to"],
                    },
                )
                await manager.send_to(
                    move["cms"],
                    {
                        "type": "REALLOCATED",
                        "from": move["from"],
                        "to": move["to"],
                        "frequency": move["frequency"],
                        "decayed_energy": move.get("decayed_energy"),
                    },
                )

        # Reclassify with a fresh timestamp after moves complete.
        # We intentionally use a new post_now here (not message_now)
        # because the channel topology has actually changed — users have
        # moved — so we want the real current state, not a snapshot
        # frozen at the moment the request arrived.
        post_now = time.time()
        post_result = classifier.classify_channel(channel_key, now=post_now)
        channel_data["status"] = post_result["status"]
        await database.update_channel_status(
            db_channel_id,
            post_result["status"],
            post_result["confidence"],
            is_jammed=(post_result["status"] == "JAMMED"),
        )

        # Promote post_result to canonical so both sender and recipient
        # see the same stabilised channel metadata.
        final_result = post_result

        # Apply delivery policy based on post-decision channel state
        if post_result["status"] == "JAMMED":
            # Channel is still jammed — reject the message entirely
            accepted = False
            delivery_status = "REJECTED_CHANNEL_JAMMED"
            warning = (
                f"Channel {channel_key} remains JAMMED after reallocation attempt. "
                "Message rejected."
            )
        elif post_result["status"] == "CONGESTED":
            # Still congested but not jammed — deliver with warning
            accepted = True
            delivery_status = "DELIVERED_CHANNEL_DEGRADED"
            warning = (
                f"Channel {channel_key} is still CONGESTED after reallocation. "
                "Message delivered under degraded conditions."
            )
        else:
            # Channel recovered to FREE or BUSY — full delivery
            accepted = True
            delivery_status = "DELIVERED_AFTER_STABILIZATION"
            warning = None

    # ── Save and deliver if accepted ───────────────────────────────
    delivered_at = None
    if accepted:
        delivered_at = dt.datetime.utcnow()
        recipient_online = manager.is_online(recipient_cms)
        await database.save_message(
            channel_id=db_channel_id,
            cms=sender_cms,
            student_name=sender_name,
            content=text,
            recipient_cms=recipient_cms,
            message_type="DM",
            delivered_at=delivered_at,
        )

        # Deliver DM to recipient via WebSocket.
        # Use final_result for signal metadata so recipient and sender
        # both see the same canonical post-decision channel state.
        await manager.send_dm(
            sender_cms,
            recipient_cms,
            {
                "type": "DM",
                "from": sender_cms,
                "from_name": sender_name,
                "to": recipient_cms,
                "text": text,
                "channel_key": channel_key,
                "sender_channel": sender_channel_before,
                "recipient_channel": recipient_channel_before,
                "route_type": (
                    "same-channel"
                    if sender_channel_before and sender_channel_before == recipient_channel_before
                    else "cross-channel"
                ),
                "timestamp": delivered_at.isoformat(),
                "signal": {
                    "energy": msg_energy,
                    "sender_total_energy": new_total,
                    # Use final_result here — same object used for the
                    # sender response — so both clients are consistent.
                    "channel_status": final_result["status"],
                    "snr_db": final_result["snr_db"],
                    "modulation": final_result["modulation"],
                },
            },
        )
        if not recipient_online:
            delivery_status = "STORED_RECIPIENT_OFFLINE"
            warning = "Recipient is offline. Message was stored but not delivered live."

    # ── Build result for sender ────────────────────────────────────
    # final_result is the canonical post-decision classification.
    # Both this dict and the recipient DM above use it, so sender and
    # recipient always see the same channel health metadata.

    # Feed the live terminal dashboard with this message event.
    dashboard.record_message(
        sender=sender_cms,
        recipient=recipient_cms,
        channel=channel_key,
        energy=msg_energy,
        status=final_result["status"],
        delivery=delivery_status,
    )

    return {
        "type": "MESSAGE_RESULT",
        "accepted": accepted,
        "to": recipient_cms,
        "text": text,
        "delivery_status": delivery_status,
        "sender_channel": sender_channel_before,
        "recipient_channel": recipient_channel_before,
        "route_type": (
            "same-channel"
            if sender_channel_before and sender_channel_before == recipient_channel_before
            else "cross-channel"
        ),
        "energy": msg_energy,
        "sender_total_energy": new_total,
        "classification": {
            "status": final_result["status"],
            "confidence": final_result["confidence"],
            "snr_db": final_result["snr_db"],
            "total_energy": final_result["total_energy"],
            "modulation": final_result["modulation"],
        },
        "warning": warning,
        "reallocation": reallocation_info,
        "timestamp": delivered_at.isoformat() if delivered_at else None,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  AI Background Loop
# ═══════════════════════════════════════════════════════════════════════════

async def _ai_loop() -> None:
    """
    Every 5 seconds, run the three-phase AI cycle:

    Phase 2 addition — Decay: apply idle energy decay to all students
    before observing channel state.  This ensures classification and
    reallocation decisions are based on time-accurate decayed energy,
    not permanently accumulated totals.

    Phase 1 — Observation: classify every channel using decayed energy.
    Phase 2 — Decision: reallocate minimum students from channels that
    are still overloaded *after* decay has had a chance to reduce load.

    Order matters:
        1. apply_idle_decay()          ← Phase 2: decay first
        2. classify all channels       ← observe decayed state
        3. reallocate overloaded ones  ← only if decay wasn't enough
    """
    while True:
        await asyncio.sleep(5)
        try:
            # Capture one shared observation timestamp for the entire
            # AI cycle.  Every decay call, snapshot, classification, and
            # allocator projection in this tick will use this same instant
            # so the whole cycle is internally consistent.
            observation_now = time.time()

            # ── Phase 2: Apply idle decay before observation ───────
            # Decay all student energies so that inactive users cool
            # down naturally.  Channels may recover from CONGESTED or
            # BUSY to FREE without any reallocation if the load has
            # simply gone quiet.
            decay_summary = sp.apply_idle_decay(now=observation_now)
            if decay_summary["decayed_count"] > 0:
                logger.debug(
                    "Idle decay applied: %d students decayed, "
                    "total energy %.2f → %.2f",
                    decay_summary["decayed_count"],
                    decay_summary["before_total"],
                    decay_summary["after_total"],
                )

            # ── Phase 1: Observation ───────────────────────────────
            # Classify every channel using the freshly decayed energy
            # snapshots.  Pass the shared observation_now so all member
            # decays within each snapshot use the same instant — no
            # snapshot can span a tick boundary mid-loop.
            classifications: dict[str, classifier.ClassificationResult] = {}
            for ch_key in ch_mod.CHANNELS:
                admin_jammed = ch_mod.CHANNELS[ch_key]["status"] == "JAMMED"
                result = classifier.classify_channel(
                    ch_key, admin_jammed=admin_jammed, now=observation_now
                )

                # Only update status if not admin-jammed (preserve admin control)
                if not admin_jammed:
                    ch_mod.CHANNELS[ch_key]["status"] = result["status"]

                classifications[ch_key] = result

                # Sync to DB
                db_id = int(ch_key.split("-")[1])
                await database.update_channel_status(
                    db_id,
                    result["status"],
                    result["confidence"],
                    is_jammed=admin_jammed,
                )

            # ── Phase 2: Decision ──────────────────────────────────
            # Only reallocate channels that are still overloaded after
            # decay.  If idle cooldown already brought a channel back
            # to FREE or BUSY, no reallocation is needed — this is the
            # key Phase 2 behaviour: decay suppresses unnecessary moves.
            # Pass the shared observation_now so the allocator's source
            # ranking and destination projections share the same baseline.
            for ch_key, result in classifications.items():
                if not classifier.is_healthy(result) and result["member_count"] > 0:
                    moved = await allocator.reallocate_users(ch_key, now=observation_now)

                    # Notify all affected users
                    for move in moved:
                        await manager.send_to(
                            move["cms"],
                            {
                                "type": "REALLOCATED",
                                "from": move["from"],
                                "to": move["to"],
                                "frequency": move["frequency"],
                                "decayed_energy": move.get("decayed_energy"),
                            },
                        )

                    if moved:
                        await manager.broadcast_to_channel(
                            ch_key,
                            {
                                "type": "SYSTEM",
                                "subtype": "CHANNEL_REBALANCED",
                                "channel_key": ch_key,
                                "moved_count": len(moved),
                            },
                        )

        except Exception:
            logger.exception("AI loop error")


async def _cli_dashboard() -> None:
    """
    Prints a mini admin portal on the terminal every 4 seconds with:
    - Server link in red
    - Channels energy progress bar
    - Top active students progress bar
    - Recent DMs
    """
    while True:
        await asyncio.sleep(4)
        try:
            lines = [
                "",
                "\033[91m" + "="*60 + "\033[0m",
                "\033[91m🚀 CogniRad Server Running -> http://localhost:8000\033[0m",
                "\033[91m" + "="*60 + "\033[0m"
            ]
            
            # Channels
            lines.append("\n\033[1m📡 CHANNELS ENERGY\033[0m")
            for ch_key, data in ch_mod.CHANNELS.items():
                energy_snap = sp.get_channel_energy_snapshot(ch_key)
                total = energy_snap["total_energy"]
                status = data["status"]
                
                bar_len = 20
                filled = int(min(total / 15.0, 1.0) * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                
                if status in ["FREE", "BUSY"]:
                    color = "\033[92m" # Green
                elif status == "CONGESTED":
                    color = "\033[93m" # Yellow
                else:
                    color = "\033[91m" # Red
                    
                lines.append(f"  {ch_key} {color}{bar}\033[0m {total:5.1f}e ({status})")

            # Students
            lines.append("\n\033[1m🎓 TOP ACTIVE STUDENTS\033[0m")
            all_students = []
            with sp._energy_lock:
                for cms, ts in sp._energy_timestamps.items():
                    energy = sp._energy_scores.get(cms, 0.0)
                    if energy > 0:
                        all_students.append((cms, energy))
            all_students.sort(key=lambda x: x[1], reverse=True)
            if not all_students:
                lines.append("  (No active students)")
            for cms, energy in all_students[:5]:
                bar_len = 20
                filled = int(min(energy / 10.0, 1.0) * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                lines.append(f"  {cms} \033[96m{bar}\033[0m {energy:5.1f}e")

            # Messages
            lines.append("\n\033[1m✉️  RECENT MESSAGES\033[0m")
            if not RECENT_DMS:
                lines.append("  (No recent messages)")
            for msg in list(RECENT_DMS):
                color = "\033[92m" if msg["status"] == "OK" else "\033[91m"
                text_len = min(15, len(str(msg.get('text', ''))))
                lines.append(f"  {msg['from']} -> {msg['to']} [{msg['ch']}] {color}{msg['status']}\033[0m (+{msg['energy']:.1f}e)")
                
            lines.append("\n" + "-"*60 + "\n")
            
            # Clear screen and print dashboard
            print("\033[2J\033[H", end="")
            print("\n".join(lines))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  Auth endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/auth/login", response_model=LoginResponse, tags=["Auth"])
async def login(body: LoginRequest):
    cms_value = body.resolved_cms()
    try:
        token = await auth.login_student(cms_value)
    except auth.AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
        
    db_student = await database.get_student_by_cms(cms_value)
    name = db_student.name if db_student else cms_value
    
    ch_key = ch_mod.find_student_channel(cms_value)
    freq = ch_mod.CHANNELS[ch_key]["frequency"] if ch_key else None
    status_str = ch_mod.CHANNELS[ch_key]["status"] if ch_key else None
    
    return LoginResponse(
        token=token, 
        cms=cms_value,
        student_name=name,
        channel_id=ch_key,
        channel_freq=freq,
        channel_status=status_str
    )


@app.post("/logout", tags=["Auth"])
async def logout(body: LogoutRequest):
    # Reset energy on logout
    cms = await database.get_cms_from_token(body.token)
    if cms:
        sp.reset_energy_score(cms)
        # Remove from channel in-memory
        ch_key = ch_mod.find_student_channel(cms)
        if ch_key and cms in ch_mod.CHANNELS[ch_key]["users"]:
            ch_mod.CHANNELS[ch_key]["users"].remove(cms)
        manager.disconnect(cms)
    await auth.logout_student(body.token)
    return {"detail": "Logged out successfully."}


# ═══════════════════════════════════════════════════════════════════════════
#  Channel endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/channel/state", tags=["Channels"])
async def channel_state() -> dict[str, Any]:
    """Return the current state of all 5 channels with energy info."""
    states = {}
    for ch_id in ch_mod.CHANNELS:
        base = ch_mod.get_channel_status(ch_id)
        energy = sp.get_channel_energy_snapshot(ch_id)
        base["total_energy"] = energy["total_energy"]
        base["snr_db"] = energy["snr_db"]
        base["modulation"] = energy["modulation"]
        base["per_student_energy"] = energy["per_student"]
        states[ch_id] = base
    return {"channels": states}


@app.post("/channel/join", tags=["Channels"])
async def join_channel(token: str) -> dict[str, Any]:
    student = await _get_student(token)
    try:
        result = await allocator.assign_channel(student.cms)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return result


@app.get("/channel/{channel_id}/members", tags=["Channels"])
async def channel_members(channel_id: int) -> dict[str, Any]:
    ch_key = f"CH-{channel_id}"
    if ch_key not in ch_mod.CHANNELS:
        raise HTTPException(status_code=404, detail=f"Unknown channel: {ch_key}")
    members = ch_mod.get_channel_members(ch_key)
    return {
        "channel_key": ch_key,
        "members": members,
        "online": [m for m in members if manager.is_online(m)],
    }


@app.post("/channel/message", tags=["Channels"])
async def send_message_rest(body: SendDMRequest) -> dict[str, Any]:
    """REST fallback for sending a DM (same pipeline as WebSocket)."""
    student = await _get_student(body.token)

    # Validate recipient
    recipient = await database.get_student_by_cms(body.to)
    if recipient is None:
        raise HTTPException(status_code=404, detail=f"Recipient '{body.to}' not found.")

    sender_channel = ch_mod.find_student_channel(student.cms)
    if sender_channel is None:
        raise HTTPException(
            status_code=400,
            detail="You are not currently assigned to a channel.",
        )

    result = await process_message(
        sender_cms=student.cms,
        sender_name=student.name,
        recipient_cms=body.to,
        text=body.text,
        channel_key=sender_channel,
    )
    return result


@app.get("/channel/{channel_id}/messages", tags=["Channels"])
async def get_messages(channel_id: int, limit: int = 30) -> dict[str, Any]:
    messages = await database.get_recent_messages(channel_id, limit=limit)
    return {
        "channel_id": channel_id,
        "messages": [
            {
                "id": m.id,
                "cms": m.cms,
                "name": m.student_name,
                "content": m.content,
                "type": m.message_type,
                "recipient": m.recipient_cms,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                "delivered_at": m.delivered_at.isoformat() if m.delivered_at else None,
            }
            for m in messages
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Admin endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/admin/jam", tags=["Admin"])
async def jam_channel(body: JamRequest) -> dict[str, Any]:
    _check_admin(body.admin_key)
    if body.channel_key not in ch_mod.CHANNELS:
        raise HTTPException(status_code=404, detail=f"Unknown channel: {body.channel_key}")

    ch_mod.CHANNELS[body.channel_key]["status"] = "JAMMED"
    db_id = int(body.channel_key.split("-")[1])
    await database.update_channel_status(db_id, "JAMMED", 1.0, is_jammed=True)

    # Notify channel
    await manager.broadcast_to_channel(
        body.channel_key,
        {"type": "SYSTEM", "subtype": "CHANNEL_JAMMED", "channel_key": body.channel_key},
    )

    moved = await allocator.reallocate_users(body.channel_key)
    for move in moved:
        await manager.send_to(
            move["cms"],
            {
                "type": "REALLOCATED",
                "from": move["from"],
                "to": move["to"],
                "frequency": move["frequency"],
            },
        )
    return {"jammed": body.channel_key, "users_moved": moved}


@app.post("/admin/unjam", tags=["Admin"])
async def unjam_channel(body: JamRequest) -> dict[str, Any]:
    _check_admin(body.admin_key)
    if body.channel_key not in ch_mod.CHANNELS:
        raise HTTPException(status_code=404, detail=f"Unknown channel: {body.channel_key}")

    ch_mod.CHANNELS[body.channel_key]["status"] = "FREE"
    ch_mod.CHANNELS[body.channel_key]["rolling_jammed_score"] = 0.0
    ch_mod.CHANNELS[body.channel_key]["transmit_frozen"] = False
    db_id = int(body.channel_key.split("-")[1])
    await database.update_channel_status(db_id, "FREE", 0.0, is_jammed=False)
    return {"unjammed": body.channel_key}


@app.get("/students", tags=["Channels"])
async def get_students(token: str) -> dict[str, Any]:
    """Get all students (authenticated endpoint for non-admin users)."""
    # Authenticate with student token
    await _get_student(token)
    
    # Fetch all students
    students = await database.get_all_students()
    
    # Return same format as /admin/students for compatibility
    return {
        "students": [
            {
                "cms": s.cms,
                "name": s.name,
                "channel_id": s.channel_id,
                "channel_key": f"CH-{s.channel_id}" if s.channel_id else None,
                "is_active": s.is_active,
                "energy": sp.get_energy_score(s.cms),
                "joined_at": s.joined_at.isoformat() if s.joined_at else None,
            }
            for s in students
        ]
    }


@app.get("/admin/students", tags=["Admin"])
async def list_students() -> dict[str, Any]:
    students = await database.get_all_students()
    return {
        "students": [
            {
                "cms": s.cms,
                "name": s.name,
                "channel_id": s.channel_id,
                "channel_key": f"CH-{s.channel_id}" if s.channel_id else None,
                "is_active": s.is_active,
                "energy": sp.get_energy_score(s.cms),
                "joined_at": s.joined_at.isoformat() if s.joined_at else None,
            }
            for s in students
        ]
    }


@app.post("/admin/reallocate", tags=["Admin"])
async def admin_reallocate(body: ReallocateRequest) -> dict[str, Any]:
    _check_admin(body.admin_key)
    if body.channel_key not in ch_mod.CHANNELS:
        raise HTTPException(status_code=404, detail=f"Unknown channel: {body.channel_key}")

    moved = await allocator.reallocate_users(body.channel_key)
    return {"channel": body.channel_key, "moved": moved}


# ═══════════════════════════════════════════════════════════════════════════
#  WebSocket — authenticated DM hub
# ═══════════════════════════════════════════════════════════════════════════

async def _get_student(token: str) -> database.Student:
    """Dependency: resolve token → Student or raise 401."""
    try:
        return await auth.verify_token(token)
    except auth.AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc))


@app.websocket("/ws/{token}")
async def websocket_endpoint(ws: WebSocket, token: str):
    # Authenticate
    try:
        student = await auth.verify_token(token)
    except auth.AuthenticationError:
        await ws.close(code=4003, reason="Invalid token")
        return

    cms = student.cms
    await manager.connect(cms, ws)

    # Send initial state
    ch_key = ch_mod.find_student_channel(cms)
    await manager.send_to(cms, {
        "type": "CONNECTED",
        "cms": cms,
        "name": student.name,
        "channel_key": ch_key,
        "peers": manager.get_reachable_peers(cms) if ch_key else [],
        "all_online": [student_cms for student_cms in manager._connections if student_cms != cms],
    })

    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_to(cms, {
                    "type": "ERROR",
                    "detail": "Invalid JSON",
                })
                continue

            # ── Validate DM payload ────────────────────────────────
            to_cms = data.get("to")
            text = data.get("text")

            if not to_cms or not text:
                await manager.send_to(cms, {
                    "type": "ERROR",
                    "detail": 'Payload must contain "to" and "text" fields.',
                })
                continue

            # Validate recipient exists
            recipient = await database.get_student_by_cms(to_cms)
            if recipient is None:
                await manager.send_to(cms, {
                    "type": "ERROR",
                    "detail": f"Recipient '{to_cms}' does not exist.",
                })
                continue

            sender_channel = ch_mod.find_student_channel(cms)
            if sender_channel is None:
                await manager.send_to(cms, {
                    "type": "ERROR",
                    "detail": "You are not currently assigned to a channel.",
                })
                continue

            # ── Process the DM ─────────────────────────────────────
            result = await process_message(
                sender_cms=cms,
                sender_name=student.name,
                recipient_cms=to_cms,
                text=text,
                channel_key=sender_channel,
            )

            # Send result back to sender
            await manager.send_to(cms, result)

    except WebSocketDisconnect:
        manager.disconnect(cms)
    except Exception:
        manager.disconnect(cms)
        logger.exception(f"WebSocket error for {cms}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
