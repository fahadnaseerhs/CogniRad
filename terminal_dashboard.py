"""
terminal_dashboard.py — CogniRad | Live Terminal Admin Portal
=============================================================
Renders a self-refreshing ASCII dashboard in the terminal every 4 seconds.

Features
--------
  • Startup banner with server URL in red
  • Live message feed  (sender → recipient, channel, energy, status)
  • Per-channel energy progress bars with status colour
  • Active-student energy progress bars
  • All rendering uses only stdlib + colorama (already installed)

Usage
-----
  Called from main.py lifespan:
      from terminal_dashboard import start_dashboard, record_message
      start_dashboard(host, port)          # starts background task
      record_message(sender, recipient, channel, energy, status)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import sys
from collections import deque
from typing import Any

import colorama
from colorama import Fore, Back, Style

colorama.init(autoreset=True)

# ── tunables ────────────────────────────────────────────────────────────────
REFRESH_INTERVAL   = 4          # seconds between full redraws
BAR_WIDTH          = 28         # character width of progress bars
MAX_MSG_FEED       = 8          # how many recent messages to show
ENERGY_SCALE_MAX   = 15.0       # energy value that fills the bar to 100 %

# ── shared state (written by process_message, read by dashboard) ────────────
_message_feed: deque[dict[str, Any]] = deque(maxlen=MAX_MSG_FEED)
_server_url: str = "http://127.0.0.1:8000"


# ── public API ───────────────────────────────────────────────────────────────

def set_server_url(host: str, port: int) -> None:
    """Called once at startup to store the URL for the banner."""
    global _server_url
    _server_url = f"http://{host}:{port}"


def record_message(
    sender: str,
    recipient: str,
    channel: str,
    energy: float,
    status: str,
    delivery: str,
) -> None:
    """
    Append one delivered/rejected message to the live feed.
    Called from process_message() in main.py after every DM attempt.
    """
    _message_feed.appendleft({
        "time":      dt.datetime.now().strftime("%H:%M:%S"),
        "sender":    sender,
        "recipient": recipient,
        "channel":   channel,
        "energy":    energy,
        "status":    status,
        "delivery":  delivery,
    })


def start_dashboard() -> None:
    """
    Schedule the dashboard refresh loop as an asyncio background task.
    Must be called from inside a running event loop (e.g. FastAPI lifespan).
    """
    asyncio.create_task(_dashboard_loop())


# ── internal helpers ─────────────────────────────────────────────────────────

def _clear() -> None:
    """Clear the terminal screen cross-platform."""
    os.system("cls" if sys.platform == "win32" else "clear")


def _bar(value: float, max_val: float, width: int, colour: str) -> str:
    """
    Render a filled ASCII progress bar.

    Example:  ████████████░░░░░░░░  60 %
    """
    ratio    = min(value / max_val, 1.0) if max_val > 0 else 0.0
    filled   = int(ratio * width)
    empty    = width - filled
    pct      = ratio * 100
    bar_body = "█" * filled + "░" * empty
    return f"{colour}{bar_body}{Style.RESET_ALL} {pct:5.1f}%"


def _status_colour(status: str) -> str:
    return {
        "FREE":      Fore.GREEN,
        "BUSY":      Fore.YELLOW,
        "CONGESTED": Fore.RED,
        "JAMMED":    Fore.MAGENTA,
    }.get(status, Fore.WHITE)


def _delivery_colour(delivery: str) -> str:
    if "REJECT" in delivery:
        return Fore.RED
    if "DEGRADED" in delivery:
        return Fore.YELLOW
    if "STABILIZATION" in delivery:
        return Fore.CYAN
    if "OFFLINE" in delivery:
        return Fore.LIGHTBLACK_EX
    return Fore.GREEN


def _divider(char: str = "─", width: int = 72) -> str:
    return Fore.LIGHTBLACK_EX + char * width + Style.RESET_ALL


def _render() -> None:
    """Build and print the full dashboard frame."""
    # Import here to avoid circular import at module load time
    import channels as ch_mod
    import signal_physics as sp
    import main as main_mod

    _clear()

    now_str = dt.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    # ── Banner ────────────────────────────────────────────────────────────
    print()
    print(
        Fore.CYAN + Style.BRIGHT +
        "  ██████╗ ██████╗  ██████╗ ███╗  ██╗██╗██████╗  █████╗ ██████╗ " +
        Style.RESET_ALL
    )
    print(
        Fore.CYAN + Style.BRIGHT +
        " ██╔════╝██╔═══██╗██╔════╝ ████╗ ██║██║██╔══██╗██╔══██╗██╔══██╗" +
        Style.RESET_ALL
    )
    print(
        Fore.CYAN + Style.BRIGHT +
        " ██║     ██║   ██║██║  ███╗██╔██╗██║██║██████╔╝███████║██║  ██║" +
        Style.RESET_ALL
    )
    print(
        Fore.CYAN + Style.BRIGHT +
        " ██║     ██║   ██║██║   ██║██║╚████║██║██╔══██╗██╔══██║██║  ██║" +
        Style.RESET_ALL
    )
    print(
        Fore.CYAN + Style.BRIGHT +
        " ╚██████╗╚██████╔╝╚██████╔╝██║ ╚███║██║██║  ██║██║  ██║██████╔╝" +
        Style.RESET_ALL
    )
    print(
        Fore.CYAN + Style.BRIGHT +
        "  ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚══╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ " +
        Style.RESET_ALL
    )
    print()
    print(
        "  " + Fore.WHITE + Style.BRIGHT + "Cognitive Radio Spectrum Manager" +
        Style.RESET_ALL +
        Fore.LIGHTBLACK_EX + f"   {now_str}" + Style.RESET_ALL
    )
    print(
        "  Server  " +
        Fore.RED + Style.BRIGHT + _server_url + Style.RESET_ALL +
        Fore.LIGHTBLACK_EX + "   (Ctrl+C to stop)" + Style.RESET_ALL
    )
    print(_divider("═"))

    # ── Channel energy bars ───────────────────────────────────────────────
    print(
        Fore.WHITE + Style.BRIGHT +
        "  CHANNELS — Energy / Status" +
        Style.RESET_ALL
    )
    print(_divider())

    for ch_key, ch_data in ch_mod.CHANNELS.items():
        status   = ch_data.get("status", "FREE")
        sc       = _status_colour(status)
        snap     = sp.get_channel_energy_snapshot(ch_key)
        energy   = snap["total_energy"]
        members  = snap["member_count"]
        snr      = snap["snr_db"]
        mod      = snap["modulation"]
        bar      = _bar(energy, ENERGY_SCALE_MAX, BAR_WIDTH, sc)
        freq     = ch_data.get("frequency", "")

        status_badge = sc + Style.BRIGHT + f"{status:<10}" + Style.RESET_ALL
        print(
            f"  {Fore.WHITE + Style.BRIGHT}{ch_key}{Style.RESET_ALL}"
            f"  {bar}"
            f"  {Fore.CYAN}{energy:6.2f}J{Style.RESET_ALL}"
            f"  {status_badge}"
            f"  {Fore.LIGHTBLACK_EX}{members}u  SNR {snr:.1f}dB  {mod}  {freq}{Style.RESET_ALL}"
        )

    print(_divider())

    # ── Active students ───────────────────────────────────────────────────
    online_cms = list(main_mod.manager._connections.keys())

    print(
        Fore.WHITE + Style.BRIGHT +
        f"  ACTIVE STUDENTS ({len(online_cms)} online)" +
        Style.RESET_ALL
    )
    print(_divider())

    if not online_cms:
        print(
            Fore.LIGHTBLACK_EX +
            "  No students currently connected." +
            Style.RESET_ALL
        )
    else:
        for cms in online_cms:
            energy   = sp.get_energy_score(cms)
            ch_key   = ch_mod.find_student_channel(cms)
            ch_label = ch_key if ch_key else "—"

            # Colour the bar by energy level
            if energy >= ENERGY_SCALE_MAX * 0.8:
                bar_colour = Fore.RED
            elif energy >= ENERGY_SCALE_MAX * 0.5:
                bar_colour = Fore.YELLOW
            else:
                bar_colour = Fore.GREEN

            bar = _bar(energy, ENERGY_SCALE_MAX, BAR_WIDTH, bar_colour)
            print(
                f"  {Fore.WHITE + Style.BRIGHT}{cms:<10}{Style.RESET_ALL}"
                f"  {bar}"
                f"  {Fore.CYAN}{energy:6.2f}J{Style.RESET_ALL}"
                f"  {Fore.LIGHTBLACK_EX}on {ch_label}{Style.RESET_ALL}"
            )

    print(_divider())

    # ── Live message feed ─────────────────────────────────────────────────
    print(
        Fore.WHITE + Style.BRIGHT +
        "  LIVE MESSAGE FEED  (last 8)" +
        Style.RESET_ALL
    )
    print(_divider())

    if not _message_feed:
        print(
            Fore.LIGHTBLACK_EX +
            "  No messages yet." +
            Style.RESET_ALL
        )
    else:
        for msg in _message_feed:
            dc       = _delivery_colour(msg["delivery"])
            sc       = _status_colour(msg["status"])
            delivery = msg["delivery"].replace("_", " ")
            print(
                f"  {Fore.LIGHTBLACK_EX}{msg['time']}{Style.RESET_ALL}"
                f"  {Fore.WHITE + Style.BRIGHT}{msg['sender']:<10}{Style.RESET_ALL}"
                f"  {Fore.LIGHTBLACK_EX}→{Style.RESET_ALL}"
                f"  {Fore.WHITE}{msg['recipient']:<10}{Style.RESET_ALL}"
                f"  {Fore.LIGHTBLACK_EX}via {msg['channel']:<5}{Style.RESET_ALL}"
                f"  {Fore.CYAN}{msg['energy']:5.2f}J{Style.RESET_ALL}"
                f"  {sc}{msg['status']:<10}{Style.RESET_ALL}"
                f"  {dc}{delivery}{Style.RESET_ALL}"
            )

    print(_divider("═"))
    print(
        Fore.LIGHTBLACK_EX +
        f"  Refreshes every {REFRESH_INTERVAL}s  │  "
        f"Docs: {_server_url}/docs  │  "
        f"API: {_server_url}/channel/state" +
        Style.RESET_ALL
    )
    print()


async def _dashboard_loop() -> None:
    """Background task: wait 4 s then redraw the dashboard forever."""
    # Small initial delay so uvicorn's own startup messages print first
    await asyncio.sleep(2)
    while True:
        try:
            _render()
        except Exception:
            # Never crash the server because of a display error
            pass
        await asyncio.sleep(REFRESH_INTERVAL)
