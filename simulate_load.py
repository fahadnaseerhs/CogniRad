import asyncio
import json
import random
import logging
from typing import List, Dict

import httpx
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

# Configuration
SERVER_URL = "http://127.0.0.1:8080"
NUM_STUDENTS_TO_LOGIN = 15
CHATTER_INTERVAL = 0.5  # seconds between baseline messages
BURST_INTERVAL = 20     # seconds between congestion bursts

# Short random messages for baseline chatter
CHITCHAT = [
    "Hey, how is your project going?",
    "Did you see the new assignment?",
    "I'm testing the cognitive radio system.",
    "Can anyone help me with allocator.py?",
    "Let's meet at the library later.",
    "Is the Wi-Fi slow for you guys too?",
    "Sending a small ping to check routing.",
    "All good here.",
    "Got it, thanks!",
]

# Large messages to force congestion (high energy)
HEAVY_PAYLOADS = [
    "This is a massive payload designed to simulate heavy file transfer over the network. " * 10,
    "Transmitting a large high-definition image block through the cognitive radio... " * 10,
    "Data dump initiated. Brace for high energy consumption on this channel! " * 10,
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Simulation")


class SimulationClient:
    def __init__(self):
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
        self.client = httpx.AsyncClient(base_url=SERVER_URL, limits=limits, timeout=10.0)
        self.active_students: List[Dict] = []
        self.tokens: Dict[str, str] = {}  # cms -> token

    async def fetch_and_login(self):
        """
        Login students and assign each one to a channel.

        Fix: after login, call POST /channel/join so the student is placed
        in the in-memory CHANNELS dict.  Without this, find_student_channel()
        returns None and every /channel/message call fails with 400.

        Uses the same deterministic first-N students (sorted by CMS ascending)
        that /admin/simulate_load assigns, so the two tools are compatible.
        """
        logger.info("Fetching available students from local database...")
        try:
            with open("students.json", "r") as f:
                data = json.load(f)
                # Sort numerically ascending — same order as /admin/simulate_load
                all_cms = sorted(data.keys(), key=lambda x: int(x) if x.isdigit() else 10**12)
        except Exception as e:
            logger.error(f"Could not load students.json: {e}")
            return False

        selected_cms = all_cms[:NUM_STUDENTS_TO_LOGIN]
        logger.info(f"Logging in {len(selected_cms)} students (sorted by CMS ascending)...")

        for cms in selected_cms:
            try:
                # ── Step 1: Login ──────────────────────────────────────────
                resp = await self.client.post("/auth/login", json={"cms": cms})
                if resp.status_code != 200:
                    logger.warning(f"Login failed for {cms}: {resp.text}")
                    continue

                token = resp.json()["token"]
                self.tokens[cms] = token
                self.active_students.append({"cms": cms, "name": data[cms]})
                logger.info(f"{Fore.GREEN}Logged in {data[cms]} ({cms}){Style.RESET_ALL}")

                # ── Step 2: Join a channel ─────────────────────────────────
                # This is the critical step that was missing.
                # Without it, find_student_channel(cms) returns None and
                # every /channel/message call fails with 400.
                join_resp = await self.client.post(f"/channel/join?token={token}")
                if join_resp.status_code == 200:
                    ch = join_resp.json().get("channel_key", "?")
                    logger.info(f"  → Assigned to {ch}")
                else:
                    logger.warning(f"  Channel join failed for {cms}: {join_resp.text}")

            except Exception as e:
                logger.error(f"Error setting up {cms}: {e}")

        return len(self.active_students) >= 2

    async def send_message(self, sender: Dict, recipient: Dict, text: str, is_burst: bool = False):
        token = self.tokens[sender["cms"]]
        payload = {
            "token": token,
            "to": recipient["cms"],
            "text": text
        }
        try:
            resp = await self.client.post("/channel/message", json=payload)
            color = Fore.RED if is_burst else Fore.CYAN
            tag = "[BURST]" if is_burst else "[Chatter]"

            if resp.status_code == 200:
                data = resp.json()
                channel  = data.get("sender_channel", "Unknown")
                status   = data.get("classification", {}).get("status", "Unknown")
                delivery = data.get("delivery_status", "DELIVERED")
                logger.info(
                    f"{color}{tag} {sender['name'][:10]} -> {recipient['name'][:10]} | "
                    f"CH: {channel} | Status: {status} | Delivery: {delivery}{Style.RESET_ALL}"
                )
            else:
                logger.warning(f"{tag} Message failed: {resp.status_code} - {resp.text}")
        except Exception:
            pass  # Suppress HTTP exceptions during heavy load

    async def baseline_chatter(self):
        logger.info("Starting baseline chatter task...")
        while True:
            sender    = random.choice(self.active_students)
            recipient = random.choice(self.active_students)
            if sender["cms"] != recipient["cms"]:
                msg = random.choice(CHITCHAT)
                asyncio.create_task(self.send_message(sender, recipient, msg))
            await asyncio.sleep(CHATTER_INTERVAL)

    async def congestion_bursts(self):
        logger.info("Starting congestion burst task...")
        while True:
            await asyncio.sleep(BURST_INTERVAL)
            logger.info(
                f"\n{Fore.YELLOW + Style.BRIGHT}=== INITIATING CONGESTION BURST ==={Style.RESET_ALL}"
            )
            spammers = random.sample(self.active_students, min(4, len(self.active_students)))
            target   = random.choice(self.active_students)
            msg      = random.choice(HEAVY_PAYLOADS)

            for _ in range(5):
                for sender in spammers:
                    if sender["cms"] != target["cms"]:
                        asyncio.create_task(
                            self.send_message(sender, target, msg, is_burst=True)
                        )
                await asyncio.sleep(0.1)


async def main():
    print(Fore.CYAN + Style.BRIGHT + "Starting CogniRad Peak Load Simulator..." + Style.RESET_ALL)

    sim     = SimulationClient()
    success = await sim.fetch_and_login()

    if not success:
        logger.error("Failed to login enough students. Make sure the server is running.")
        return

    task1 = asyncio.create_task(sim.baseline_chatter())
    task2 = asyncio.create_task(sim.congestion_bursts())

    try:
        await asyncio.gather(task1, task2)
    except asyncio.CancelledError:
        logger.info("Simulation stopped.")
    finally:
        await sim.client.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSimulation aborted by user.")
