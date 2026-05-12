<div align="center">
  <img src="https://img.shields.io/badge/Status-Active-success.svg" alt="Status" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.109-009688.svg" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Architecture-Async-purple.svg" alt="Async" />
  <h1>CogniRad</h1>
  <p><strong>5G Network Slicing & Cognitive Radio Simulation Platform</strong></p>
</div>

---

## 📡 What is CogniRad?

**CogniRad** is a professional, full-stack simulation environment designed to model cognitive radio behavior and **5G Network Slicing** concepts in real-time. It provides a visual and interactive way to understand how modern telco networks manage shared spectrum under heavy load.

Instead of a static network, CogniRad treats communication channels as living entities. Every message sent between users adds physical RF "energy" to the network. As energy accumulates, the system degrades from `FREE` ➔ `BUSY` ➔ `CONGESTED` ➔ `JAMMED`.

To prevent network collapse, CogniRad employs a **Predictive AI Orchestrator**. This AI constantly monitors the energy slope (J/s) of the network slices and performs fair, minimum-move **reallocations** to balance the load before packets are dropped.

### Key Use Cases
- 🎓 **Educational Demonstration:** Visually explain 5G slicing, resource allocation, and SNR degradation.
- 🧪 **Algorithm Testing:** Provide a sandbox to test load-balancing and predictive QoS algorithms.
- 💬 **Real-time Messaging App:** Functions entirely as a secure, WebSocket-based chatting application under the hood.

---

## ✨ Core Features

*   **5G Slice Modeling:** Traffic is routed across 5 distinct bands representing modern network slices:
    *   `CH-1`: **Primary User (eMBB Slice)** - Enhanced Mobile Broadband
    *   `CH-2`: **Secondary User (URLLC Slice)** - Ultra-Reliable Low Latency
    *   `CH-3`: **Secondary User (mMTC Slice)** - Massive Machine Type Comm.
    *   `CH-4`: **V2X Edge Slice** - Vehicle-to-Everything
    *   `CH-5`: **Public Safety Band**
*   **Dynamic Threshold Scaling:** The capacity of a slice expands sub-linearly ($\sqrt{N}$) based on the number of users connected, modeling dynamic multi-user MIMO contention.
*   **State-Dependent Decay:** Network energy dissipates logarithmically when idle, mirroring real-world signal dissipation.
*   **Predictive AI Loop:** A 1-second telemetry loop analyzes 60-second rolling history to forecast congestion and evacuate users proactively.
*   **Forced Jamming:** Admins can trigger simulated DDoS/Jamming attacks, instantly knocking out a slice and forcing the AI to reroute traffic.

---

## 🚀 Getting Started

CogniRad is designed to be lightweight and zero-config. **No database setup or `students.json` files are required**—the built-in async SQLite database handles dynamic registrations on the fly!

### 1. Installation

Clone the repository, create a virtual environment, and install the dependencies:

```bash
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Run the Server

Start the FastAPI application. By default, it runs on port `8080`.

```bash
python main.py
# Or use uvicorn directly:
# uvicorn main:app --host 0.0.0.0 --port 8080
```

### 3. Access the Dashboards

Once the server is running, navigate to:

*   **👨‍🎓 Student App (Client):** [http://127.0.0.1:8080/](http://127.0.0.1:8080/)
*   **🎛️ Admin Dashboard:** [http://127.0.0.1:8080/admin](http://127.0.0.1:8080/admin) *(Password: `admin`)*
*   **📖 API Docs:** [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs)

---

## 🎢 Simulating 5G Slicing Concepts

CogniRad truly shines when you simulate network stress. Here is how you can demonstrate 5G concepts:

### Method 1: The Built-in Dashboard Simulator
1. Open the **Admin Dashboard** (`/admin`).
2. Locate the **Simulation** panel on the right sidebar.
3. Inject a massive payload of energy (e.g., `50 Joules`) into the network.
4. Watch the **5G Slice Radar Graph** visually distort as one slice takes the heavy load. The Predictive AI will detect the sharp energy slope and instantly trigger **Reallocation Events** to move users to the `Public Safety` or `V2X` slices.

### Method 2: The Bot Swarm (Load Testing)
Want to simulate an entire classroom of devices communicating over mMTC?
1. Open a new terminal.
2. Run the load simulator:
   ```bash
   python simulate_load.py
   ```
3. This script will spawn dozens of virtual students, connect them via WebSockets, and have them spam the network. Watch the Admin Dashboard light up as the AI dynamically load-balances the swarm across all 5 network slices in real-time.

---

## 🛠️ Architecture

CogniRad is built for high-concurrency and real-time visualization:

*   **Backend Framework:** `FastAPI` + `Uvicorn`
*   **Real-time Protocol:** Native WebSockets (`/ws/{token}` for clients, `/ws/spectrum` for admins)
*   **Persistence:** `SQLAlchemy` + `aiosqlite`
*   **Frontend:** Vanilla JS, CSS3, and HTML5 Canvas (No heavy frameworks, allowing for instant 60fps graph rendering even under extreme DOM updates).

### Key Files
*   `classifier.py`: The mathematical models for signal degradation and $\sqrt{N}$ channel capacity.
*   `allocator.py`: The minimum-move algorithm that finds the most optimal 5G slice for displaced users.
*   `signal_physics.py`: Calculates SNR, bitrates, and logarithmic energy decay.

---

## ☁️ Deployment

CogniRad is production-ready for platforms like **Railway**, **Render**, or **Heroku**. 
Because it relies heavily on WebSockets and persistent memory state, **do not deploy to Serverless environments (like Vercel).**

For containerized or VPS deployments, use:
```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080 --workers 1
```
*(Note: Keep workers at `1` to maintain the in-memory AI telemetry loop unless you implement a Redis pub/sub layer).*
