<div align="center">
  <img src="https://img.shields.io/badge/Status-Live_on_Railway-success.svg" alt="Status" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-0.109-009688.svg" alt="FastAPI" />
  <img src="https://img.shields.io/badge/Architecture-Async_WebSocket-purple.svg" alt="Async" />
  <h1>🌐 CogniRad</h1>
  <p><strong>Next-Generation 5G Network Slicing & Cognitive Radio Simulation Engine</strong></p>

  <h3>
    <a href="https://web-production-db164.up.railway.app/">🔴 View Live Demo</a>
    <span> | </span>
    <a href="https://web-production-db164.up.railway.app/admin">🎛️ Admin Dashboard</a>
  </h3>
  <p><em>(Admin Password: <code>admin</code>)</em></p>
</div>

---

## 📡 About CogniRad

**CogniRad** is a professional, high-performance simulation environment designed to model cognitive radio behavior, dynamic spectrum allocation, and **5G Network Slicing** concepts in real-time. 

Modern telco networks must manage massive spikes in traffic while guaranteeing Quality of Service (QoS). Instead of a static, pre-allocated network, CogniRad treats communication channels as living physical entities. Every message transmitted adds simulated physical RF "energy" to the network. As energy accumulates across a frequency band, the system's SNR (Signal-to-Noise Ratio) degrades, pushing the network state from `FREE` ➔ `BUSY` ➔ `CONGESTED` ➔ `JAMMED`.

To prevent network collapse and packet loss, CogniRad employs a **Predictive AI Orchestrator**. This autonomous agent constantly monitors the energy slope ($J/s$) of the network slices. When an anomaly or traffic spike is detected, it performs fair, minimum-move **reallocations** to balance the load across available slices *before* the network fails.

### Core Objectives
- 🎓 **Educational Visualization:** Provide a stunning, real-time graphical interface to demonstrate 5G slicing, resource allocation, and multi-user MIMO contention.
- 🧪 **Algorithmic Sandbox:** Serve as a testbed for predictive QoS algorithms and automated load-balancing heuristics.
- 💬 **Live Messaging Layer:** Beyond a simulation, the platform functions entirely as a secure, WebSocket-based chatting application under the hood, routing actual payloads through the simulated spectrum.

---

## ✨ Enterprise-Grade Features

*   **⚡ 5G Slice Modeling:** Traffic is routed across 5 distinct bands representing modern, specialized network slices:
    *   `CH-1`: **Primary User (eMBB Slice)** — Enhanced Mobile Broadband
    *   `CH-2`: **Secondary User (URLLC Slice)** — Ultra-Reliable Low Latency
    *   `CH-3`: **Secondary User (mMTC Slice)** — Massive Machine Type Comm.
    *   `CH-4`: **V2X Edge Slice** — Vehicle-to-Everything
    *   `CH-5`: **Public Safety Band**
*   **📈 Dynamic Threshold Scaling:** Slice capacity expands sub-linearly ($\sqrt{N}$) based on connected users, mathematically modeling dynamic multi-user MIMO contention algorithms.
*   **📉 State-Dependent Decay:** Network energy dissipates logarithmically when idle, mirroring real-world physical signal dissipation.
*   **🧠 Predictive AI Orchestration:** A 1-second telemetry loop analyzes 60-second rolling histories to forecast congestion, evacuating users proactively to healthier slices.
*   **🚨 Forced Jamming & Security:** Admins can trigger simulated DDoS or Jamming attacks, instantly knocking out a slice to test the AI's emergency rerouting capabilities.
*   **📊 Live Telemetry Dashboards:** Featuring WebSocket-driven HTML5 Canvas charts—including a dynamic Radar/Spider graph—that render at 60fps under extreme data loads.

---

## 🎢 Experiencing the Simulation

CogniRad truly shines when the network is placed under extreme stress. Here is how you can visualize the 5G concepts in action on the live deployment:

### Scenario A: The Traffic Spike (Admin Driven)
1. Open the [Admin Dashboard](https://web-production-db164.up.railway.app/admin).
2. Locate the **Simulation** panel on the right sidebar.
3. Inject a massive payload of energy (e.g., `50 Joules`) into the network.
4. Watch the **5G Slice Radar Graph** visually warp as one slice absorbs the heavy load. The Predictive AI will detect the sharp energy slope and instantly trigger **Reallocation Events** to evacuate users to the `Public Safety` or `V2X` slices.

### Scenario B: The Bot Swarm (Local Load Testing)
Want to simulate an entire city block of devices communicating simultaneously over mMTC?
1. Clone the repository and run the load simulator locally:
   ```bash
   python simulate_load.py
   ```
2. This script spawns dozens of virtual users, connects them to the live WebSocket server, and spams the network. 
3. Watch the Admin Dashboard light up as the AI frantically, yet efficiently, load-balances the swarm across all 5 network slices in real-time.

---

## 🛠️ Technical Architecture

CogniRad is engineered for high-concurrency and real-time data streaming without the overhead of heavy frontend frameworks.

*   **Backend Framework:** `FastAPI` + `Uvicorn`
*   **Real-time Protocol:** Native WebSockets (`/ws/{token}` for clients, `/ws/spectrum` for admins)
*   **Persistence:** `SQLAlchemy` + `aiosqlite`
*   **Frontend Data-Viz:** Vanilla JS, CSS3, and native HTML5 Canvas drawing (Bezier curves, gradients, and glows).

### Mathematical Core
*   `classifier.py`: Implements the mathematical models for signal degradation and $\sqrt{N}$ channel capacity.
*   `allocator.py`: Executes the minimum-move algorithm that finds the most optimal 5G slice for displaced users.
*   `signal_physics.py`: Calculates continuous SNR, bitrates, and logarithmic energy decay.

---

## 🚀 Local Deployment

CogniRad is lightweight and zero-config. **No database setup or JSON seed files are required**—the built-in async SQLite database handles dynamic user registrations on the fly!

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/fahadnaseerhs/CogniRad.git
cd CogniRad
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Run the Server

Start the FastAPI application. By default, it runs on port `8080`.

```bash
python main.py
```

### 3. Access Locally

*   **👨‍🎓 Client App:** `http://127.0.0.1:8080/`
*   **🎛️ Admin Dashboard:** `http://127.0.0.1:8080/admin`
*   **📖 API Docs:** `http://127.0.0.1:8080/docs`

---

## ☁️ Production Deployment

CogniRad relies heavily on persistent memory state and active WebSocket connections. 

**✅ Recommended Platforms:** Railway, Render, Heroku, or standard VPS (EC2/DigitalOcean).  
**❌ Unsupported Platforms:** Serverless environments (like Vercel or AWS Lambda) will kill the background AI loops and sever WebSockets.

**Production Start Command:**
```bash
gunicorn main:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080 --workers 1
```
*(Note: Workers must remain at `1` to maintain the in-memory AI telemetry loop unless a Redis pub/sub layer is implemented).*
