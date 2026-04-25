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
