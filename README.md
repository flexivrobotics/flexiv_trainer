# Flexiv Trainer

Flexiv Trainer is a local-first platform for dual-arm Flexiv teleoperation,
episode recording, dataset combination, and LeRobot policy training.

## Current Scope

- Python backend scaffold with FastAPI routes and CLI entrypoints.
- Flexiv TDK teleoperation wrapper around `TransparentCartesianTeleopLAN`.
- Flexiv DDK snapshot reader for remote robots.
- RealSense discovery and stream bootstrap service.
- LeRobot episode writer for local single-episode datasets.
- Dataset combination and training job orchestration scaffolds.

## Prerequisites

The backend expects a Python environment with:

- `flexivtdk==1.6.0`
- `flexivddk==1.4.0`
- `lerobot==0.5.1`
- `pyrealsense2`
- `fastapi`, `uvicorn`, `pydantic-settings`, `typer`

The frontend expects a recent Node.js and npm installation.

## Repository Layout

- `backend/src/flexiv_trainer/`: backend package and CLI tools
- `.local/episodes/`: saved single-episode LeRobot datasets
- `.local/combined/`: combined datasets
- `.local/training/`: training outputs
- `.local/calibration/`: camera calibration files

## Backend Setup

Create and activate a virtual environment, then install the package in editable
mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If your Flexiv SDK packages are already installed in a different environment,
activate that environment first and then run:

```bash
pip install -e .
```

## Run the Backend

```bash
source .venv/bin/activate
flexiv-trainer-server
```

Equivalent `uvicorn` command:

```bash
source .venv/bin/activate
python -m uvicorn flexiv_trainer.api.app:app --host 0.0.0.0 --port 8000 --app-dir backend/src
```

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

If the frontend is served from a different host or port than the backend, set
`VITE_API_BASE` before starting the frontend. See `frontend/.env.example`.

## CLI Entrypoints

```bash
teleop --help
record_data --help
combine_episodes --help
train_policy --help
```
