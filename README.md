# Flexiv Trainer

Flexiv Trainer is a local-first platform for dual-arm Flexiv teleoperation,
episode recording, dataset combination, and LeRobot policy training.

## What End Users Need To Do

1. Prepare a Python environment containing the required robot and runtime packages.
2. Start the backend with `flexiv-trainer-server`.
3. Open the single URL printed by the backend.

You do not need to start a separate frontend development server. The backend
serves the web UI directly.

## Current Product Scope

- Backend-served operator UI available from the same URL as the backend.
- Flexiv TDK teleoperation wrapper around `TransparentCartesianTeleopLAN`.
- Flexiv DDK snapshot reader for remote robots.
- RealSense discovery and stream bootstrap service.
- LeRobot episode writer for local single-episode datasets.
- Dataset combination and training job orchestration scaffolds.
- Command-line entrypoints for teleoperation, recording, combining, and training.

## Prerequisites

The runtime expects a Python environment with:

- `flexivtdk==1.6.0`
- `flexivddk==1.4.0`
- `lerobot==0.5.1`
- `pyrealsense2`
- `fastapi`, `uvicorn`, `pydantic-settings`, `typer`

These can be installed into a dedicated virtual environment or into an existing
vendor-provided Flexiv environment.

## Repository Layout

- `backend/src/flexiv_trainer/`: backend package and API/CLI logic
- `backend/src/flexiv_trainer/web/`: packaged web UI served by the backend
- `.local/episodes/`: saved single-episode LeRobot datasets
- `.local/combined/`: combined datasets
- `.local/training/`: training outputs
- `.local/calibration/`: camera calibration files

## Setup

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

## Start Flexiv Trainer

```bash
source .venv/bin/activate
flexiv-trainer-server
```

After startup, the backend prints a single clickable URL. Open that URL in a
browser to use the UI.

By default this is:

```text
http://127.0.0.1:8000/
```

If you want to open the UI from another device on the same LAN, set a public
base URL before startup:

```bash
export FLEXIV_TRAINER_PUBLIC_BASE_URL=http://<backend-host-ip>:8000
flexiv-trainer-server
```

The backend also exposes the API documentation at:

```text
http://127.0.0.1:8000/docs
```

## Optional CLI Tools

In addition to the browser UI, the following command-line entrypoints are
available:

```bash
teleop --help
record_data --help
combine_episodes --help
train_policy --help
```

## Maintainer Notes

The repository also contains a separate `frontend/` workspace used for UI source
maintenance and future packaging work. End users do not need Node.js or npm to
run Flexiv Trainer in its packaged backend-served form.
