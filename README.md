# Flexiv Trainer

Flexiv Trainer is a local-first application for collecting teleoperation demonstrations, saving them as LeRobot episodes, combining episodes into training datasets, and launching policy training runs.

The steps to set up and use this software can be summarized as:

1. Install the software
2. Start the server
3. Set up robots and cameras in the UI
4. Teleoperate and record episodes in Data Collection
5. Review and merge episodes in Skill Training
6. Start training

## Before You Start

- Python 3.12 or newer
- Two local Flexiv robots available to the machine running Flexiv Trainer
- Two remote Flexiv robots available for DDK data streaming
- Supported RealSense cameras connected and discoverable
- Robot serial numbers for the local and remote robots
- A Python virtual environment is strongly recommended

## Install

Create and activate a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install Flexiv Trainer:

```bash
pip install .
```

When you install `flexivtrainer`, `pip` automatically installs the Python dependencies declared in `pyproject.toml`. You do not need to install these one by one.

That automatic install includes:

- `fastapi`
- `rich`
- `uvicorn`
- `pydantic-settings`
- `typer`
- `numpy`
- `flexivrdk`
- `flexivtdk`
- `flexivddk`
- `lerobot`
- `pyrealsense2`

## Start Flexiv Trainer

Start the backend server from the same environment where you installed the package:

```bash
source .venv/bin/activate
flexiv-trainer-server
```

Compatible alternate command:

```bash
source .venv/bin/activate
flexivtrainer-server
```

After startup, the terminal prints the URL for the web UI. By default this is:

```text
http://127.0.0.1:8000/
```

Open that URL in your browser.

If the port is still occupied after closing a previous server session, clear leftover processes and start again:

```bash
source .venv/bin/activate
scripts/cleanup-server.sh
flexiv-trainer-server
```

## First-Time Setup In The UI

When the UI opens, you will start on the `Home` page.

1. Enter the serial numbers for the two local robots.
2. Enter the serial numbers for the two remote robots.
3. In the service status area, connect:
	- Teleoperation
	- Robot Data
	- Cameras
4. Confirm the system shows healthy status before moving on.

The `Home` page also shows the storage locations used by the app for episodes, combined datasets, and training outputs.

## Collect Demonstrations In Data Collection

Open `Data Collection` from the top navigation.

On this page you can verify robot and camera health, view the camera feeds, monitor telemetry, run teleoperation, and record episodes.

### Recommended workflow

1. Check `System Status` and make sure teleoperation, robot data, and cameras are all connected.
2. Confirm that the egocentric and wrist camera feeds are updating.
3. If needed, use `Home All Robots` before starting teleoperation.
4. In `Teleoperation Control`, click `Start` to begin teleoperation.
5. In `Episode Recording`, choose which entries you want to record.
	- The default selection includes camera observations and supported robot state/action entries.
	- Use `Select All` or `Deselect All` if needed.
6. Click the recording `Start` button when you are ready to capture a demonstration.
7. Perform the task through teleoperation.
8. Click the recording `Stop` button when the episode is complete.
9. Click `Save Episode` to keep the recording, or `Discard Episode` to throw it away.

Saved episodes are written to the episode storage directory shown on the `Home` page. By default, episodes are stored under:

```text
.local/episodes/
```

## Load And Combine Episodes In Skill Training

Open `Skill Training` from the top navigation.

### Step 1: Load episodes

1. Click the add button in `Load Episodes`.
2. In the episode browser, select one or more saved episode directories.
3. Use `Select All` or `Deselect All` if you want to quickly toggle the full list.
4. Click `Load`.

### Step 2: Review selected episodes

1. Click an episode in the list to preview it.
2. Use the checkboxes to choose which episodes should be merged.
3. Use `Select All` or `Deselect All` to toggle the current list.
4. Review the preview area:
	- Available camera feeds are shown in the preview player
	- Observation/action plots are shown when that data exists in the dataset
5. Click `Combine Selected Episodes`.

### Step 3: Review the combined dataset

After combining finishes, Flexiv Trainer opens a preview of the combined dataset.

1. Review the combined camera feeds and plots.
2. Click `Next` when the merged dataset looks correct.

Combined datasets are stored under:

```text
.local/combined/
```

## Start Training

After combining, Skill Training walks you through policy selection and output configuration.

1. Choose a training policy.
2. Click `Choose Directory` and select the training output directory.
3. Click `Start Training`.
4. Watch the training progress in the UI and terminal output.

Training outputs are stored in the directory you choose. The default app-managed location is:

```text
.local/training/
```

## Typical End-To-End Session

For a normal session, the order is:

1. Start `flexiv-trainer-server`
2. Open the web UI
3. Configure robot serial numbers on `Home`
4. Connect teleoperation, robot data, and cameras
5. Open `Data Collection`
6. Start teleoperation and record one or more episodes
7. Save the episodes you want to keep
8. Open `Skill Training`
9. Load and combine the saved episodes
10. Choose a policy and output directory
11. Start training

## Open From Another Device On The Same LAN

If you want to use the UI from another machine on the same network, set a public base URL before starting the server:

```bash
export FLEXIV_TRAINER_PUBLIC_BASE_URL=http://<backend-host-ip>:8000
flexiv-trainer-server
```

Then open that public URL in a browser on the other device.

## API Documentation

If you need the backend API documentation, open:

```text
http://127.0.0.1:8000/docs
```
