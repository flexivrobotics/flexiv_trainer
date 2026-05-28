# Flexiv Trainer

Flexiv Trainer is a local-first application for collecting teleoperation demonstrations, saving them as LeRobot episodes, merging episodes into training datasets, and launching policy training runs.

The steps to set up and use this software can be summarized as:

1. Install the software.
2. Start the server.
3. Set up robots and cameras in the UI.
4. Teleoperate and record demonstration episodes.
5. Review and merge episodes into one training dataset.
6. Choose policy and start training.

## Software Requirements

1. OS: Ubuntu 22.04 or newer
2. Environment: Python 3.12 or newer
3. System config: see below

### Grant realtime privileges to non-root users

This is required to run the teleoperation module:

```bash
echo "${USER}    -   rtprio    99" | sudo tee -a /etc/security/limits.conf
echo "${USER}    -   nice     -20" | sudo tee -a /etc/security/limits.conf
```

## Hardware Requirements

1. A dual-arm teleoperation setup eligible for high-transparency teleoperation.
2. Supported cameras connected to your computer: 1 egocentric and 2 in-hand, both mounted on the follower robots side.

### Supported cameras

#### Egocentric

- RealSense D435

#### In-hand

- RealSense D405

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

When you install `flexivtrainer`, `pip` automatically installs the Python dependencies declared in `pyproject.toml`. You do not need to install them manually.

## Start Flexiv Trainer

Start the backend server from the same environment where you installed the package:

```bash
source .venv/bin/activate
flexiv-trainer-server
```

After startup, the terminal prints the URL for the web UI. By default this is:

```text
http://127.0.0.1:8000/
```

Open that URL in your browser to use the UI.

If the port is still occupied after closing a previous server session, clear leftover processes and start again:

```bash
source .venv/bin/activate
scripts/cleanup-server.sh
flexiv-trainer-server
```

## First-Time Setup In The UI

When the UI opens, you will start on the home page.

1. Enter the serial numbers for the two leader robots.
2. Enter the serial numbers for the two follower robots.
3. In the System Status tile, connect:
   - Teleop Service
   - Robot Data Service
   - Cameras
4. Confirm the system shows healthy status before moving on.

The home page also shows the storage locations used by the app for episodes, merged datasets, and training outputs.

## Collect Demonstrations In Data Collection

Open `Data Collection` from the navigation bar.

On this page you can verify robot and camera health, view the camera feeds, monitor telemetry, run teleoperation, and record episodes.

### Recommended workflow

1. Check `System Status` and make sure teleoperation, robot data, and cameras are all connected.
2. Confirm that the egocentric and wrist camera feeds are updating at around 30 FPS.
3. If needed, use `Home All Robots` before starting teleoperation.
4. In `Teleoperation Control`, click `Start` to begin teleoperation.
5. In `Episode Recording`, choose which entries you want to record.
   - The default selection includes all available camera observations, robot states, and robot actions.
   - Use `Select All` or `Deselect All` if needed.
6. Click the recording `Start` button when you are ready to capture a demonstration.
7. Perform the task through teleoperation.
8. Click the recording `Stop` button when one episode is complete.
9. Click `Save Episode` to keep the recording, or `Discard Episode` to give up saving.

Saved episodes are written to the episode storage directory shown on the home page. By default, episodes are stored under:

```text
.local/episodes/
```

## Load And Merge Episodes In Skill Training

Open `Skill Training` from the navigation bar.

### Step 1: Load episodes

1. Click the add button in `Load Episodes`.
2. In the episode browser, select one or more saved episode datasets.
3. Use `Select All` or `Deselect All` if you want to quickly toggle the full list.
4. Click `Load`.

### Step 2: Review selected episodes

1. Click an episode in the list to review it.
2. Use the checkboxes to choose which episodes should be merged.
3. Use `Select All` or `Deselect All` to toggle the current list.
4. Click `Merge Selected Episodes`.

### Step 3: Review the merged dataset

The merging will take some time depending on the actual dataset size. When done:

1. Review the merged dataset.
2. Click `Next` if the merged dataset looks correct.

Merged datasets are stored under:

```text
.local/merged/
```

## Start Training

After merging, Skill Training walks you through policy selection and output configuration.

1. Choose a training policy.
2. Click `Choose Directory` and select the training output directory.
3. Click `Start Training`.
4. See the training progress in the UI and terminal output.

Training outputs are stored in the directory you choose. The default app-managed location is:

```text
.local/training/
```

## Typical End-To-End Session

For a normal session, the order is:

1. Start `flexiv-trainer-server`.
2. Open the web UI.
3. Configure robot serial numbers on `Home`.
4. Connect teleop service, robot data service, and cameras.
5. Open `Data Collection`.
6. Start teleoperation and record one or more demonstration episodes.
7. Save the episodes you want to keep.
8. Open `Skill Training`.
9. Load and merge the saved episodes into one training dataset.
10. Choose a policy and output directory.
11. Start training.

## Open UI From Another Device

If you want to use the UI from another device on the same LAN, set a public base URL before starting the server:

```bash
export FLEXIV_TRAINER_PUBLIC_BASE_URL=http://<backend-host-ip>:8000
flexiv-trainer-server
```

Then open that public URL in a browser on the other device.

## API Documentation

The backend API doc can be accessed at:

```text
http://127.0.0.1:8000/docs
```
