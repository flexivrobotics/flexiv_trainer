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

1. OS: Ubuntu 22.04+ (x86-64 or NVIDIA Jetson/aarch64) for the full workflow; macOS (Apple Silicon) is supported for training and data processing (see [Install](#install))
2. Environment: Python 3.12 or newer
3. System config: see below

### Grant realtime privileges to non-root users

This is required to run the teleoperation module:

```bash
echo "${USER}    -   rtprio    99" | sudo tee -a /etc/security/limits.conf
echo "${USER}    -   nice     -20" | sudo tee -a /etc/security/limits.conf
echo "${USER} soft memlock unlimited" | sudo tee -a /etc/security/limits.conf
echo "${USER} hard memlock unlimited" | sudo tee -a /etc/security/limits.conf
```

## Hardware Requirements

1. A dual-arm teleoperation setup eligible for high-transparency teleoperation.
2. Supported cameras connected to your computer: 1 egocentric and 2 in-hand, both mounted on the follower robots side.

### Supported cameras

- RealSense

## Install

Flexiv Trainer needs Python 3.12+. For GPU-accelerated training it also needs a PyTorch build that matches your hardware, so the prerequisite steps differ by platform. Every platform finishes the same way — create a virtual environment, then `pip install .` from the project directory, which automatically installs the Python dependencies declared in `pyproject.toml` (you do not need to install them manually).

After installing, continue to [Start Flexiv Trainer](#start-flexiv-trainer).

### Ubuntu 22.04+ (x86-64)

1. Install system prerequisites:

   ```bash
   sudo apt-get update
   sudo apt-get install -y python3.12 python3.12-venv python3-pip git libopenblas-dev
   ```

2. Create and activate a virtual environment, then install (from the project directory):

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install .
   ```

3. (NVIDIA GPU) The x86-64 PyPI PyTorch already ships with CUDA, so no extra wheel is needed. Verify the GPU is visible:

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

> For teleoperation, also grant realtime privileges (see [Software Requirements](#software-requirements)), then restart your computer.

### macOS (Apple Silicon)

> macOS is suited to training and data processing. Live teleoperation and RealSense capture require the Flexiv robot SDK and RealSense, which target Linux.

1. Install prerequisites with Homebrew:

   ```bash
   brew install python@3.12 git
   ```

2. Create and activate a virtual environment, then install (from the project directory):

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install .
   ```

3. PyTorch uses the Apple GPU (Metal/MPS) automatically. Verify:

   ```bash
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

### NVIDIA Jetson (AGX Thor / JetPack 7 / CUDA 13)

> The generic PyPI PyTorch is CPU-only on aarch64. Install NVIDIA's Jetson PyTorch wheel plus the CUDA math libraries it links against, otherwise training falls back to CPU.

1. Install system prerequisites:

   ```bash
   sudo apt-get update
   sudo apt-get install -y python3.12 python3.12-venv python3-pip git libopenblas-dev
   ```

2. Install the CUDA math libraries the Jetson PyTorch wheel links against. Get **cuSPARSELt** and **NVPL** from NVIDIA's official download pages — select Linux, arm64-sbsa, Ubuntu, your version, and deb (network), then run the commands each page generates:

   - cuSPARSELt: <https://developer.nvidia.com/cusparselt-downloads>
   - NVPL: <https://developer.nvidia.com/nvpl-downloads>

   Those pages also configure NVIDIA's apt repositories. With them in place, install NVPL and cuDSS, then register cuDSS with the dynamic linker (it installs into a versioned directory the loader does not scan by default):

   ```bash
   sudo apt-get install -y nvpl libcudss0-cuda-13
   echo "/usr/lib/aarch64-linux-gnu/libcudss/13" | sudo tee /etc/ld.so.conf.d/cudss.conf
   sudo ldconfig
   ```

3. Create and activate a virtual environment, then install the project (from the project directory):

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install .
   ```

4. Replace the CPU PyTorch with the CUDA build for Thor (from the NVIDIA Jetson AI Lab index):

   ```bash
   pip install --no-deps --force-reinstall \
     --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130 \
     torch==2.10.0 torchvision==0.25.0
   ```

5. Verify the GPU is visible (expect `True NVIDIA Thor`):

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

> For teleoperation, also grant realtime privileges (see [Software Requirements](#software-requirements)), then log out and back in.

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
.local/datasets/
```

## Start Training

In Policy Training, select a dataset and configure training.

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
8. Open `Data Processing`.
9. Load and merge the saved episodes into one training dataset.
10. Open `Policy Training`.
11. Select the merged dataset, choose a policy and output directory.
12. Start training.

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
