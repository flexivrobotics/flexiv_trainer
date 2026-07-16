# RGB and Depth Camera Backend Flow

This document explains how the backend captures synchronized RGB and depth from
Intel RealSense cameras and stores both streams in a LeRobot 0.6 dataset.

## Overview

```text
RealSense camera
    │
    ├── BGR8 color stream ───────────────┐
    │                                    │
    └── Z16 depth stream                 │
            │                            │
            ├── align depth to color     │
            └── convert device units     │
                to uint16 millimeters    │
                         │                │
                         ▼                ▼
                 RealSense frame cache
                         │
                         ▼
                  RecordingService
                         │
            ┌────────────┴────────────┐
            │                         │
       BGR → RGB                 H×W → H×W×1
            │                         │
            ▼                         ▼
    LeRobot RGB feature       LeRobot depth feature
    H.264, yuv420p            HEVC, gray12le, lossless
```

## 1. Camera configuration

Camera settings are defined by `CameraConfig` in
`src/flexivtrainer/config.py`.

Each camera has:

- A logical name such as `ego`, `wrist`, `left_wrist`, or `right_wrist`.
- A RealSense device serial number.
- Width, height, and FPS settings.
- `use_depth`, which defaults to `true`.

`AppSettings.depth_max_m` defaults to `2.0` meters. It controls the upper
quantization range used by LeRobot's depth encoder and the display range used
for colorized previews. It does not change the raw RealSense capture range.

## 2. Starting the RealSense streams

`RealSenseService._start_runtime()` in
`src/flexivtrainer/cameras/realsense.py` creates one RealSense pipeline for each
active camera.

When depth is enabled, the pipeline requests two streams with matching
resolution and FPS:

```text
color: rs.stream.color, rs.format.bgr8
depth: rs.stream.depth, rs.format.z16
```

If the depth-enabled pipeline cannot start, the service retries with a fresh
pipeline using color only. The camera remains usable for RGB, while its status
reports that depth did not start. A recording that explicitly selects that
depth stream will be rejected instead of silently producing an RGB-only
dataset.

When depth starts successfully, the service also creates:

```python
rs.align(rs.stream.color)
```

This makes every captured depth map use the color image's pixel grid.

## 3. Background acquisition and frame cache

Each camera has one background acquisition thread running
`RealSenseService._acquire_loop()`.

Only this thread calls `pipeline.wait_for_frames()`. Live WebUI requests and
recording do not read from the RealSense pipeline independently. This prevents
multiple consumers from competing for frames.

For every frame set, the acquisition thread:

1. Waits for the next RealSense frame set.
2. Aligns the frame set to the color stream when depth is active.
3. Reads the BGR color image as a NumPy array.
4. Reads the Z16 depth image as a NumPy array.
5. Converts RealSense depth units into millimeters.
6. Stores the latest synchronized payload in the service's frame cache.

The depth conversion uses the device's reported scale:

```text
depth_mm = round(raw_z16 × depth_scale_m × 1000)
```

The result is clipped to the `uint16` range and stored as a two-dimensional
`uint16` array. On typical RealSense devices, the scale is `0.001` meters per
unit, so one raw unit already corresponds to one millimeter.

A cached camera payload has this general form:

```python
{
    "image": bgr_uint8_hwc,
    "depth": depth_uint16_hw,
    "timestamp_ms": camera_timestamp,
    "fps": measured_fps,
    "width": width,
    "height": height,
}
```

If one aligned depth frame is temporarily absent, the acquisition loop keeps
the previous cached depth map rather than dropping the corresponding color
frame.

## 4. Selecting recording streams

The recording entry list is resolved in
`src/flexivtrainer/data/lerobot_io.py`.

RGB entries use names such as:

```text
observation.images.ego
observation.images.wrist
```

Depth entries use the corresponding `_depth` suffix:

```text
observation.images.ego_depth
observation.images.wrist_depth
```

Depth entries are included in the default recording selection. The WebUI can
still deselect individual RGB or depth streams before recording.

## 5. Preparing the first recording frame

`RecordingService.start()` in
`src/flexivtrainer/data/recording_service.py` resolves the requested RGB and
depth camera names and asks `RealSenseService` to start those cameras.

Before creating a dataset, it verifies:

- Every selected camera has an active color stream.
- Every selected depth camera reports `depth.started=true`.
- A frame containing every requested stream can be read from the cache.

`RecordingService._grab_camera_data()` captures each selected camera only once
per recording tick, even when both RGB and depth are selected.

It then prepares the arrays for LeRobot:

- RGB is converted from RealSense BGR to contiguous RGB `uint8` in `H×W×3`
  layout.
- Depth remains `uint16` millimeters and is reshaped from `H×W` to contiguous
  `H×W×1` layout.

## 6. LeRobot feature schema

`build_features_from_sample()` creates the dataset feature definitions.

An RGB stream becomes:

```json
{
  "dtype": "video",
  "shape": [480, 640, 3],
  "names": ["height", "width", "channels"]
}
```

A depth stream becomes:

```json
{
  "dtype": "video",
  "shape": [480, 640, 1],
  "names": ["height", "width", "channels"],
  "info": {
    "is_depth_map": true
  }
}
```

The `is_depth_map` flag is what makes LeRobot route the feature through its
native depth encoder instead of treating it as an RGB video.

Because the input array is `uint16`, LeRobot records `depth_unit` as `mm` in
the final dataset metadata.

## 7. Video encoding

`RecordingService` creates the dataset with separate encoder configurations:

```python
rgb_encoder=RGBEncoderConfig(...)
depth_encoder=DepthEncoderConfig(depth_max=settings.depth_max_m)
```

The resulting formats are:

| Stream | Codec | Pixel format | Channels | Purpose |
|---|---|---|---:|---|
| RGB | H.264 | `yuv420p` | 3 | Browser-compatible RGB playback |
| Depth | HEVC | `gray12le` | 1 | Lossless 12-bit native depth storage |

LeRobot's depth encoder quantizes values between its configured minimum and
`depth_max_m`. With the current configuration, values above 2 meters are
clamped to the upper depth code. This affects stored far-range precision, so
`depth_max_m` should match the intended workspace.

Depth MP4 files are not ordinary browser videos. Most browsers cannot directly
display single-channel 12-bit HEVC. Failure to play a depth MP4 in an HTML
`<video>` element does not mean that the recording is corrupt.

## 8. Recording loop

During recording, `RecordingService._capture_loop()` repeats at the requested
dataset FPS:

1. Read the latest cached RGB and depth payload for each selected camera.
2. Convert RGB from BGR to RGB and shape depth as `H×W×1`.
3. Read the robot observation/action snapshot.
4. Add RGB, depth, robot data, and task text to one LeRobot frame.
5. Pass the frame to LeRobot's streaming encoders.

RGB and depth for a camera originate from the same cached aligned frame set,
although the final dataset timestamp is generated at the recording FPS by
LeRobot.

## 9. Live and saved-dataset visualization

Visualization does not alter the recorded depth values.

For live depth preview:

```text
GET /teleop/cameras/{camera_name}/frame?view=depth
```

The backend reads the cached `uint16` depth map, scales the configured
`0..depth_max_m` range to `0..255`, applies OpenCV's `COLORMAP_JET`, makes
invalid zero-depth pixels black, and returns a PNG.

The Data Collection checkbox labeled **Visualize depth** inserts separate
colorized depth windows below the RGB feeds. The RGB windows remain active.

For a saved dataset, the WebUI does not send the depth MP4 directly to the
browser. It requests decoded and colorized JPEG frames through:

```text
GET /datasets/frame-image?path=...&key=...&index=...
```

`RuntimeManager.dataset_frame_image()` loads the numeric depth through LeRobot,
colorizes it, and returns a browser-compatible JPEG.

## 10. Training behavior

LeRobot 0.6 supports native depth dataset storage and decoding, but the policy
families exposed by this application do not yet have an application-defined
depth ingestion path.

When a new-policy training dataset contains depth keys, `TrainingService`
therefore emits explicit `--policy.input_features` containing the RGB and state
features only. The recorded depth remains in the dataset for future policy
work, but current training does not consume it accidentally.

Fine-tuning continues to use the checkpoint's declared input features.

## 11. Reliability and performance fixes

This section documents the camera reliability and depth-preview performance
fixes made on top of the flow above.

### The intuition first

All of these problems came from one underlying reality: **RealSense cameras
(especially D405 units sharing a USB hub) are slow and flaky to start, and
depth video cannot be played by a browser.** The original code assumed cameras
start instantly and reliably, and it assumed a decoded video frame is cheap.
Neither is true. The fixes fall into four intuitive ideas:

1. **"Started" is not the same as "working."** Opening a pipeline succeeds long
   before the first frame actually arrives, and a pipeline can be open yet dead.
   The fix everywhere is to trust *frames*, not the "started" flag — the UI, the
   connected-count, and the recording checks all now key off whether recent
   frames are actually flowing (`streaming`), not whether the pipeline opened.

2. **Be patient while connecting, loud only when something truly breaks.** A
   camera that has never delivered a frame is simply still warming up, so the
   backend stays silent and the UI shows "connecting…" no matter how long it
   takes. An error is only surfaced for a camera that *was* streaming and then
   dropped — a genuine failure. And a stuck camera is retried forever in the
   background (plain restart, then a USB hardware reset), so one click is enough;
   you never have to mash "Connect."

3. **Never let two things touch one camera at once.** The original stop/start
   released its lock mid-teardown, so a second request could interleave and
   leave a zombie acquisition thread fighting the new one for the device. A
   single lifecycle lock now serializes every start/stop/restart, and each
   acquisition thread owns its own stop signal so it can never be orphaned.

4. **Do the expensive work once, in the background, and never on the critical
   path.** Depth can't stream to the browser, so it's decoded server-side. The
   first frame is now decoded on its own (~15 ms) so the image appears almost as
   fast as RGB, while the *whole clip* is decoded in a background thread so
   scrubbing and playback become instant a moment later. Heavy one-time imports
   are warmed at server startup so the very first depth request isn't stalled.

The rest of this section gives the specifics.

### 11.1 Streaming readiness vs. "started"

`RealSenseService.status()` reports a per-camera `streaming` flag
(`_is_streaming()`): the pipeline is started **and** a frame arrived within the
last ~2 seconds. `RuntimeManager.service_summary()` counts `streaming` cameras
for the "N/3 connected" home-page badge, so a camera that opened but delivers no
frames no longer shows as a clean "3/3 connected." A started-but-not-yet-
streaming camera reports a `working` tone with a "Connecting camera feeds…"
detail rather than an error, which is what keeps the UI honest during warm-up.

The dataset/live frontend (`renderCameraFps` in `web/app.js`) gates the live
`<img>` on `streaming` (falling back to `started`), so a stalled camera shows the
"Awaiting data" placeholder instead of a permanently blank live tile.

### 11.2 Start/stop race and zombie acquisition threads

Two request threads could previously overlap because `_stop_runtime()` releases
the short-hold frame-cache lock while it joins the acquisition thread (the dying
thread needs that lock to store its final frame). During that window another
`start_streams()` / `set_device_serials()` / `set_active_locations()` could run
and, critically, `_start_runtime()` used to **replace** `runtime.stop_event`,
leaving the still-blocked old thread reading a fresh, unset event — a zombie that
never exits and keeps competing for the device.

Fixes in `realsense.py`:

- A dedicated `self._lifecycle_lock` (RLock) wraps the entire body of every
  public mutating operation, so no two start/stop/restart cycles interleave. The
  short-hold `self._lock` still guards the frame cache and status reads.
- `_acquire_loop()` receives its `stop_event` and `pipeline` as **arguments**
  (thread-local) instead of re-reading them off the runtime, so replacing the
  runtime's event can never orphan a running thread.
- `_stop_runtime()` clears `self._last_frames[name]`, so a stopped camera can
  never serve (or record) a stale frozen frame.
- `_restart_started_cameras()` restarts only **active** slots, so an inactive
  slot (e.g. `wrist` in dual-arm mode) can't grab a device an active slot needs.

### 11.3 Single-click connect: never-give-up watchdog with hardware reset

D405 cameras on a shared USB hub are unreliable on a plain pipeline restart — a
fresh `pipeline.start()` sometimes never delivers a frame. The old code gave up
after a few restarts, so the user had to click "Connect" several times before a
camera happened to start cleanly.

`_acquire_loop()` now runs a self-healing watchdog (`_restart_if_silent()`):

- A started-but-silent stream is restarted after `SILENT_RESTART_AFTER_S`
  (3 seconds) with no frame.
- It **never permanently gives up** — it keeps retrying so one click is enough.
- Every third attempt escalates from a plain restart to a device
  `hardware_reset()` (`_hardware_reset()`), which reliably recovers a wedged
  D405 (verified: a plain restart is intermittent, ~15 ms first frame after a
  hardware reset). The reset drops the USB device, so the watchdog waits for it
  to re-enumerate before reopening.

### 11.4 Quiet warm-up, loud only on real failure

The acquisition loop no longer reports every `wait_for_frames` timeout as an
error. A camera that has **never delivered a frame** is treated as "still
connecting" and stays silent regardless of how long warm-up takes (some cameras
take several seconds). A timeout is surfaced as an error only when a camera that
**was** streaming drops out past the watchdog's grace window — a genuine
mid-session failure. Either way the watchdog keeps retrying.

### 11.5 Connect/disconnect UI feedback

`controlHomeService()` in `web/app.js`:

- After a cameras **connect**, it polls `/system/summary` until the service
  reaches `ok` (all streaming) or a timeout, keeping the "Connecting…" spinner up
  the whole time instead of flashing "0/3" and stopping after the ~0.5 s POST.
- **Disconnect** shows a "Disconnecting…" spinner for its full duration (the busy
  state now covers both actions), so it no longer looks frozen.
- After a **teleop** connect (which briefly blocks the backend while building the
  TDK controller and can momentarily starve the camera threads), it re-polls
  until the cameras settle, so the home panel doesn't get stuck showing a stale
  "Connecting camera feeds…" for cameras that are actually fine.

### 11.6 Depth preview that plays, and plays fast

Depth MP4s are single-channel 12-bit HEVC and cannot be played in a browser
`<video>`, so the review UI shows depth as a per-frame colorized `<img>` served
by `GET /datasets/frame-image` (see section 9). Two problems were fixed:

- **It only showed the first frame.** Each depth frame is a server-side decode
  (originally ~0.3 s), far slower than the ~10 fps playback tick, so assigning
  `img.src` every tick cancelled the still-loading previous request and froze the
  tile. `_pumpDepthImage()` in `web/app.js` now **gates on load**: it holds only
  the latest requested frame and issues the next request when the current one
  settles, so the tile keeps up by skipping frames it can't decode in time.

- **It was slow to appear (~1.7 s per episode).** The first depth request used to
  synchronously decode the *entire* clip before returning frame 0. Now
  `_depth_frame_jpeg()` in `runtime/manager.py`:
  - Serves the requested frame **on demand** via a single-frame PyAV seek/decode
    (~15 ms), so the first paint is near-instant — comparable to RGB.
  - Kicks off a **background daemon thread** (`_prewarm_depth_clip()`) that
    decodes the whole clip into `_depth_jpeg_cache`, so subsequent scrubbing and
    playback are memory-speed (~2 ms/frame). An in-flight set prevents a file
    from being decoded twice; a small per-frame LRU covers the warm-up window.
  - Encodes with `cv2.imencode(".jpg", ...)` instead of PIL — ~7× faster with
    byte-identical output.
  - The heavy `lerobot.datasets.depth_utils` import (which pulls in torch, ~2.7 s
    cold) is warmed at server startup by a background thread
    (`RuntimeManager.warm_up_depth_decode()`, launched from the app lifespan),
    so the first depth request in the server's life isn't stalled by it.

  All depth caches and the in-flight set are guarded by `self._depth_lock`, since
  they are now touched by both request threads and the prewarm thread.

## 12. Relevant source files

- `src/flexivtrainer/config.py`: camera and depth-range settings.
- `src/flexivtrainer/cameras/realsense.py`: stream startup, alignment, unit
  conversion, acquisition thread and frame cache, the lifecycle lock, the
  never-give-up watchdog / hardware reset, the `streaming` flag, and the
  quiet-warm-up error handling (sections 11.1–11.4).
- `src/flexivtrainer/data/lerobot_io.py`: RGB/depth entry resolution and
  LeRobot feature definitions.
- `src/flexivtrainer/data/recording_service.py`: stream validation, frame
  preparation, dataset creation, and recording loop.
- `src/flexivtrainer/api/routes/teleop.py`: live RGB/depth PNG endpoint.
- `src/flexivtrainer/api/app.py`: startup background warm-up of the depth-decode
  imports (section 11.6).
- `src/flexivtrainer/runtime/manager.py`: streaming-aware connected count,
  saved-dataset depth decoding/colorization, the on-demand single-frame decode,
  the background clip prewarm and caches, and `warm_up_depth_decode()`.
- `src/flexivtrainer/web/app.js`: live depth checkbox, dataset preview, the
  load-gated depth image pump, `streaming`-gated live tiles, and the
  connect/disconnect spinner + poll-until-ready logic.
