// Copyright 2026 Flexiv Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const state = {
    activeView: "home",
    summary: null,
    // Per-side manual gripper control inputs (velocity/force), keyed by arm
    // side. Seeded from the gripper params once teleop is running; kept in
    // memory so user edits survive re-renders within a session.
    gripperControl: {},
    teleopStatus: null,
    cameraConfig: null,
    trainingPolicies: null,
    trainingStatus: null,
    trainingDevices: null,
    teleopBootstrapped: false,
    trainingBootstrapped: false,
    processingBootstrapped: false,
    processingStep: 1,
    trainingStep: 1,
    episodes: [],
    selectedEpisodes: [],
    preview: null,
    mergedPreview: null,
    mergedPath: "",
    mergeProgress: null,
    // True while a merge is running and the progress overlay is shown.
    merging: false,
    previewSeries: null,
    mergedSeries: null,
    previewFrame: 0,
    mergedFrame: 0,
    previewPlaying: false,
    mergedPlaying: false,
    // Merged-dataset preview: which scope is shown (null = whole dataset, else
    // an episode index), the episode list for the picker, and a per-scope cache.
    mergedSelectedEpisode: null,
    mergedEpisodes: [],
    mergedScopeCache: {},
    // Training page state
    mergedDatasetPath: "",
    mergedDatasetPreview: null,
    mergedDatasetSeries: null,
    mergedDatasetFrame: 0,
    mergedDatasetPlaying: false,
    // Training-dataset preview scope (mirrors the merged-preview state above):
    // null = whole dataset, else an episode index; plus the episode list/cache.
    mergedDatasetSelectedEpisode: null,
    mergedDatasetEpisodes: [],
    mergedDatasetScopeCache: {},
    selectedPolicy: "diffusion",
    // Per-policy training-config edits, keyed by policy -> field name -> value.
    // Seeded from each policy's schema defaults; only values differing from the
    // default are sent as extra_args on start.
    policyConfig: {},
    trainingOutputStamp: "",
    trainingLogView: {
        stickToBottom: true,
        distanceFromBottom: 0,
        scrollLeft: 0,
    },
    pathBrowser: {
        mode: "generic",
        title: "",
        currentPath: "/",
        rootPath: "/",
        directoriesOnly: true,
        multiSelect: false,
        selected: [],
        items: [],
        allowNavigation: true,
        annotateEpisodeDirs: false,
        showSelectAll: false,
        hideHeader: false,
        hideEyebrow: false,
        hideClose: false,
        hideUp: false,
        requireSelection: false,
        fallbackToCurrentPath: true,
        emptyMessage: "No entries available.",
        confirmLabel: "Select",
        pathNote: "Current path",
        eyebrow: "Server Path Browser",
        onConfirm: null,
    },
    intervals: {
        teleop: null,
        training: null,
        recordingSave: null,
        // View-independent robot-fault watcher (1s); the floating fault widget
        // must surface a fault no matter which page the user is on.
        fault: null,
    },
    timers: {
        robotConfigSave: null,
        cameraConfigSave: null,
    },
    ui: {
        teleopRefreshBusy: false,
        teleopStartBusy: false,
        teleopZeroingSensors: false,
        teleopHomeBusy: false,
        recordingStartBusy: false,
        recordingSaveBusy: false,
        recordingSaveProgress: 0,
        trainingDeviceEvalBusy: false,
        trainingDeviceAutoTriggeredStep: null,
        // Smoothed realtime recording FPS, derived from the change in captured
        // frames between status polls. Reset whenever a recording (re)starts.
        recordingFps: 0,
        recordingFpsSample: null,
        // Signature of the currently-built Gripper Control panel; the panel is
        // only rebuilt when the set of gripper sides or their params change, so
        // velocity/force inputs aren't clobbered on every status poll.
        gripperPanelSignature: null,
        // True while the Gripper Control Init request (and its post-Init wait)
        // is in flight.
        gripperInitBusy: false,
        serviceResetBusy: {
            teleop_service: false,
            cameras: false,
        },
        serviceConnectBusy: {
            teleop: false,
            cameras: false,
        },
    },
    telemetryHistory: {
        left: [],
        right: [],
    },
    cameraFeeds: {},
    recordingEntries: [],
    // Entry ids the checklist has already offered. Used to distinguish "newly
    // offered" entries (auto-selected default-on, e.g. a gripper just enabled)
    // from ones the user explicitly deselected (must stay off). null until the
    // first reconcile seeds it.
    recordingOfferedEntries: null,
    datasetPlotScope: {},
    notifications: {
        items: [],
        unreadCount: 0,
        open: false,
        nextId: 1,
        lastTeleopIssueSignature: "",
    },
    // Floating fault widget shown above the notification center while any robot
    // reports a fault. `open` expands the message + Clear Fault control; `busy`
    // keeps the spinner while ClearFault() blocks; `cleared` swaps the panel to
    // the "fault cleared" confirmation with an OK button.
    fault: {
        open: false,
        busy: false,
        cleared: false,
        message: "",
    },
};

const TELEMETRY_HISTORY_LIMIT = 90;
// How often the teleop view polls /teleop/status. This sets the telemetry
// refresh rate (e.g. 100ms ≈ 10 FPS); refreshTeleopStatus() self-throttles via
// a queue, so requests never pile up if the backend is briefly slower.
const TELEOP_POLL_INTERVAL_MS = 100;
// How often the view-independent fault watcher polls /teleop/status (1s) so the
// floating fault widget surfaces a robot fault on any page, not just teleop.
const FAULT_POLL_INTERVAL_MS = 1000;
// Active arm sides in capture order, mirroring RobotSerialConfig.active_sides()
// on the backend. Single mode exposes only the chosen side; the index in this
// list is the arm's capture order (and its follower-serial index).
const DUAL_SIDES = ["left_arm", "right_arm"];
// Mirrors lerobot_io._WRIST_CAMERA_BY_SIDE. Single mode is side-free: the lone
// arm is "single_arm" with a neutral "wrist" camera (no left/right).
const WRIST_CAMERA_BY_SIDE = {
    left_arm: "left_wrist",
    right_arm: "right_wrist",
    single_arm: "wrist",
};
const ARM_SIDE_LABELS = {
    left_arm: { serial: "Serial Number - Left", feed: "Left Wrist", wrench: "LEFT" },
    right_arm: { serial: "Serial Number - Right", feed: "Right Wrist", wrench: "RIGHT" },
    single_arm: { serial: "Serial Number", feed: "Wrist", wrench: "ARM" },
};

function getActiveSides() {
    const config = state.summary?.robot_config;
    if (config?.arm_mode === "single") {
        return ["single_arm"];
    }
    return [...DUAL_SIDES];
}

const RECORDING_METRICS = [
    { metric: "tcp_pose", stateField: "tcp_pose", actionField: "tcp_pose_d" },
    { metric: "tcp_twist", stateField: "tcp_vel", actionField: "tcp_vel_d" },
    { metric: "tcp_wrench", stateField: "ext_wrench_in_world", actionField: "ext_wrench_d" },
];

// True when a side's follower end effector is configured as a gripper, so its
// measured width/force is available to record. Reads the cached end effector
// config without seeding defaults (getEndEffectorConfig mutates state).
function sideHasGripper(side) {
    const eec = state.summary?.robot_config?.end_effector_config || {};
    return eec[side]?.follower === "gripper";
}

// The checklist offers one toggle per (side, metric); selected metrics from
// every arm are concatenated into the single `observation.state` / `action`
// feature. Each row's label is the feature name-prefix it contributes (e.g.
// `left_arm.tcp_pose`), matching the per-axis names in those features. A side
// whose follower is a gripper also gets a `gripper` toggle (width + force);
// the same measured gripper states feed both its state and action entries, so
// both verify against the follower's `gripper` payload section.
function buildArmMetricRecordingOptions(sides) {
    const options = [];
    sides.forEach((side, index) => {
        for (const { metric, stateField } of RECORDING_METRICS) {
            options.push({
                id: `observation.state.${side}.${metric}`,
                label: `${side}.${metric}`,
                group: "observation.state",
                bucket: "observation",
                payload: "states",
                side: index,
                verifyField: stateField,
            });
        }
        if (sideHasGripper(side)) {
            options.push({
                id: `observation.state.${side}.gripper`,
                label: `${side}.gripper`,
                group: "observation.state",
                bucket: "observation",
                payload: "gripper",
                side: index,
                verifyField: "width",
            });
        }
    });
    sides.forEach((side, index) => {
        for (const { metric, actionField } of RECORDING_METRICS) {
            options.push({
                id: `action.${side}.${metric}`,
                label: `${side}.${metric}`,
                group: "action",
                bucket: "action",
                payload: "actions",
                side: index,
                verifyField: actionField,
            });
        }
        if (sideHasGripper(side)) {
            options.push({
                id: `action.${side}.gripper`,
                label: `${side}.gripper`,
                group: "action",
                bucket: "action",
                payload: "gripper",
                side: index,
                verifyField: "width",
            });
        }
    });
    return options;
}

// The ego camera is shared; each active arm contributes its own wrist camera.
function recordingEntryOptions() {
    const sides = getActiveSides();
    const images = [
        { id: "observation.images.ego", label: "ego", group: "observation.images", bucket: "image", sourceField: "ego" },
    ];
    sides.forEach((side) => {
        const camera = WRIST_CAMERA_BY_SIDE[side];
        images.push({ id: `observation.images.${camera}`, label: camera, group: "observation.images", bucket: "image", sourceField: camera });
    });
    return [...images, ...buildArmMetricRecordingOptions(sides)];
}

function defaultRecordingEntryIds() {
    return recordingEntryOptions().map((option) => option.id);
}
const SERVICE_RESET_TARGETS = {
    teleop_service: "teleop",
    cameras: "cameras",
};
const SERVICE_NAME_TO_KEY = {
    teleop: "teleop_service",
    cameras: "cameras",
};
const SERVICE_RESET_MESSAGES = {
    teleop_service: "Reconnect teleoperation service",
    cameras: "Reconnect cameras",
};
const RESET_ICON_SVG = `
    <svg class="icon-reset" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M7 7a6.5 6.5 0 0 1 10.4 1.1" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"></path>
        <path d="M18.5 8.5 17.2 5 13.7 6.3" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"></path>
        <path d="M17 17a6.5 6.5 0 0 1-10.4-1.1" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"></path>
        <path d="M5.5 15.5 6.8 19l3.5-1.3" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"></path>
    </svg>
`;
const CHECK_ICON_SVG = `
    <svg class="icon-check" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M5 12.5 9.2 16.7 19 7.4" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"></path>
    </svg>
`;
// Toggle-all icons for the Episodes sidebar: a checked box for "select all",
// a dashed box for "deselect all". Keeping it an icon button means its width
// never changes between states, so it can't overflow the narrow header.
const SELECT_ALL_ICON_SVG = `
    <svg class="icon-check" viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="4" width="16" height="16" rx="3" fill="none" stroke="currentColor" stroke-width="1.8"></rect>
        <path d="M8 12.3 10.8 15 16 8.8" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8"></path>
    </svg>
`;
const DESELECT_ALL_ICON_SVG = `
    <svg class="icon-check" viewBox="0 0 24 24" aria-hidden="true">
        <rect x="4" y="4" width="16" height="16" rx="3" fill="none" stroke="currentColor" stroke-width="1.8"></rect>
        <path d="M8 12h8" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="1.8"></path>
    </svg>
`;

// Set a toggle-all button to the icon + accessible label for its current state.
// `noun` (optional) tailors the label, e.g. "episodes" -> "Select all episodes".
function setToggleAllButton(button, allSelected, noun = "") {
    button.innerHTML = allSelected ? DESELECT_ALL_ICON_SVG : SELECT_ALL_ICON_SVG;
    const label = `${allSelected ? "Deselect all" : "Select all"}${noun ? ` ${noun}` : ""}`;
    button.title = label;
    button.setAttribute("aria-label", label);
}
const STOP_SQUARE_ICON_SVG = `
    <span class="recording-status__stop-square" aria-hidden="true"></span>
`;
// Crisp, centered play/pause glyphs for the dataset playback button. Unicode
// "▶"/"⏸" render as off-center colored emoji on some platforms.
// Inline width/height/flex via style (not presentation attributes): SVGs sized
// only by attributes collapse to ~0 inside a flex container in Chrome.
const DATASET_ICON_STYLE = "width:22px;height:22px;flex:0 0 auto;display:block";
const DATASET_PLAY_ICON = `
    <svg class="dataset-playback__icon" style="${DATASET_ICON_STYLE}" viewBox="0 0 24 24" aria-hidden="true">
        <path d="M7 5 20 12 7 19Z" fill="currentColor"></path>
    </svg>
`;
const DATASET_PAUSE_ICON = `
    <svg class="dataset-playback__icon" style="${DATASET_ICON_STYLE}" viewBox="0 0 24 24" aria-hidden="true">
        <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor"></rect>
        <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor"></rect>
    </svg>
`;
const TELEOP_START_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--play" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M8 6 18 12 8 18Z" fill="currentColor"></path>
        </svg>
        <span>Start</span>
    </span>
`;
const TELEOP_STOP_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--stop" viewBox="0 0 24 24" aria-hidden="true">
            <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"></rect>
        </svg>
        <span>Stop</span>
    </span>
`;
const TRAINING_RESUME_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--play" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M8 6 18 12 8 18Z" fill="currentColor"></path>
        </svg>
        <span>Resume</span>
    </span>
`;
const TRAINING_PAUSE_MARKUP = `
    <span class="button-content">
        <svg class="button-icon" viewBox="0 0 24 24" aria-hidden="true">
            <rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor"></rect>
            <rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor"></rect>
        </svg>
        <span>Pause</span>
    </span>
`;
const TRAINING_RESTART_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--play" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M8 6 18 12 8 18Z" fill="currentColor"></path>
        </svg>
        <span>Restart</span>
    </span>
`;
const TRAINING_START_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--play" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M8 6 18 12 8 18Z" fill="currentColor"></path>
        </svg>
        <span>Start</span>
    </span>
`;
const TELEOP_STARTING_MARKUP = `
    <span class="button-content">
        <span class="button-spinner" aria-hidden="true"></span>
        <span>Starting…</span>
    </span>
`;
const TELEOP_ENGAGE_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--bolt button-icon--bolt-engage" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M7 2v11h3v9l7-12h-4l4-8z" fill="currentColor"></path>
        </svg>
        <span>Engage</span>
    </span>
`;
const TELEOP_DISENGAGE_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--bolt button-icon--bolt-disengage" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M7 2v11h3v9l7-12h-4l4-8z" fill="currentColor"></path>
        </svg>
        <span>Disengage</span>
    </span>
`;
// Default training job name prefilled in the Episode Recording panel's Job name
// box; mirrors the server-side DEFAULT_JOB_NAME used when none is supplied.
const DEFAULT_JOB_NAME = "job_0";
// localStorage key the last-used job name is cached under so it persists across
// reloads; the box falls back to DEFAULT_JOB_NAME when nothing is cached.
const JOB_NAME_STORAGE_KEY = "flexivtrainer.lastJobName";

// Read the cached last-used job name, or "" if none/unavailable. Wrapped in
// try/catch because localStorage can throw in private-mode or sandboxed frames.
function loadCachedJobName() {
    try {
        return (window.localStorage.getItem(JOB_NAME_STORAGE_KEY) || "").trim();
    } catch (error) {
        return "";
    }
}

function saveCachedJobName(jobName) {
    const value = (jobName || "").trim();
    if (!value) {
        return;
    }
    try {
        window.localStorage.setItem(JOB_NAME_STORAGE_KEY, value);
    } catch (error) {
        // Persistence is best-effort; ignore storage failures.
    }
}

// localStorage key the last-used gripper velocity/force are cached under, keyed
// by arm side, so the sliders/number boxes resume their values across reloads
// instead of always reverting to the model defaults.
const GRIPPER_PARAMS_STORAGE_KEY = "flexivtrainer.gripperParams";

// Read the cached gripper params for one side as {velocity, force}, or {} when
// none/unavailable. Wrapped in try/catch (localStorage can throw in sandboxed
// or private-mode frames).
function loadCachedGripperParams(side) {
    try {
        const raw = window.localStorage.getItem(GRIPPER_PARAMS_STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : {};
        const entry = parsed && typeof parsed === "object" ? parsed[side] : null;
        return entry && typeof entry === "object" ? entry : {};
    } catch (error) {
        return {};
    }
}

function saveCachedGripperParams(side, velocity, force) {
    try {
        const raw = window.localStorage.getItem(GRIPPER_PARAMS_STORAGE_KEY);
        const parsed = raw ? JSON.parse(raw) : {};
        const next = parsed && typeof parsed === "object" ? parsed : {};
        next[side] = { velocity, force };
        window.localStorage.setItem(GRIPPER_PARAMS_STORAGE_KEY, JSON.stringify(next));
    } catch (error) {
        // Persistence is best-effort; ignore storage failures.
    }
}
const RECORD_START_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--play" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M8 6 18 12 8 18Z" fill="currentColor"></path>
        </svg>
        <span>Start</span>
    </span>
`;
const RECORD_STOP_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--stop" viewBox="0 0 24 24" aria-hidden="true">
            <rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"></rect>
        </svg>
        <span>Stop</span>
    </span>
`;
const RECORD_STARTING_MARKUP = `
    <span class="button-content">
        <span class="button-spinner" aria-hidden="true"></span>
        <span>Starting…</span>
    </span>
`;
const RECORD_SAVE_DEFAULT_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--save" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 12.5 9.2 16.7 19 7.4" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"></path>
        </svg>
        <span>Save Episode</span>
    </span>
`;
const RECORD_DISCARD_DEFAULT_MARKUP = `
    <span class="button-content">
        <svg class="button-icon button-icon--discard" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M7 7 17 17" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"></path>
            <path d="M17 7 7 17" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2"></path>
        </svg>
        <span>Discard Episode</span>
    </span>
`;
const LOADING_WHEEL_SEGMENTS = Array.from(
    { length: 12 },
    (_, index) => `<span style="--spinner-index:${index}"></span>`,
).join("");
state.recordingEntries = [...defaultRecordingEntryIds()];

let teleopStatusRefreshPromise = null;
let teleopStatusRefreshQueued = false;
// Monotonic token guarding against stale /teleop/status responses. The 100ms
// poller can have a request already in flight when the user clicks
// Engage/Disengage; that older request captured the pre-click backend state and
// its response can land *after* the click's refresh, reverting the button while
// teleop is still engaged. Each status read claims the next sequence number;
// only the newest claimed read is allowed to write state.teleopStatus.
let teleopStatusSeq = 0;

const TELEMETRY_SERIES = {
    force: {
        title: "Cartesian Force",
        units: "N",
        labels: ["f<sub>x</sub>", "f<sub>y</sub>", "f<sub>z</sub>"],
        colors: ["#8de0ff", "#86e4a8", "#ffbf7a"],
    },
    moment: {
        title: "Cartesian Moment",
        units: "Nm",
        labels: ["m<sub>x</sub>", "m<sub>y</sub>", "m<sub>z</sub>"],
        // Distinct from the force palette (blue/green/orange) so wrench bars and
        // the moment trend graphs are easy to tell apart.
        colors: ["#b89cff", "#ff9ecb", "#ffe08a"],
    },
};

// Recordings group each arm's signals into vector features, expanded by the
// backend into per-axis series keyed as
// `observation.state.<side>.<metric>.<axis>` (observed) and
// `action.<side>.<metric>.<axis>` (commanded), e.g.
// `observation.state.left_arm.tcp_pose.x`. Plot groups are derived from these
// keys so the layout adapts to whatever arms/axes a dataset actually contains.
const DATASET_METRIC_META = {
    tcp_pose: { title: "TCP Pose" },
    tcp_twist: { title: "TCP Twist" },
    tcp_wrench: { title: "TCP Wrench" },
    gripper: { title: "Gripper" },
};
const DATASET_METRIC_ORDER = { tcp_pose: 0, tcp_twist: 1, tcp_wrench: 2, gripper: 3 };
const DATASET_STATE_COLORS = ["#8de0ff", "#86e4a8", "#ffbf7a", "#c78dff", "#ff8da8", "#a8d8ff", "#ffe08a"];
const DATASET_ACTION_COLORS = ["#4db8db", "#4dba72", "#db9a4d", "#9a4ddb", "#db4d6a", "#6aa8db", "#dbc24d"];

function _datasetSideTitle(side) {
    if (side === "left_arm") return "Left Arm";
    if (side === "right_arm") return "Right Arm";
    return side.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function buildDatasetPlotGroups(numericKeys) {
    // gid "side|metric" -> { side, metric, axes: Map<axis, {axis, stateKey, actionKey}> }
    const groups = new Map();
    const sideOrder = [];
    (numericKeys || []).forEach((key) => {
        const match = key.match(/^(observation\.state|action)\.([a-z0-9_]+)\.(tcp_pose|tcp_twist|tcp_wrench|gripper)\.(.+)$/);
        if (!match) return;
        const [, section, side, metric, axis] = match;
        if (!(metric in DATASET_METRIC_META)) return;
        if (!sideOrder.includes(side)) sideOrder.push(side);
        const gid = `${side}|${metric}`;
        let group = groups.get(gid);
        if (!group) {
            group = { side, metric, axes: new Map() };
            groups.set(gid, group);
        }
        let entry = group.axes.get(axis);
        if (!entry) {
            entry = { axis, stateKey: null, actionKey: null };
            group.axes.set(axis, entry);
        }
        if (section === "action") entry.actionKey = key;
        else entry.stateKey = key;
    });

    return [...groups.values()]
        .sort((a, b) =>
            sideOrder.indexOf(a.side) - sideOrder.indexOf(b.side) ||
            (DATASET_METRIC_ORDER[a.metric] ?? 99) - (DATASET_METRIC_ORDER[b.metric] ?? 99)
        )
        .map(({ side, metric, axes }) => {
            const meta = DATASET_METRIC_META[metric];
            const axisEntries = [...axes.values()];
            const labels = axisEntries.map((e) => e.axis);
            return {
                id: `${side}.${metric}`,
                title: `${_datasetSideTitle(side)} · ${meta.title}`,
                units: "",
                labels,
                stateKeys: axisEntries.map((e) => e.stateKey),
                actionKeys: axisEntries.map((e) => e.actionKey),
                stateColors: labels.map((_, i) => DATASET_STATE_COLORS[i % DATASET_STATE_COLORS.length]),
                actionColors: labels.map((_, i) => DATASET_ACTION_COLORS[i % DATASET_ACTION_COLORS.length]),
            };
        });
}

// Pending pacing-timer handle per playback context (keyed by its playingKey:
// "previewPlaying", "mergedPlaying", "mergedDatasetPlaying").
const _datasetPlaybackTimers = {};
// Monotonic generation per playback context. Bumped whenever playback is
// (re)started or stopped, so a loop awaiting an image load can detect that it
// has been superseded (e.g. by a re-render) and exit instead of racing.
const _datasetPlaybackGen = {};

function _buildDatasetPlotSvg(seriesData, group, numFrames, currentFrame, scope, fps) {
    // Keep SVG aspect ratio aligned with .trend-chart (3:1) so the plot uses
    // the full canvas width without horizontal letterboxing.
    const width = 1200;
    const height = 400;
    const left = 58;
    const right = 12;
    const top = 18;
    const bottom = 48;
    const innerWidth = width - left - right;
    const innerHeight = height - top - bottom;

    // Collect all values for scale
    const allValues = [];
    for (const keys of [group.stateKeys, group.actionKeys]) {
        keys.forEach((key) => {
            const arr = key && seriesData[key];
            if (arr) {
                arr.forEach((v) => { if (v !== null && Number.isFinite(v)) allValues.push(v); });
            }
        });
    }

    let min, max;
    if (!allValues.length) {
        min = -1; max = 1;
    } else {
        min = Math.min(...allValues);
        max = Math.max(...allValues);
        if (Math.abs(max - min) < 1e-6) {
            const pad = Math.max(1, Math.abs(max) * 0.12 || 1);
            min -= pad; max += pad;
        } else {
            const pad = (max - min) * 0.12;
            min -= pad; max += pad;
        }
    }

    // Grid
    const lines = [];
    for (let i = 0; i <= 5; i++) {
        const x = left + (innerWidth * i) / 5;
        lines.push(`<line class="trend-chart__grid-line" x1="${x}" y1="${top}" x2="${x}" y2="${height - bottom}"></line>`);
    }
    for (let i = 0; i <= 4; i++) {
        const y = top + (innerHeight * i) / 4;
        lines.push(`<line class="trend-chart__grid-line" x1="${left}" y1="${y}" x2="${width - right}" y2="${y}"></line>`);
    }
    if (min < 0 && max > 0) {
        const zeroY = top + (1 - ((0 - min) / (max - min))) * innerHeight;
        lines.push(`<line class="trend-chart__zero" x1="${left}" y1="${zeroY}" x2="${width - right}" y2="${zeroY}"></line>`);
    }

    // Axis ticks/labels
    const labels = [];
    const xTicks = 6;
    const yTicks = 5;
    const safeFps = fps && Number.isFinite(fps) && fps > 0 ? fps : 30;
    const totalSeconds = numFrames > 1 ? (numFrames - 1) / safeFps : 0;
    for (let i = 0; i <= xTicks; i++) {
        const ratio = i / xTicks;
        const x = left + ratio * innerWidth;
        const seconds = (ratio * totalSeconds).toFixed(1);
        labels.push(`<text class="trend-chart__axis" x="${x}" y="${height - 14}" text-anchor="middle">${seconds}s</text>`);
    }
    for (let i = 0; i <= yTicks; i++) {
        const ratio = i / yTicks;
        const y = top + ratio * innerHeight;
        const value = (max - ratio * (max - min)).toFixed(1);
        labels.push(`<text class="trend-chart__axis" x="${left - 8}" y="${y + 5}" text-anchor="end">${value}</text>`);
    }
    // Playhead
    if (numFrames > 1 && currentFrame >= 0) {
        const px = left + (currentFrame / (numFrames - 1)) * innerWidth;
        lines.push(`<line class="dataset-plot__playhead" x1="${px}" y1="${top}" x2="${px}" y2="${height - bottom}"></line>`);
    }

    // Paths
    const paths = [];
    const drawSeries = (keys, colors, dash, role) => {
        for (let ci = 0; ci < keys.length; ci++) {
            const enabled = role === "state"
                ? !!scope?.stateEnabled?.[ci]
                : !!scope?.actionEnabled?.[ci];
            if (!enabled) continue;
            const key = keys[ci];
            const arr = key && seriesData[key];
            if (!arr) continue;
            let d = "";
            let drawing = false;
            for (let fi = 0; fi < numFrames; fi++) {
                const val = arr[fi];
                if (val === null || !Number.isFinite(val)) { drawing = false; continue; }
                const ratio = numFrames === 1 ? 1 : fi / (numFrames - 1);
                const x = left + ratio * innerWidth;
                const y = top + (1 - ((val - min) / (max - min))) * innerHeight;
                d += `${drawing ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)} `;
                drawing = true;
            }
            if (d) {
                paths.push(`<path class="trend-chart__line" style="--trend-color:${colors[ci]}${dash ? ";stroke-dasharray:8 4" : ""}" d="${d.trim()}"></path>`);
            }
        }
    };
    drawSeries(group.stateKeys, group.stateColors, false, "state");
    drawSeries(group.actionKeys, group.actionColors, true, "action");

    return `<svg class="trend-chart__svg" viewBox="0 0 ${width} ${height}" aria-hidden="true"><g>${lines.join("")}</g>${paths.join("")}${labels.join("")}</svg>`;
}

function _formatLegendValue(value) {
    return value !== null && value !== undefined && Number.isFinite(value)
        ? value.toFixed(3)
        : "—";
}

function _datasetPlotScopeKey(containerId, groupId) {
    return `${containerId}:${groupId}`;
}

function _ensureDatasetPlotScope(containerId, group) {
    const key = _datasetPlotScopeKey(containerId, group.id);
    const size = group.labels.length;
    const existing = state.datasetPlotScope[key];
    const scope = existing || {
        axisEnabled: Array(size).fill(true),
        stateEnabled: Array(size).fill(true),
        actionEnabled: Array(size).fill(true),
    };
    for (const field of ["axisEnabled", "stateEnabled", "actionEnabled"]) {
        const source = Array.isArray(scope[field]) ? scope[field] : [];
        scope[field] = Array.from({ length: size }, (_, i) => !!source[i]);
    }
    for (let i = 0; i < size; i++) {
        scope.axisEnabled[i] = !!(scope.stateEnabled[i] || scope.actionEnabled[i]);
    }
    state.datasetPlotScope[key] = scope;
    return scope;
}

function _groupHasVisibleData(seriesData, group, scope) {
    for (let i = 0; i < group.labels.length; i++) {
        if (scope.stateEnabled[i]) {
            const arr = group.stateKeys[i] ? seriesData[group.stateKeys[i]] : null;
            if (arr && arr.some((v) => v !== null && Number.isFinite(v))) return true;
        }
        if (scope.actionEnabled[i]) {
            const arr = group.actionKeys[i] ? seriesData[group.actionKeys[i]] : null;
            if (arr && arr.some((v) => v !== null && Number.isFinite(v))) return true;
        }
    }
    return false;
}

function _buildDatasetPlotLegend(seriesData, group, currentFrame, scope, groupIndex) {
    const items = [];
    for (let i = 0; i < group.labels.length; i++) {
        const stateKey = group.stateKeys[i];
        const actionKey = group.actionKeys[i];
        const stateArr = stateKey ? seriesData[stateKey] : null;
        const actionArr = actionKey ? seriesData[actionKey] : null;
        const hasStateData = !!(stateArr && stateArr.some((v) => v !== null && Number.isFinite(v)));
        const hasActionData = !!(actionArr && actionArr.some((v) => v !== null && Number.isFinite(v)));
        const axisEnabled = !!scope.axisEnabled[i];
        const stateEnabled = !!scope.stateEnabled[i];
        const actionEnabled = !!scope.actionEnabled[i];
        items.push(`
            <div class="dataset-scope-chip${axisEnabled ? "" : " dataset-scope-chip--off"}">
                <label class="dataset-scope-chip__axis">
                    <input class="dataset-plot-scope-toggle" data-group-index="${groupIndex}" data-axis-index="${i}" data-scope-role="axis" type="checkbox" ${axisEnabled ? "checked" : ""} />
                    <strong>${group.labels[i]}</strong>
                </label>
                <div class="dataset-scope-chip__subrows">
                    <label class="dataset-scope-chip__sub${hasStateData ? "" : " trend-chart__legend-item--dim"}">
                        <input class="dataset-plot-scope-toggle" data-group-index="${groupIndex}" data-axis-index="${i}" data-scope-role="state" type="checkbox" ${stateEnabled ? "checked" : ""} ${stateKey ? "" : "disabled"} />
                        <span class="trend-chart__swatch" style="--swatch:${group.stateColors[i]}"></span>
                        <span>state</span>
                        <span class="trend-chart__legend-value">${_formatLegendValue(stateArr ? stateArr[currentFrame] : null)}</span>
                    </label>
                    <label class="dataset-scope-chip__sub${hasActionData ? "" : " trend-chart__legend-item--dim"}">
                        <input class="dataset-plot-scope-toggle" data-group-index="${groupIndex}" data-axis-index="${i}" data-scope-role="action" type="checkbox" ${actionEnabled ? "checked" : ""} ${actionKey ? "" : "disabled"} />
                        <span class="trend-chart__swatch trend-chart__swatch--dashed" style="--swatch:${group.actionColors[i]}"></span>
                        <span>action</span>
                        <span class="trend-chart__legend-value">${_formatLegendValue(actionArr ? actionArr[currentFrame] : null)}</span>
                    </label>
                </div>
            </div>
        `);
    }
    return items.join("");
}

function _showPreviewLoadingOverlay(containerId) {
    const container = byId(containerId);
    if (!container) return;
    container.innerHTML = `
        <div class="panel panel--soft" style="position:relative;min-height:200px">
            <div class="preview-loading-overlay">
                <span class="preview-loading-overlay__label">Loading data ...</span>
                <div class="preview-loading-bar"><span></span></div>
            </div>
        </div>`;
}

// Show/hide the small merge-progress overlay (a modal over the current page).
function _showMergeModal() {
    const modal = byId("merge-modal");
    if (!modal) return;
    const bar = modal.querySelector(".progress-bar");
    if (bar) {
        bar.querySelector("span:first-child").style.width = "0%";
        const txt = bar.querySelector(".progress-bar__text");
        if (txt) txt.textContent = `0/${state.selectedEpisodes.length}`;
    }
    modal.classList.remove("hidden");
}

function _hideMergeModal() {
    const modal = byId("merge-modal");
    if (modal) modal.classList.add("hidden");
}

async function _pollMergeProgress() {
    while (true) {
        await new Promise((r) => setTimeout(r, 400));
        if (!state.merging) return;
        try {
            const prog = await api("/datasets/merge-progress");
            state.mergeProgress = prog;
            if (prog.status === "done") {
                const result = prog.result;
                state.mergedPath = result.root;
                const wholePreview = await api(`/datasets/preview?path=${encodeURIComponent(result.root)}`);
                state.mergedFrame = 0;
                state.mergedPlaying = false;
                state.mergedSelectedEpisode = null;
                state.mergedEpisodes = wholePreview.episodes || [];
                state.mergedScopeCache = {};
                _stopDatasetPlayback("mergedPlaying");
                let wholeSeries = null;
                try {
                    wholeSeries = await api(`/datasets/series?path=${encodeURIComponent(result.root)}`);
                } catch (_) {
                    wholeSeries = null;
                }
                state.mergedScopeCache.all = { preview: wholePreview, series: wholeSeries };
                state.mergedPreview = wholePreview;
                state.mergedSeries = wholeSeries;
                // Done: close the overlay and jump to the merged-dataset preview page.
                state.merging = false;
                _hideMergeModal();
                state.processingStep = 3;
                renderProcessing();
                return;
            } else if (prog.status === "error") {
                state.merging = false;
                _hideMergeModal();
                showToast(prog.error || "Merge failed", true);
                return;
            }
            // Update the overlay's progress bar in place without re-rendering.
            _updateMergeProgressBars(prog);
        } catch (_) {
            // endpoint not available yet, keep polling
        }
    }
}

// Per-scope merged-dataset preview controller, shared by the Data Processing
// (step 3) and Policy Training (step 1) pages. Each context names the state
// keys it reads/writes, the preview block element, and how to re-render.
const MERGED_PREVIEW_CTX = {
    processing: {
        path: () => state.mergedPath,
        previewBlockId: "merged-preview-block",
        rerender: () => renderProcessing(),
        keys: {
            selected: "mergedSelectedEpisode", episodes: "mergedEpisodes",
            cache: "mergedScopeCache", preview: "mergedPreview",
            series: "mergedSeries", frame: "mergedFrame", playing: "mergedPlaying",
        },
    },
    training: {
        path: () => state.mergedDatasetPath,
        previewBlockId: "merged-dataset-preview-block",
        rerender: () => renderTraining(),
        keys: {
            selected: "mergedDatasetSelectedEpisode", episodes: "mergedDatasetEpisodes",
            cache: "mergedDatasetScopeCache", preview: "mergedDatasetPreview",
            series: "mergedDatasetSeries", frame: "mergedDatasetFrame",
            playing: "mergedDatasetPlaying",
        },
    },
};

// Switch a merged-dataset preview to a scope: null = whole dataset, or an
// episode index. Results are cached per scope so toggling back is instant.
async function _selectDatasetScope(ctxName, episodeIndex) {
    const ctx = MERGED_PREVIEW_CTX[ctxName];
    const k = ctx.keys;
    const cacheKey = episodeIndex == null ? "all" : String(episodeIndex);
    state[k.selected] = episodeIndex;
    state[k.frame] = 0;
    state[k.playing] = false;
    _stopDatasetPlayback(k.playing);

    const cache = state[k.cache] || (state[k.cache] = {});
    if (cache[cacheKey]) {
        state[k.preview] = cache[cacheKey].preview;
        state[k.series] = cache[cacheKey].series;
        ctx.rerender();
        return;
    }

    ctx.rerender();
    _showPreviewLoadingOverlay(ctx.previewBlockId);
    const path = ctx.path();
    const epQuery = episodeIndex == null ? "" : `&episode_index=${episodeIndex}`;
    try {
        const preview = await api(`/datasets/preview?path=${encodeURIComponent(path)}${epQuery}`);
        let series = null;
        try {
            series = await api(`/datasets/series?path=${encodeURIComponent(path)}${epQuery}`);
        } catch (_) {
            series = null;
        }
        cache[cacheKey] = { preview, series };
        // Ignore a stale response if the user switched scope while loading.
        if (state[k.selected] !== episodeIndex) return;
        state[k.preview] = preview;
        state[k.series] = series;
        ctx.rerender();
    } catch (error) {
        showToast(error.message, true);
    }
}

// Populate an episode picker ("Whole Dataset" + each episode) for a context.
function _renderDatasetEpisodePicker(pickerEl, ctxName) {
    if (!pickerEl) return;
    const k = MERGED_PREVIEW_CTX[ctxName].keys;
    const sel = state[k.selected];
    const current = state[k.preview];
    const episodeList =
        (current && current.episodes && current.episodes.length
            ? current.episodes
            : state[k.episodes]) || [];
    const totalFrames =
        episodeList.reduce((sum, ep) => sum + (ep.num_frames || 0), 0)
        || (current && current.num_frames) || 0;
    const entries = [{ label: "Whole Dataset", index: null, num_frames: totalFrames }].concat(
        episodeList.map((ep) => ({ label: `Episode ${ep.index}`, index: ep.index, num_frames: ep.num_frames }))
    );
    entries.forEach((entry) => {
        const isSelected = entry.index === sel || (entry.index == null && sel == null);
        const row = document.createElement("div");
        row.className = `episode-row episode-row--selectable dataset-scope-row ${isSelected ? "episode-row--selected" : ""}`.trim();
        const meta = `<span class="episode-row__meta">${(entry.num_frames || 0).toLocaleString()} frames</span>`;
        row.innerHTML = `<div class="episode-row__main"><span>${escapeHtml(entry.label)}</span></div>${meta}`;
        row.onclick = () => _selectDatasetScope(ctxName, entry.index);
        pickerEl.appendChild(row);
    });
}

function _updateMergeProgressBars(prog) {
    const overallPercent = prog.total_episodes ? Math.round((prog.episode_index / prog.total_episodes) * 100) : 0;
    const block = document.querySelector(".merge-progress-block");
    if (!block) return;
    const bar = block.querySelector(".progress-bar");
    if (bar) {
        bar.querySelector("span:first-child").style.width = `${overallPercent}%`;
        const txt = bar.querySelector(".progress-bar__text");
        if (txt) txt.textContent = `${prog.episode_index}/${prog.total_episodes}`;
    }
}

function renderDatasetPreviewBlock(containerId, preview, seriesData, frameKey, playingKey) {
    const container = byId(containerId);
    if (!container || !preview) return;

    const currentFrame = state[frameKey];
    const numFrames = preview.num_frames || 0;
    const isPlaying = state[playingKey];
    const datasetPath = preview.path;

    // Camera feeds: stream the MP4 directly so the browser decodes it natively
    // (smooth playback) instead of fetching one re-encoded JPEG per frame.
    // Display the left wrist before the right wrist (swap their positions).
    const orderedCameraKeys = [...(preview.camera_keys || [])];
    const leftIdx = orderedCameraKeys.findIndex((k) => k.includes("left_wrist"));
    const rightIdx = orderedCameraKeys.findIndex((k) => k.includes("right_wrist"));
    if (leftIdx !== -1 && rightIdx !== -1) {
        [orderedCameraKeys[leftIdx], orderedCameraKeys[rightIdx]] = [
            orderedCameraKeys[rightIdx],
            orderedCameraKeys[leftIdx],
        ];
    }
    // For an episode-scoped preview the camera feeds live inside a shared MP4,
    // so target the episode's video file and play only its time window (native
    // looping is disabled — playback wraps within the window in the tick loop).
    const videoWindows = preview.video_windows || null;
    const feedsHtml = orderedCameraKeys.map((cameraKey) => {
        const win = videoWindows ? videoWindows[cameraKey] : null;
        const chunkIndex = win ? win.chunk_index : 0;
        const fileIndex = win ? win.file_index : 0;
        const loopAttr = win ? "" : "loop";
        const src = `/datasets/video?path=${encodeURIComponent(datasetPath)}&key=${encodeURIComponent(cameraKey)}&chunk_index=${chunkIndex}&file_index=${fileIndex}`;
        return `
        <div class="feed">
            <div class="feed__header"><span>${cameraKey}</span></div>
            <div class="feed__placeholder" data-render-mode="live">
                <video class="feed__image dataset-frame-video" data-camera-key="${cameraKey}" src="${src}" muted playsinline preload="auto" ${loopAttr}></video>
            </div>
        </div>
    `;
    }).join("");

    // Playback controls
    const playbackHtml = `
        <div class="dataset-playback">
            <button class="dataset-playback__btn" data-action="${isPlaying ? "pause" : "play"}" type="button" aria-label="${isPlaying ? "Pause" : "Play"}" style="display:inline-flex;align-items:center;justify-content:center">${isPlaying ? DATASET_PAUSE_ICON : DATASET_PLAY_ICON}</button>
            <input type="range" class="dataset-playback__slider" min="0" max="${Math.max(numFrames - 1, 0)}" value="${currentFrame}" />
            <div class="dataset-playback__meta">
                <span class="dataset-playback__timer">0:00 / 0:00</span>
                <span class="dataset-playback__counter">${currentFrame + 1} / ${numFrames}</span>
            </div>
        </div>
    `;

    // Plots
    const plotGroups = buildDatasetPlotGroups(preview.numeric_keys);
    const plotsHtml = plotGroups.map((group, groupIndex) => {
        const scope = _ensureDatasetPlotScope(containerId, group);
        const hasAnyData = seriesData ? _groupHasVisibleData(seriesData, group, scope) : false;
        const svg = seriesData ? _buildDatasetPlotSvg(seriesData, group, numFrames, currentFrame, scope, preview.fps || 30) : "";
        const legend = seriesData ? _buildDatasetPlotLegend(seriesData, group, currentFrame, scope, groupIndex) : "";
        return `
            <div class="dataset-plot-card" data-plot-group-index="${groupIndex}">
                <span class="eyebrow">${group.title}</span>
                <div class="trend-chart">
                    ${svg}
                    ${!hasAnyData ? `<div class="trend-chart__empty">No visible data</div>` : ""}
                </div>
                <div class="trend-chart__legend">${legend}</div>
            </div>
        `;
    }).join("");

    container.innerHTML = `
        <span class="eyebrow dataset-feed-title">Videos</span>
        <div class="feed-row dataset-feed-row">${feedsHtml}</div>
        ${playbackHtml}
        <div class="dataset-plots-grid">${plotsHtml}</div>
    `;

    // Attach playback event handlers
    const slider = container.querySelector(".dataset-playback__slider");
    if (slider) {
        slider.oninput = () => {
            state[frameKey] = parseInt(slider.value, 10);
            _updateDatasetFrames(container, preview, seriesData, frameKey, playingKey);
        };
    }
    const playBtn = container.querySelector(".dataset-playback__btn");
    if (playBtn) {
        playBtn.onclick = () => {
            state[playingKey] = !state[playingKey];
            if (state[playingKey]) {
                _startDatasetPlayback(container, preview, seriesData, frameKey, playingKey);
            } else {
                _stopDatasetPlayback(playingKey);
                container.querySelectorAll(".dataset-frame-video").forEach((video) => video.pause());
            }
            // Update button
            playBtn.innerHTML = state[playingKey] ? DATASET_PAUSE_ICON : DATASET_PLAY_ICON;
            playBtn.dataset.action = state[playingKey] ? "pause" : "play";
            playBtn.ariaLabel = state[playingKey] ? "Pause" : "Play";
        };
    }

    container.onchange = (event) => {
        const target = event.target;
        if (!(target instanceof HTMLInputElement)) return;
        if (!target.classList.contains("dataset-plot-scope-toggle")) return;
        const groupIndex = Number(target.dataset.groupIndex);
        const axisIndex = Number(target.dataset.axisIndex);
        const role = target.dataset.scopeRole;
        if (!Number.isInteger(groupIndex) || !Number.isInteger(axisIndex)) return;
        const group = plotGroups[groupIndex];
        if (!group) return;
        const scope = _ensureDatasetPlotScope(containerId, group);
        if (role === "axis") {
            scope.axisEnabled[axisIndex] = target.checked;
            scope.stateEnabled[axisIndex] = target.checked;
            scope.actionEnabled[axisIndex] = target.checked;
        } else if (role === "state") {
            scope.stateEnabled[axisIndex] = target.checked;
            scope.axisEnabled[axisIndex] = !!(scope.stateEnabled[axisIndex] || scope.actionEnabled[axisIndex]);
        } else if (role === "action") {
            scope.actionEnabled[axisIndex] = target.checked;
            scope.axisEnabled[axisIndex] = !!(scope.stateEnabled[axisIndex] || scope.actionEnabled[axisIndex]);
        }
        _renderDatasetFrameMeta(container, preview, seriesData, frameKey);
        _rebuildDatasetLegends(container, preview, seriesData, frameKey);
    };

    // Restore the playhead position on the freshly-mounted videos.
    _seekDatasetVideos(container, preview, currentFrame);
    // Start playback if already playing
    if (isPlaying) {
        _startDatasetPlayback(container, preview, seriesData, frameKey, playingKey);
    }
    _renderDatasetFrameMeta(container, preview, seriesData, frameKey);
}

function _datasetPreviewFps(preview) {
    return preview.fps && Number.isFinite(preview.fps) && preview.fps > 0 ? preview.fps : 30;
}

// Video time offset for a camera. For an episode-scoped preview the feed sits
// inside a shared MP4, so frame 0 maps to the episode's from_timestamp; for a
// whole-dataset preview there is no window and the offset is 0.
function _videoWindowOffset(preview, cameraKey) {
    const win = preview.video_windows && preview.video_windows[cameraKey];
    return win ? (win.from_timestamp || 0) : 0;
}

// Seek every camera <video> to the given frame (time = offset + frame / fps).
// Frames are at constant fps, so the frame index maps linearly to video time.
function _seekDatasetVideos(container, preview, frameIndex) {
    const fps = _datasetPreviewFps(preview);
    container.querySelectorAll(".dataset-frame-video").forEach((video) => {
        const t = _videoWindowOffset(preview, video.dataset.cameraKey) + frameIndex / fps;
        if (video.readyState >= 1) {
            video.currentTime = t;
        } else {
            video.addEventListener("loadedmetadata", () => { video.currentTime = t; }, { once: true });
        }
    });
}

function _formatPlaybackTime(seconds) {
    const safe = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
    const total = Math.floor(safe);
    const minutes = Math.floor(total / 60);
    const secs = total % 60;
    return `${minutes}:${String(secs).padStart(2, "0")}`;
}

function _renderDatasetFrameMeta(container, preview, seriesData, frameKey) {
    const currentFrame = state[frameKey];
    const numFrames = preview.num_frames || 0;
    const fps = preview.fps && Number.isFinite(preview.fps) && preview.fps > 0 ? preview.fps : 30;

    // Update slider and counter
    const slider = container.querySelector(".dataset-playback__slider");
    if (slider) slider.value = currentFrame;
    const timer = container.querySelector(".dataset-playback__timer");
    if (timer) {
        const currentSeconds = currentFrame / fps;
        const totalSeconds = Math.max(0, (Math.max(numFrames - 1, 0)) / fps);
        timer.textContent = `${_formatPlaybackTime(currentSeconds)} / ${_formatPlaybackTime(totalSeconds)}`;
    }
    const counter = container.querySelector(".dataset-playback__counter");
    if (counter) counter.textContent = `${currentFrame + 1} / ${numFrames}`;

    // Update playheads in SVGs
    const plotGroups = buildDatasetPlotGroups(preview.numeric_keys);
    container.querySelectorAll(".dataset-plot-card").forEach((card, gi) => {
        const group = plotGroups[gi];
        if (!group || !seriesData) return;
        const chartDiv = card.querySelector(".trend-chart");
        if (!chartDiv) return;
        const scope = _ensureDatasetPlotScope(container.id, group);
        chartDiv.innerHTML = _buildDatasetPlotSvg(seriesData, group, numFrames, currentFrame, scope, preview.fps || 30);
        // Re-add "no data" overlay if needed
        const hasAnyData = _groupHasVisibleData(seriesData, group, scope);
        if (!hasAnyData) {
            chartDiv.innerHTML += `<div class="trend-chart__empty">No visible data</div>`;
        }
        // Refresh the per-legend current-frame values so they track the
        // playhead. Update only the value spans in place — rebuilding the whole
        // legend here would recreate the checkbox inputs every frame, which
        // makes them impossible to click during playback.
        const legendDiv = card.querySelector(".trend-chart__legend");
        if (legendDiv) {
            const valueSpans = legendDiv.querySelectorAll(".trend-chart__legend-value");
            let vi = 0;
            for (let i = 0; i < group.labels.length; i++) {
                const stateArr = group.stateKeys[i] ? seriesData[group.stateKeys[i]] : null;
                const actionArr = group.actionKeys[i] ? seriesData[group.actionKeys[i]] : null;
                if (valueSpans[vi]) valueSpans[vi].textContent = _formatLegendValue(stateArr ? stateArr[currentFrame] : null);
                vi++;
                if (valueSpans[vi]) valueSpans[vi].textContent = _formatLegendValue(actionArr ? actionArr[currentFrame] : null);
                vi++;
            }
        }
    });
}

// Rebuild the full legend markup (checkboxes + chip dimming) for every plot
// card. Called after a scope toggle, where the enabled/disabled state and the
// derived axis checkbox need to be reflected in the DOM.
function _rebuildDatasetLegends(container, preview, seriesData, frameKey) {
    const currentFrame = state[frameKey] || 0;
    const plotGroups = buildDatasetPlotGroups(preview.numeric_keys);
    container.querySelectorAll(".dataset-plot-card").forEach((card, gi) => {
        const group = plotGroups[gi];
        if (!group || !seriesData) return;
        const legendDiv = card.querySelector(".trend-chart__legend");
        if (!legendDiv) return;
        const scope = _ensureDatasetPlotScope(container.id, group);
        legendDiv.innerHTML = _buildDatasetPlotLegend(seriesData, group, currentFrame, scope, gi);
    });
}

// Used for manual scrubbing (slider): update meta and seek the videos to match.
function _updateDatasetFrames(container, preview, seriesData, frameKey, playingKey) {
    _renderDatasetFrameMeta(container, preview, seriesData, frameKey);
    _seekDatasetVideos(container, preview, state[frameKey]);
}

function _startDatasetPlayback(container, preview, seriesData, frameKey, playingKey) {
    _stopDatasetPlayback(playingKey);
    const fps = _datasetPreviewFps(preview);
    const numFrames = preview.num_frames || 0;
    if (numFrames <= 0) return;
    const gen = _datasetPlaybackGen[playingKey];
    const live = () => state[playingKey] && _datasetPlaybackGen[playingKey] === gen;

    const videos = [...container.querySelectorAll(".dataset-frame-video")];
    if (!videos.length) return;
    const master = videos[0];
    const masterOffset = _videoWindowOffset(preview, master.dataset.cameraKey);
    const masterWindow = preview.video_windows && preview.video_windows[master.dataset.cameraKey];

    // Play every feed natively; the browser handles decode/timing smoothly.
    videos.forEach((video) => {
        const playResult = video.play();
        if (playResult && typeof playResult.catch === "function") {
            playResult.catch(() => { });
        }
    });

    // Drive the slider/plots/legend from the master video's clock once per
    // animation frame, so the playhead tracks the natively-decoded video.
    const tick = () => {
        if (!live()) return;
        // Episode windows share an MP4 with other episodes, so wrap back to the
        // window start instead of letting the feed run into the next episode.
        if (masterWindow && master.currentTime >= masterWindow.to_timestamp - 0.5 / fps) {
            videos.forEach((video) => {
                video.currentTime = _videoWindowOffset(preview, video.dataset.cameraKey);
            });
        }
        const frame = Math.min(
            numFrames - 1,
            Math.max(0, Math.round((master.currentTime - masterOffset) * fps))
        );
        if (frame !== state[frameKey]) {
            state[frameKey] = frame;
            _renderDatasetFrameMeta(container, preview, seriesData, frameKey);
            // Correct any drift between feeds without disrupting the master.
            for (let i = 1; i < videos.length; i++) {
                const offsetDelta = _videoWindowOffset(preview, videos[i].dataset.cameraKey) - masterOffset;
                const target = master.currentTime + offsetDelta;
                if (Math.abs(videos[i].currentTime - target) > 0.15) {
                    videos[i].currentTime = target;
                }
            }
        }
        _datasetPlaybackTimers[playingKey] = requestAnimationFrame(tick);
    };
    _datasetPlaybackTimers[playingKey] = requestAnimationFrame(tick);
}

function _stopDatasetPlayback(playingKey) {
    // Bump the generation so any in-flight tick callback bails out.
    _datasetPlaybackGen[playingKey] = (_datasetPlaybackGen[playingKey] || 0) + 1;
    const handle = _datasetPlaybackTimers[playingKey];
    if (handle) {
        cancelAnimationFrame(handle);
        _datasetPlaybackTimers[playingKey] = null;
    }
}

function byId(id) {
    return document.getElementById(id);
}

// Human-readable local timestamp suffix (YYYYMMDD_HHMMSS), matching how
// recorded episode directories are named.
function _timestampSuffix() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

function setMarkupIfChanged(element, renderKey, markup) {
    if (!element) {
        return false;
    }
    if (element.dataset.renderKey === renderKey) {
        return false;
    }
    element.innerHTML = markup;
    element.dataset.renderKey = renderKey;
    return true;
}

function formatNotificationTimestamp(timestamp) {
    return new Date(timestamp).toLocaleString([], {
        month: "short",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
    });
}

function renderNotificationCenter() {
    const toggle = byId("notification-toggle");
    const badge = byId("notification-badge");
    const panel = byId("notification-panel");
    const list = byId("notification-list");
    if (!toggle || !badge || !panel || !list) {
        return;
    }

    toggle.setAttribute("aria-expanded", state.notifications.open ? "true" : "false");
    panel.classList.toggle("hidden", !state.notifications.open);

    const unread = state.notifications.unreadCount;
    badge.textContent = String(unread);
    badge.classList.toggle("hidden", unread === 0);

    if (!state.notifications.items.length) {
        list.innerHTML = `<div class="notification-empty">No messages yet.</div>`;
        return;
    }

    list.innerHTML = state.notifications.items.map((item) => `
        <article class="notification-item notification-item--${escapeHtml(item.level)}">
            <div class="notification-item__row">
                <span class="notification-item__pill">${escapeHtml(item.level.toUpperCase())}</span>
                <time class="notification-item__time">${formatNotificationTimestamp(item.timestamp)}</time>
            </div>
            <p class="notification-item__message">${escapeHtml(item.message)}</p>
            ${item.count > 1 ? `<span class="notification-item__count">x${item.count}</span>` : ""}
        </article>
    `).join("");
}

function toggleNotificationCenter(forceOpen) {
    const nextOpen = typeof forceOpen === "boolean" ? forceOpen : !state.notifications.open;
    state.notifications.open = nextOpen;
    if (nextOpen) {
        state.notifications.unreadCount = 0;
    }
    renderNotificationCenter();
}

// Drive the floating fault widget from the latest teleop status. The widget is
// hidden when no fault is present (and not mid-clear), shown as a pulsing red
// caution icon otherwise. Clicking the icon expands the message + Clear Fault
// button; while ClearFault() runs the button spins; once the fault clears the
// panel swaps to a confirmation with an OK button.
function syncFaultCenter(teleopStatus) {
    const fault = state.fault;
    const message = teleopStatus?.teleop?.fault;
    const faulted = !!message;

    if (faulted) {
        fault.message = String(message);
        // A fresh fault after a prior clear should reset the confirmation state
        // so the panel shows the actionable message again.
        if (fault.cleared) {
            fault.cleared = false;
        }
    } else if (!fault.busy && !fault.cleared) {
        // No fault and nothing in-flight: fully reset and hide.
        fault.open = false;
        fault.message = "";
    }

    renderFaultCenter(faulted);
}

function renderFaultCenter(faulted) {
    const center = byId("fault-center");
    const toggle = byId("fault-toggle");
    const panel = byId("fault-panel");
    const messageEl = byId("fault-message");
    const actions = byId("fault-actions");
    if (!center || !toggle || !panel || !messageEl || !actions) {
        return;
    }

    const fault = state.fault;
    // Keep the widget mounted while clearing or showing the cleared confirmation
    // even though any_fault() already reads false.
    const visible = faulted || fault.busy || fault.cleared;
    center.classList.toggle("hidden", !visible);
    if (!visible) {
        fault.open = false;
    }

    toggle.setAttribute("aria-expanded", fault.open ? "true" : "false");
    panel.classList.toggle("hidden", !fault.open);
    if (!fault.open) {
        return;
    }

    const message = fault.cleared
        ? "Fault cleared. The robot is ready."
        : (fault.message || "Robot fault detected.");
    if (messageEl.textContent !== message) {
        messageEl.textContent = message;
    }

    // renderFaultCenter runs on every teleop poll tick (~10x/sec). Rewriting the
    // action button's innerHTML unconditionally detaches the node mid-click and
    // drops the click (the bug seen on the Data Collection page). Only rebuild
    // when the action state actually changes, keyed below.
    const actionState = fault.cleared ? "ok" : (fault.busy ? "busy" : "clear");
    if (actions.dataset.faultActionState !== actionState) {
        actions.dataset.faultActionState = actionState;
        if (actionState === "ok") {
            actions.innerHTML = `<button class="fault-action-button fault-ok-button" id="fault-ok" type="button">OK</button>`;
        } else if (actionState === "busy") {
            actions.innerHTML = `<button class="fault-action-button fault-clear-button" id="fault-clear" type="button" disabled><span class="button-spinner" aria-hidden="true"></span><span>Clearing…</span></button>`;
        } else {
            actions.innerHTML = `<button class="fault-action-button fault-clear-button" id="fault-clear" type="button"><span>Clear Fault</span></button>`;
        }

        const okButton = byId("fault-ok");
        if (okButton) {
            okButton.onclick = () => {
                state.fault.cleared = false;
                state.fault.open = false;
                state.fault.message = "";
                renderFaultCenter(false);
            };
        }
        const clearButton = byId("fault-clear");
        if (clearButton && !fault.busy) {
            clearButton.onclick = handleClearFault;
        }
    }
}

async function handleClearFault() {
    if (state.fault.busy) {
        return;
    }
    state.fault.busy = true;
    renderFaultCenter(true);
    try {
        // ClearFault() blocks server-side until the fault clears or times out
        // (up to 30s), so allow a generous client timeout to match.
        const result = await api("/teleop/clear-fault", { method: "POST" }, {
            timeoutMs: 35000,
            timeoutMessage: "Clear fault timed out",
        });
        if (result?.cleared) {
            state.fault.cleared = true;
        } else {
            showToast(result?.error || "Failed to clear robot fault", true);
        }
    } catch (error) {
        showToast(error.message, true);
    } finally {
        state.fault.busy = false;
        // Re-poll immediately so the widget reflects any_fault()'s real state;
        // syncFaultCenter on the next tick keeps it honest if the fault returns.
        refreshTeleopStatus().catch(() => { });
        // Drive visibility from the widget's own state, not state.teleopStatus,
        // which the background poller no longer keeps current on non-teleop pages.
        renderFaultCenter(!state.fault.cleared && !!state.fault.message);
    }
}

function toggleFaultCenter(forceOpen) {
    const nextOpen = typeof forceOpen === "boolean" ? forceOpen : !state.fault.open;
    state.fault.open = nextOpen;
    renderFaultCenter(!!state.fault.message);
}

function pushNotification(message, level = "info") {
    const normalizedMessage = String(message || "").trim();
    if (!normalizedMessage) {
        return;
    }

    const latest = state.notifications.items[0];
    if (latest && latest.message === normalizedMessage && latest.level === level) {
        latest.timestamp = Date.now();
        latest.count += 1;
    } else {
        state.notifications.items.unshift({
            id: state.notifications.nextId,
            message: normalizedMessage,
            level,
            timestamp: Date.now(),
            count: 1,
        });
        state.notifications.nextId += 1;
        state.notifications.items = state.notifications.items.slice(0, 120);
    }

    if (!state.notifications.open) {
        state.notifications.unreadCount += 1;
    }
    renderNotificationCenter();
}

async function api(path, init, options = {}) {
    const timeoutMs = Number(options.timeoutMs || 0);
    const timeoutMessage = options.timeoutMessage || "Request timed out";
    const controller = timeoutMs > 0 ? new AbortController() : null;
    const timer = controller
        ? window.setTimeout(() => controller.abort(), timeoutMs)
        : null;

    try {
        const response = await fetch(path, {
            headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
            ...init,
            signal: init?.signal || controller?.signal,
        });
        if (!response.ok) {
            const detail = await readErrorDetail(response);
            const baseMessage = `Request failed: ${response.status} ${response.statusText}`;
            throw new Error(detail ? `${baseMessage} - ${detail}` : baseMessage);
        }
        return response.json();
    } catch (error) {
        if (error?.name === "AbortError") {
            throw new Error(timeoutMessage);
        }
        throw error;
    } finally {
        if (timer !== null) {
            window.clearTimeout(timer);
        }
    }
}

async function readErrorDetail(response) {
    try {
        const contentType = (response.headers.get("content-type") || "").toLowerCase();
        if (contentType.includes("application/json")) {
            const payload = await response.json();
            const detail = payload?.detail;
            if (typeof detail === "string" && detail.trim()) {
                return detail.trim();
            }
            if (Array.isArray(detail)) {
                return detail.map((entry) => {
                    if (typeof entry === "string") {
                        return entry;
                    }
                    if (entry && typeof entry === "object") {
                        return entry.msg || JSON.stringify(entry);
                    }
                    return String(entry);
                }).join(" | ");
            }
            if (typeof payload?.message === "string" && payload.message.trim()) {
                return payload.message.trim();
            }
            if (payload && typeof payload === "object" && Object.keys(payload).length > 0) {
                return JSON.stringify(payload);
            }
            return "";
        }

        const text = (await response.text()).trim();
        return text;
    } catch (_) {
        return "";
    }
}

function showToast(message, isError = false) {
    pushNotification(message, isError ? "error" : "info");
}

function formatValue(value) {
    if (Array.isArray(value)) {
        return value.join(", ");
    }
    if (value === null || value === undefined || value === "") {
        return "Not available";
    }
    return String(value);
}

function createStatusCard(label, value, tone = "neutral") {
    const card = document.createElement("div");
    card.className = `status-card status-card--${tone}`;
    card.innerHTML = `<span class="eyebrow">${label}</span><h3>${formatValue(value)}</h3>`;
    return card;
}

function setTeleopHomeBusy(busy) {
    state.ui.teleopHomeBusy = busy;
    if (state.activeView === "teleoperation") {
        renderTeleop();
    }
}

function setTeleopStartBusy(busy) {
    state.ui.teleopStartBusy = busy;
    if (state.activeView === "teleoperation") {
        renderTeleop();
    }
}

function setRecordingStartBusy(busy) {
    state.ui.recordingStartBusy = busy;
    if (state.activeView === "teleoperation") {
        renderTeleop();
    }
}

function setRecordingSaveBusy(busy) {
    state.ui.recordingSaveBusy = busy;
    if (!busy) {
        state.ui.recordingSaveProgress = 0;
    }
    if (state.activeView === "teleoperation") {
        renderTeleop();
    }
}

function setServiceResetBusy(serviceKey, busy) {
    state.ui.serviceResetBusy[serviceKey] = busy;
    if (state.activeView === "teleoperation") {
        renderTeleop();
    }
}

async function resetTeleopSystemService(serviceKey) {
    const serviceName = SERVICE_RESET_TARGETS[serviceKey];
    if (!serviceName || state.ui.serviceResetBusy[serviceKey]) {
        return;
    }
    if (serviceKey === "teleop_service" && !allRobotSerialsConfigured()) {
        showToast(ROBOT_SERIALS_REQUIRED_MESSAGE, true);
        return;
    }

    const label = state.teleopStatus?.services?.[serviceKey]?.label
        || state.summary?.services?.[serviceKey]?.label
        || serviceKey;

    setServiceResetBusy(serviceKey, true);
    try {
        await controlHomeService(serviceName, "disconnect", { silentToast: true });
        await controlHomeService(serviceName, "connect", { silentToast: true });
        showToast(`${label} reset.`);
    } finally {
        setServiceResetBusy(serviceKey, false);
    }
}

function stopTeleopPolling() {
    if (state.intervals.teleop !== null) {
        window.clearInterval(state.intervals.teleop);
        state.intervals.teleop = null;
    }
    stopRecordingSavePolling();
}

function startTeleopPolling() {
    if (state.intervals.teleop !== null) {
        return;
    }

    state.intervals.teleop = window.setInterval(() => {
        if (state.activeView !== "teleoperation") {
            return;
        }

        refreshTeleopStatus().catch((error) => {
            stopTeleopPolling();
            showToast(error.message, true);
        });
    }, TELEOP_POLL_INTERVAL_MS);
}

// View-independent fault watcher: poll /teleop/status once a second so the
// floating fault widget appears on any page. On the teleoperation view the
// 100ms poller above already drives syncFaultCenter (and the full telemetry
// render), so skip there to avoid redundant fetches. Runs for the app's
// lifetime -- never stopped -- so it must stay lightweight: it only updates the
// fault widget, not the heavy teleop panels.
let faultPollInFlight = false;
function startFaultPolling() {
    if (state.intervals.fault !== null) {
        return;
    }

    state.intervals.fault = window.setInterval(async () => {
        if (state.activeView === "teleoperation" || faultPollInFlight) {
            return;
        }
        faultPollInFlight = true;
        try {
            const status = await api("/teleop/status");
            // Drive only the fault widget from this lightweight poll. Don't
            // write state.teleopStatus: it's the shared snapshot many flows
            // read (engage handler, recording, telemetry), and this 1s, off-view
            // poll would feed them stale, out-of-order data. syncFaultCenter
            // keeps the widget's own state (state.fault) current on its own.
            syncFaultCenter(status);
        } catch (error) {
            // Network blips shouldn't spam toasts from a background poller;
            // leave the widget in its last known state and retry next tick.
        } finally {
            faultPollInFlight = false;
        }
    }, FAULT_POLL_INTERVAL_MS);
}

function normalizePercent(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
        return 0;
    }
    return Math.max(0, Math.min(100, Math.round(numeric)));
}

function stopRecordingSavePolling() {
    if (state.intervals.recordingSave !== null) {
        window.clearInterval(state.intervals.recordingSave);
        state.intervals.recordingSave = null;
    }
}

async function pollRecordingSaveProgressOnce() {
    const nextStatus = await api("/teleop/status");
    state.teleopStatus = nextStatus;
    const backendProgress = normalizePercent(nextStatus?.recording?.save_progress || 0);
    state.ui.recordingSaveProgress = Math.max(state.ui.recordingSaveProgress, backendProgress);
    if (state.summary) {
        state.summary.services = nextStatus.services || state.summary.services;
        renderHomeStatus();
    }
    renderTeleop();
}

function startRecordingSavePolling() {
    stopRecordingSavePolling();
    pollRecordingSaveProgressOnce().catch(() => {
        // Ignore transient polling failures while save is in progress
    });
    state.intervals.recordingSave = window.setInterval(() => {
        pollRecordingSaveProgressOnce().catch(() => {
            // Ignore transient polling failures while save is in progress
        });
    }, 250);
}

function hasActiveTeleopServices(teleopStatus) {
    if (!teleopStatus) {
        return false;
    }

    const teleop = teleopStatus.teleop || {};
    if (teleop.initialized || teleop.started) {
        return true;
    }

    const robots = Object.values(teleopStatus.robot_data?.robots || {});
    if (robots.some((robot) => !!robot?.connected)) {
        return true;
    }

    const cameras = Object.values(teleopStatus.cameras?.cameras || {});
    if (cameras.some((camera) => !!camera?.started)) {
        return true;
    }

    const recording = teleopStatus.recording || {};
    return !!recording.active || !!recording.awaiting_save;
}

// Render the System Status cards in place. renderTeleop runs on every poll
// tick (~10x/sec); wiping and rebuilding the grid each time detached the reset
// button between a click's mousedown and mouseup, so the click event never
// fired and users had to click repeatedly. Reuse each card node and only
// rewrite its markup when the rendered content actually changes.
function renderTeleopSystemCards(grid, services = {}) {
    ["teleop_service", "robot_data_service", "cameras"].forEach((serviceKey) => {
        let card = grid.querySelector(`[data-service-key="${serviceKey}"]`);
        if (!card) {
            card = document.createElement("div");
            card.dataset.serviceKey = serviceKey;
            card.className = "status-card teleop-system-card";
            grid.appendChild(card);
        }
        updateTeleopSystemCard(card, serviceKey, services[serviceKey] || {});
    });
}

function updateTeleopSystemCard(card, serviceKey, service = {}) {
    const tone = service.tone || "neutral";
    const resetBusy = !!state.ui.serviceResetBusy[serviceKey];
    const serviceState = formatValue(service.state);
    // Only services backed by a controllable connection expose a reconnect
    // button. The robot data service mirrors the teleoperation (TDK) status and
    // has no connection of its own to reset.
    const canReset = serviceKey in SERVICE_RESET_TARGETS;
    const reconnectMessage = SERVICE_RESET_MESSAGES[serviceKey] || `Reconnect ${service.label || serviceKey}`;
    const label = service.label || serviceKey;
    // Reconnecting the teleop service re-pairs the robots, so it needs all four
    // serials just like the initial connect.
    const serialsReady =
        serviceKey === "teleop_service" ? allRobotSerialsConfigured() : true;
    const resetDisabled = resetBusy || !serialsReady;
    const resetTitle = serialsReady ? reconnectMessage : ROBOT_SERIALS_REQUIRED_MESSAGE;
    const resetButtonMarkup = canReset
        ? `<button class="secondary-button icon-button teleop-system-card__reset ${resetBusy ? "icon-button--spinning" : ""}" type="button" aria-label="${reconnectMessage}" title="${resetTitle}" ${resetDisabled ? "disabled" : ""}>
                ${RESET_ICON_SVG}
            </button>`
        : "";
    // Key on everything that affects the markup so the button node (and its
    // click handler) survive across ticks whenever nothing visible changed.
    const signature = `${tone}:${serviceState}:${label}:${canReset ? "reset" : "noreset"}:${resetBusy ? "busy" : "idle"}:${serialsReady ? "ready" : "blocked"}`;
    const rewritten = setMarkupIfChanged(
        card,
        signature,
        `
        <div class="teleop-system-card__header">
            <div class="teleop-system-card__title">
                <span class="teleop-system-card__dot teleop-system-card__dot--${tone}" role="img" aria-label="${serviceState}" title="${serviceState}"></span>
                <span class="eyebrow teleop-system-card__label">${label}</span>
            </div>
            ${resetButtonMarkup}
        </div>
    `,
    );

    if (rewritten && canReset) {
        const resetButton = card.querySelector("button");
        if (resetButton) {
            resetButton.onclick = () =>
                resetTeleopSystemService(serviceKey).catch((error) => showToast(error.message, true));
        }
    }
}

function createServiceStatusCard(serviceKey, service) {
    const card = document.createElement("div");
    card.className = `status-card status-card--${service.tone || "neutral"} status-card--service`;
    card.innerHTML = `
        <span class="eyebrow">${service.label}</span>
        <h3>${formatValue(service.state)}</h3>
        <p class="status-card__detail">${service.detail || ""}</p>
        <div class="status-card__actions"></div>
    `;
    const actions = card.querySelector(".status-card__actions");
    const definitions = {
        teleop_service: [
            { label: "Connect", serviceName: "teleop", control: "connect", className: "start-button" },
            { label: "Disconnect", serviceName: "teleop", control: "disconnect", className: "stop-button" },
        ],
        cameras: [
            { label: "Connect", serviceName: "cameras", control: "connect", className: "start-button" },
            { label: "Disconnect", serviceName: "cameras", control: "disconnect", className: "stop-button" },
        ],
    };
    (definitions[serviceKey] || []).forEach((definition) => {
        const button = document.createElement("button");
        button.type = "button";
        if (definition.className) {
            button.classList.add(definition.className);
        }
        const connecting = !!state.ui.serviceConnectBusy?.[definition.serviceName];
        if (definition.control === "connect" && connecting) {
            button.classList.add("button--busy");
            button.disabled = true;
            button.innerHTML = `<span class="button-spinner" aria-hidden="true"></span><span>Connecting…</span>`;
        } else {
            button.textContent = definition.label;
            // While a connect is in progress, keep the sibling actions inert too.
            button.disabled = connecting;
        }
        // The teleop service can only connect once all four robot serials are set.
        if (
            definition.serviceName === "teleop" &&
            definition.control === "connect" &&
            !allRobotSerialsConfigured()
        ) {
            button.disabled = true;
            button.title = ROBOT_SERIALS_REQUIRED_MESSAGE;
        }
        button.onclick = () => controlHomeService(definition.serviceName, definition.control);
        actions.appendChild(button);
    });
    return card;
}

function formatElapsed(seconds) {
    const clamped = Math.max(0, Math.floor(Number(seconds) || 0));
    const hh = Math.floor(clamped / 3600);
    const mm = Math.floor((clamped % 3600) / 60);
    const ss = clamped % 60;
    if (hh > 0) {
        return `${String(hh).padStart(2, "0")}:${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
    }
    return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

function setTeleopRefreshBusy(busy) {
    state.ui.teleopRefreshBusy = busy;
    const button = byId("teleop-refresh");
    if (!button) {
        return;
    }
    button.classList.toggle("icon-button--spinning", busy);
    button.setAttribute("aria-busy", busy ? "true" : "false");
}

function setTrainingDeviceEvalBusy(busy) {
    state.ui.trainingDeviceEvalBusy = busy;
    const button = byId("training-device-reload");
    if (!button) {
        return;
    }
    button.disabled = busy;
    button.classList.toggle("icon-button--spinning", busy);
    button.setAttribute("aria-busy", busy ? "true" : "false");
}

function setActiveView(view) {
    state.activeView = view;
    if (view !== "teleoperation") {
        stopAllCameraFeeds();
    }
    document.querySelectorAll(".view").forEach((element) => {
        element.classList.toggle("view--active", element.dataset.view === view);
    });
    document.querySelectorAll("[data-nav]").forEach((element) => {
        const isBrand = element.classList.contains("brand");
        const active = element.dataset.nav === view || (view === "home" && isBrand);
        element.classList.toggle("nav-link--active", active && !isBrand);
        element.classList.toggle("brand--active", active && isBrand);
    });
    if (view === "teleoperation" && state.teleopStatus) {
        renderTeleop();
    }
}

function setComponentLoading(wrapperId, loading, label = "Initializing") {
    const wrapper = byId(wrapperId);
    if (!wrapper) {
        return;
    }
    const existing = wrapper.querySelector(".component-loading-overlay");
    if (loading) {
        if (existing) {
            return;
        }
        const overlay = document.createElement("div");
        overlay.className = "component-loading-overlay";
        overlay.innerHTML = `
            <div class="mini-progress-bar"><span></span></div>
            <span class="component-loading-overlay__label">${label}…</span>
        `;
        wrapper.appendChild(overlay);
    } else {
        if (existing) {
            existing.remove();
        }
    }
}

const ROBOT_SERIALS_REQUIRED_MESSAGE =
    "Enter every leader and follower robot serial number before connecting the teleop service.";

// Teleoperation needs each active arm's leader and follower serial. Require the
// full set for the active arm count before the teleop service can be connected,
// so a half-typed configuration can't start a pairing with a missing arm.
function allRobotSerialsConfigured() {
    const config = state.summary?.robot_config;
    if (!config) return false;
    const count = getActiveSides().length;
    const required = [];
    for (let index = 0; index < count; index += 1) {
        required.push(config.leader_robot_serials?.[index]);
        required.push(config.follower_robot_serials?.[index]);
    }
    return required.every((serial) => typeof serial === "string" && serial.trim() !== "");
}

// Per-side serial labels. In single mode the lone arm sits at index 0, so its
// label comes from the chosen side rather than the dual LEFT/RIGHT order.
function robotSerialSideLabels() {
    return getActiveSides().map((side) => ARM_SIDE_LABELS[side].serial);
}

function renderArmConfig() {
    if (!state.summary) {
        return;
    }
    const config = state.summary.robot_config || { arm_mode: "dual" };
    const armMode = config.arm_mode || "dual";

    const modeContainer = byId("home-arm-mode");
    modeContainer.innerHTML = `
        <label class="robot-input-group">
            <span>Arm Pairs</span>
            <select id="arm-mode-select">
                <option value="dual"${armMode === "dual" ? " selected" : ""}>Dual</option>
                <option value="single"${armMode === "single" ? " selected" : ""}>Single</option>
            </select>
        </label>
    `;
    modeContainer.querySelector("select").onchange = (event) => {
        state.summary.robot_config.arm_mode = event.target.value;
        onArmConfigChanged();
    };
}

// Reconcile the recording checklist against the now-valid entries: drop any
// that are no longer offered, and add any newly-offered default entries (e.g. a
// follower just switched to "gripper", exposing its width/force) while keeping
// the user's existing selections. Order follows the canonical default order.
function reconcileRecordingEntries() {
    const defaults = defaultRecordingEntryIds();
    const valid = new Set(defaults);
    const selected = new Set(
        state.recordingEntries.filter((entry) => valid.has(entry))
    );
    // An entry that has never been offered before is auto-selected (entries are
    // default-on); one that was offered and is absent from the selection was
    // explicitly deselected, so it stays off. `recordingOfferedEntries` tracks
    // what's been offered so a re-render of an unchanged checklist doesn't undo
    // a deselection. On first run nothing has been offered, so the full default
    // set is selected.
    const offered = state.recordingOfferedEntries;
    defaults.forEach((entry) => {
        if (offered === null || !offered.has(entry)) selected.add(entry);
    });
    state.recordingOfferedEntries = valid;
    state.recordingEntries = defaults.filter((entry) => selected.has(entry));
}

// The arm count/side changed: re-derive the recording checklist to the now-valid
// entries, persist, and re-render the affected home controls.
function onArmConfigChanged() {
    reconcileRecordingEntries();
    renderArmConfig();
    renderHomeRobotConfigInputs();
    renderHomeStatus();
    renderHomeEndEffectors();
    queueRobotConfigSave();
}

function renderHomeRobotConfigInputs() {
    if (!state.summary) {
        return;
    }
    const sideLabels = robotSerialSideLabels();
    const count = sideLabels.length;
    const robotConfig = state.summary.robot_config || {
        leader_robot_serials: ["", ""],
        follower_robot_serials: ["", ""],
    };
    const configs = [
        ["home-leader-robots", "leader_robot_serials", robotConfig.leader_robot_serials || []],
        ["home-follower-robots", "follower_robot_serials", robotConfig.follower_robot_serials || []],
    ];
    configs.forEach(([containerId, key, serials]) => {
        const container = byId(containerId);
        container.innerHTML = "";
        for (let index = 0; index < count; index += 1) {
            const serial = serials[index] || "";
            const field = document.createElement("label");
            field.className = "robot-input-group";
            field.innerHTML = `
                <span>${sideLabels[index] || `Robot ${index + 1}`}</span>
                <input type="text" value="${serial}" placeholder="Rizon4s-xxxxxx" />
            `;
            const input = field.querySelector("input");
            input.oninput = () => {
                state.summary.robot_config[key][index] = input.value;
                // Refresh the service cards (not the inputs) so the teleop
                // Connect button enables/disables live as serials are typed.
                renderHomeStatus();
                queueRobotConfigSave();
            };
            container.appendChild(field);
        }
    });
}

function renderHomeStorage() {
    if (!state.summary) {
        return;
    }
    const storage = byId("home-storage");
    storage.innerHTML = "";
    [
        ["UI URL", state.summary.backend.ui_url],
        ["Episodes", state.summary.storage.episodes],
        ["Merged datasets", state.summary.storage.merged],
        ["Training outputs", state.summary.storage.training],
    ].forEach(([label, value]) => {
        const item = document.createElement("div");
        item.className = "kv-item";
        item.innerHTML = `<strong>${label}</strong><span>${formatValue(value)}</span>`;
        storage.appendChild(item);
    });
}

function renderHomeStatus() {
    if (!state.summary) {
        return;
    }
    const status = byId("home-status");
    status.innerHTML = "";
    Object.entries(state.summary.services || {}).forEach(([key, service]) => {
        status.appendChild(createServiceStatusCard(key, service));
    });
}

// Tile titles per arm side. Dual mode shows one tile per side; single mode
// shows a single side-free tile.
const END_EFFECTOR_TILE_LABELS = {
    left_arm: "End Effector Configuration - Left",
    right_arm: "End Effector Configuration - Right",
    single_arm: "End Effector Configuration",
};

const GRIPPER_MODELS = ["Flexiv-GN01", "Robotiq-2F-85", "Robotiq-Hand-E"];

function defaultEndEffectorConfig() {
    return {
        leader: "none", // "none" | "digital_input"
        leader_channel: 0, // DI[0]..DI[15]
        leader_activating_state: "high", // "high" | "low" (digital input)
        follower: "none", // "none" | "digital_output" | "gripper"
        follower_channel: 0, // DO[0]..DO[15]
        follower_activated_state: "high", // "high" | "low" (digital output)
        gripper_model: GRIPPER_MODELS[0],
        gripper_activated_state: "close", // "close" | "open" (gripper)
    };
}

// Selections live inside robot_config.end_effector_config so they ride the same
// PUT /system/robot-config save path as the serial numbers and are cached in
// robot_serials.json across reloads. Lazily seed defaults per side.
function getEndEffectorConfig(side) {
    const robotConfig = state.summary.robot_config || (state.summary.robot_config = {});
    if (!robotConfig.end_effector_config) {
        robotConfig.end_effector_config = {};
    }
    if (!robotConfig.end_effector_config[side]) {
        robotConfig.end_effector_config[side] = defaultEndEffectorConfig();
    }
    return robotConfig.end_effector_config[side];
}

// Build the 16 DI/DO channel <option>s, marking the current selection.
function channelOptionsHtml(prefix, selected) {
    let html = "";
    for (let i = 0; i < 16; i += 1) {
        html += `<option value="${i}"${i === selected ? " selected" : ""}>${prefix}[${i}]</option>`;
    }
    return html;
}

// Build a labelled state dropdown (e.g. Activating/Activated state). `options`
// is a list of [value, label] pairs; `selected` marks the current value.
function stateSelectHtml(title, field, options, selected) {
    const optionsHtml = options
        .map(([value, label]) => `<option value="${value}"${value === selected ? " selected" : ""}>${label}</option>`)
        .join("");
    return `<label class="robot-input-group">
                    <span>${title}</span>
                    <select data-field="${field}">${optionsHtml}</select>
                </label>`;
}

function renderHomeEndEffectors() {
    if (!state.summary) {
        return;
    }
    const container = byId("home-end-effectors");
    if (!container) {
        return;
    }
    container.innerHTML = "";
    getActiveSides().forEach((side) => {
        const cfg = getEndEffectorConfig(side);

        // The leader's secondary dropdowns only appear for a digital input device.
        const leaderExtra =
            cfg.leader === "digital_input"
                ? `<label class="robot-input-group">
                    <span>Digital input channel</span>
                    <select data-field="leader_channel">${channelOptionsHtml("DI", cfg.leader_channel)}</select>
                </label>${stateSelectHtml(
                    "Activating state",
                    "leader_activating_state",
                    [["high", "Port high"], ["low", "Port low"]],
                    cfg.leader_activating_state,
                )}`
                : "";

        // The follower's secondary dropdowns depend on the device kind.
        let followerExtra = "";
        if (cfg.follower === "digital_output") {
            followerExtra = `<label class="robot-input-group">
                    <span>Digital output channel</span>
                    <select data-field="follower_channel">${channelOptionsHtml("DO", cfg.follower_channel)}</select>
                </label>${stateSelectHtml(
                "Activated state",
                "follower_activated_state",
                [["high", "Port high"], ["low", "Port low"]],
                cfg.follower_activated_state,
            )}`;
        } else if (cfg.follower === "gripper") {
            const options = GRIPPER_MODELS.map(
                (model) =>
                    `<option value="${model}"${model === cfg.gripper_model ? " selected" : ""}>${model}</option>`,
            ).join("");
            followerExtra = `<label class="robot-input-group">
                    <span>Gripper model</span>
                    <select data-field="gripper_model">${options}</select>
                </label>${stateSelectHtml(
                "Activated state",
                "gripper_activated_state",
                [["close", "Close"], ["open", "Open"]],
                cfg.gripper_activated_state,
            )}`;
        }

        const tile = document.createElement("section");
        tile.className = "panel end-effector-tile";
        tile.innerHTML = `
            <div class="panel-header">
                <h2>${END_EFFECTOR_TILE_LABELS[side] || "End Effector Configuration"}</h2>
            </div>
            <div class="end-effector-row">
                <label class="robot-input-group">
                    <span>Leader end effector</span>
                    <select data-field="leader">
                        <option value="none"${cfg.leader === "none" ? " selected" : ""}>None</option>
                        <option value="digital_input"${cfg.leader === "digital_input" ? " selected" : ""}>Digital input device</option>
                    </select>
                </label>
                ${leaderExtra}
            </div>
            <div class="end-effector-row">
                <label class="robot-input-group">
                    <span>Follower end effector</span>
                    <select data-field="follower">
                        <option value="none"${cfg.follower === "none" ? " selected" : ""}>None</option>
                        <option value="digital_output"${cfg.follower === "digital_output" ? " selected" : ""}>Digital output device</option>
                        <option value="gripper"${cfg.follower === "gripper" ? " selected" : ""}>Gripper</option>
                    </select>
                </label>
                ${followerExtra}
            </div>
        `;

        tile.querySelectorAll("select[data-field]").forEach((select) => {
            select.onchange = () => {
                const field = select.dataset.field;
                // Look the config up fresh: a completed save replaces
                // state.summary.robot_config, orphaning the `cfg` captured above,
                // so mutating that stale reference would silently no-op.
                const current = getEndEffectorConfig(side);
                current[field] =
                    field === "leader_channel" || field === "follower_channel"
                        ? Number(select.value)
                        : select.value;
                // A follower gripper exposes width/force recording entries (and
                // dropping the gripper retires them); renderRecordingOptions
                // reconciles the checklist itself, so refresh it now rather than
                // waiting for the next poll tick.
                if (field === "follower") {
                    renderRecordingOptions(state.teleopStatus?.recording || {});
                }
                // Re-render so the dependent secondary dropdowns appear/hide,
                // then persist alongside the serial numbers.
                renderHomeEndEffectors();
                queueRobotConfigSave();
            };
        });

        container.appendChild(tile);
    });
}

function renderHome() {
    if (!state.summary) {
        return;
    }
    renderArmConfig();
    renderHomeRobotConfigInputs();
    renderHomeStatus();
    renderHomeEndEffectors();
    renderHomeStorage();
}

async function refreshSummary() {
    state.summary = await api("/system/summary");
    renderHome();
}

function queueRobotConfigSave() {
    window.clearTimeout(state.timers.robotConfigSave);
    state.timers.robotConfigSave = window.setTimeout(async () => {
        try {
            const result = await api("/system/robot-config", {
                method: "PUT",
                body: JSON.stringify(state.summary.robot_config),
            });
            state.summary.robot_config = result.robot_config;
            state.summary.services = result.services;
            renderHomeStatus();
        } catch (error) {
            showToast(error.message, true);
        }
    }, 180);
}

const CAMERA_LOCATION_LABELS = {
    ego: "Egocentric",
    left_wrist: "Left Wrist",
    right_wrist: "Right Wrist",
    wrist: "Wrist",
};

async function loadTeleopCameraConfig() {
    try {
        state.cameraConfig = await api("/teleop/cameras/config");
        renderTeleopCameraConfig();
    } catch (error) {
        // Non-fatal: the panel keeps its placeholder until cameras are reachable.
    }
}

function renderTeleopCameraConfig() {
    const container = byId("teleop-camera-config");
    if (!container) {
        return;
    }
    const config = state.cameraConfig;
    const cameras = config?.cameras || [];
    const devices = config?.devices || [];
    if (cameras.length === 0) {
        container.innerHTML = `<p class="camera-control-empty">No cameras are configured.</p>`;
        return;
    }
    container.innerHTML = "";
    cameras.forEach((camera) => {
        const field = document.createElement("label");
        field.className = "robot-input-group camera-input-group";
        const labelText = CAMERA_LOCATION_LABELS[camera.name] || camera.name;
        const current = camera.device_serial || "";
        const seen = new Set();
        const options = [`<option value=""${current ? "" : " selected"}>N/A</option>`];
        devices.forEach((device) => {
            seen.add(device.serial);
            const selected = device.serial === current ? " selected" : "";
            options.push(`<option value="${device.serial}"${selected}>${device.serial}</option>`);
        });
        if (current && !seen.has(current)) {
            options.push(`<option value="${current}" selected>${current}</option>`);
        }
        field.innerHTML = `
            <span>${labelText}</span>
            <select>${options.join("")}</select>
            <small class="camera-input-caption"></small>
        `;
        const select = field.querySelector("select");
        const caption = field.querySelector(".camera-input-caption");
        const updateCaption = () => {
            caption.textContent = describeCameraSelection(select.value, devices);
        };
        updateCaption();
        select.onchange = () => {
            const value = select.value;
            const previous = camera.device_serial || "";
            let other = null;
            if (value) {
                other = (state.cameraConfig?.cameras || []).find(
                    (entry) => entry !== camera && entry.device_serial === value,
                ) || null;
            } else if (previous) {
                // If there is exactly one other unassigned slot, treat selecting
                // N/A as swapping with that slot.
                const emptyOthers = (state.cameraConfig?.cameras || []).filter(
                    (entry) => entry !== camera && !(entry.device_serial || ""),
                );
                other = emptyOthers.length === 1 ? emptyOthers[0] : null;
            }

            camera.device_serial = value;
            if (other) {
                other.device_serial = previous;
            }
            renderTeleopCameraConfig();
            queueCameraConfigSave();
        };
        container.appendChild(field);
    });
}

function describeCameraSelection(serial, devices) {
    if (!serial) {
        return "Not assigned";
    }
    const device = (devices || []).find((entry) => entry.serial === serial);
    return device ? device.name : "Camera not detected";
}

function queueCameraConfigSave() {
    window.clearTimeout(state.timers.cameraConfigSave);
    state.timers.cameraConfigSave = window.setTimeout(async () => {
        try {
            const serials = {};
            (state.cameraConfig?.cameras || []).forEach((camera) => {
                serials[camera.name] = camera.device_serial || "";
            });
            const result = await api("/teleop/cameras/config", {
                method: "PUT",
                body: JSON.stringify({ serials }),
            });
            if (state.cameraConfig && result.camera_config?.cameras) {
                state.cameraConfig.cameras = result.camera_config.cameras;
            }
            if (state.summary && result.services) {
                state.summary.services = result.services;
                renderHomeStatus();
            }
            renderTeleopCameraConfig();
            if (state.activeView === "teleoperation") {
                await refreshTeleopStatus();
            }
        } catch (error) {
            showToast(error.message, true);
        }
    }, 180);
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function coerceFiniteNumber(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function readVectorFromObject(source, keys) {
    const values = keys.map((key) => coerceFiniteNumber(source?.[key]));
    return values.every((value) => value !== null) ? values : null;
}

function vectorFromArray(values, startIndex = 0) {
    const vector = values
        .slice(startIndex, startIndex + 3)
        .map((value) => coerceFiniteNumber(value));
    return vector.length === 3 && vector.every((value) => value !== null) ? vector : null;
}

function scoreVectorKey(key, kind) {
    const value = String(key || "").toLowerCase();
    let score = 0;
    const preferred = kind === "force"
        ? ["force", "wrench", "contact", "linear"]
        : ["moment", "torque", "wrench", "angular"];
    const discouraged = kind === "force"
        ? ["moment", "torque", "angular"]
        : ["force", "contact", "linear"];
    preferred.forEach((token, index) => {
        if (value.includes(token)) {
            score += 8 - index;
        }
    });
    discouraged.forEach((token, index) => {
        if (value.includes(token)) {
            score -= 5 - index;
        }
    });
    if (kind === "force" && /^(f|force)[_xya-z]*$/.test(value)) {
        score += 4;
    }
    if (kind === "moment" && /^(m|moment|torque|tau)[_xya-z]*$/.test(value)) {
        score += 4;
    }
    return score;
}

function extractCartesianVector(payload, kind, keyHint = "") {
    if (payload === null || payload === undefined) {
        return null;
    }

    const hint = String(keyHint || "").toLowerCase();
    if (Array.isArray(payload)) {
        if (hint.includes("wrench") && payload.length >= 6) {
            return kind === "force" ? vectorFromArray(payload, 0) : vectorFromArray(payload, 3);
        }
        if (kind === "force" && /(force|contact|linear)/.test(hint) && payload.length >= 3) {
            return vectorFromArray(payload, 0);
        }
        if (kind === "moment" && /(moment|torque|angular)/.test(hint) && payload.length >= 3) {
            return vectorFromArray(payload, 0);
        }
        for (const item of payload) {
            const candidate = extractCartesianVector(item, kind, hint);
            if (candidate) {
                return candidate;
            }
        }
        return null;
    }

    if (typeof payload !== "object") {
        return null;
    }

    const directPatterns = kind === "force"
        ? [
            ["fx", "fy", "fz"],
            ["f_x", "f_y", "f_z"],
            ["force_x", "force_y", "force_z"],
            ["forceX", "forceY", "forceZ"],
            ["fX", "fY", "fZ"],
        ]
        : [
            ["mx", "my", "mz"],
            ["m_x", "m_y", "m_z"],
            ["moment_x", "moment_y", "moment_z"],
            ["momentX", "momentY", "momentZ"],
            ["torque_x", "torque_y", "torque_z"],
            ["torqueX", "torqueY", "torqueZ"],
            ["tau_x", "tau_y", "tau_z"],
            ["mX", "mY", "mZ"],
        ];
    let directVector = null;
    for (const pattern of directPatterns) {
        directVector = readVectorFromObject(payload, pattern);
        if (directVector) {
            break;
        }
    }
    if (!directVector && kind === "force" && /(force|wrench|contact|linear)/.test(hint)) {
        directVector = readVectorFromObject(payload, ["x", "y", "z"]);
    }
    if (!directVector && kind === "moment" && /(moment|torque|wrench|angular)/.test(hint)) {
        directVector = readVectorFromObject(payload, ["x", "y", "z"])
            || readVectorFromObject(payload, ["tx", "ty", "tz"]);
    }
    if (directVector) {
        return directVector;
    }

    const preferredKeys = kind === "force"
        ? ["force", "cartesian_force", "linear_force", "contact_force", "wrench"]
        : ["moment", "cartesian_moment", "torque", "cartesian_torque", "angular", "wrench"];
    for (const key of preferredKeys) {
        if (payload[key] !== undefined) {
            const candidate = extractCartesianVector(payload[key], kind, key);
            if (candidate) {
                return candidate;
            }
        }
    }

    const orderedKeys = Object.keys(payload).sort((left, right) => scoreVectorKey(right, kind) - scoreVectorKey(left, kind));
    for (const key of orderedKeys) {
        const candidate = extractCartesianVector(payload[key], kind, key);
        if (candidate) {
            return candidate;
        }
    }
    return null;
}

function readRobotTelemetry(robot) {
    return {
        force: extractCartesianVector(robot?.states, "force", "states")
            || extractCartesianVector(robot?.actions, "force", "actions"),
        moment: extractCartesianVector(robot?.states, "moment", "states")
            || extractCartesianVector(robot?.actions, "moment", "actions"),
    };
}

function getRobotTelemetryByIndex(sideIndex, teleopStatus) {
    const robots = teleopStatus.robot_data?.robots || {};
    const preferredSerial = state.summary?.robot_config?.follower_robot_serials?.[sideIndex];
    if (preferredSerial && robots[preferredSerial]) {
        return { serial: preferredSerial, robot: robots[preferredSerial] };
    }

    const fallback = Object.entries(robots)[sideIndex];
    if (fallback) {
        return { serial: fallback[0], robot: fallback[1] };
    }
    return { serial: preferredSerial || null, robot: null };
}

// One descriptor per active arm, mapped onto the two static teleop columns
// (slot "left" = index 0, "right" = index 1). Single mode yields one descriptor
// in the left slot with neutral camera/labels; the right column is then hidden.
function activeArmPanels() {
    const slots = ["left", "right"];
    return getActiveSides().map((side, index) => ({
        slot: slots[index],
        camera: WRIST_CAMERA_BY_SIDE[side],
        feedTitle: ARM_SIDE_LABELS[side].feed,
        wrenchLabel: ARM_SIDE_LABELS[side].wrench,
        followerIndex: index,
    }));
}

function appendTelemetrySample(side, telemetry) {
    const history = state.telemetryHistory[side] || (state.telemetryHistory[side] = []);
    if (!telemetry.force && !telemetry.moment && !history.length) {
        return history;
    }
    history.push({
        timestamp: Date.now(),
        force: telemetry.force ? [...telemetry.force] : null,
        moment: telemetry.moment ? [...telemetry.moment] : null,
    });
    if (history.length > TELEMETRY_HISTORY_LIMIT) {
        history.splice(0, history.length - TELEMETRY_HISTORY_LIMIT);
    }
    return history;
}

// Default full-scale of each gauge and the increment it auto-scales up by when
// any component exceeds the current range.
const FORCE_GAUGE_DEFAULT_RANGE_N = 10;
const FORCE_GAUGE_STEP_N = 5;
const MOMENT_GAUGE_DEFAULT_RANGE_NM = 1;
const MOMENT_GAUGE_STEP_NM = 1;
const FORCE_BAR_LABELS = ["Fx", "Fy", "Fz"];
const MOMENT_BAR_LABELS = ["Mx", "My", "Mz"];

function computeGaugeRange(history, kind, currentVector, defaultRange, step) {
    let maxAbs = 0;
    const consider = (value) => {
        if (Number.isFinite(value)) {
            maxAbs = Math.max(maxAbs, Math.abs(value));
        }
    };
    history.forEach((sample) => {
        if (sample[kind]) {
            sample[kind].forEach(consider);
        }
    });
    if (currentVector) {
        currentVector.forEach(consider);
    }
    return Math.max(defaultRange, Math.ceil(maxAbs / step) * step);
}

function buildBarGauge(vector, range, colors, labels, unit) {
    const width = 150;
    const height = 248;
    const axisX = 38;
    const right = 6;
    const plotTop = 10;
    const plotBottom = 188;
    const zeroY = (plotTop + plotBottom) / 2;
    const halfHeight = (plotBottom - plotTop) / 2;
    const columnWidth = (width - axisX - right) / 3;
    const barWidth = Math.min(32, columnWidth * 0.66);
    const labelY = 208;
    const chipY = 216;
    const chipHeight = 24;
    const chipWidth = Math.min(columnWidth - 2, 46);
    const safeRange = range > 0 ? range : 1;

    const parts = [
        `<line class="force-gauge__axis-line" x1="${axisX}" y1="${plotTop}" x2="${axisX}" y2="${plotBottom}"></line>`,
        `<line class="force-gauge__zero" x1="${axisX}" y1="${zeroY}" x2="${width - right}" y2="${zeroY}"></line>`,
        `<text class="force-gauge__axis" x="${axisX - 5}" y="${plotTop + 4}" text-anchor="end">${safeRange} ${unit}</text>`,
        `<text class="force-gauge__axis" x="${axisX - 5}" y="${zeroY + 3}" text-anchor="end">0</text>`,
        `<text class="force-gauge__axis" x="${axisX - 5}" y="${plotBottom}" text-anchor="end">-${safeRange} ${unit}</text>`,
    ];

    for (let index = 0; index < 3; index += 1) {
        const raw = vector?.[index];
        const hasValue = Number.isFinite(raw);
        const value = hasValue ? raw : 0;
        const fraction = clamp(Math.abs(value) / safeRange, 0, 1);
        const barHeight = fraction * halfHeight;
        const centerX = axisX + columnWidth * (index + 0.5);
        const x = centerX - barWidth / 2;
        const y = value >= 0 ? zeroY - barHeight : zeroY;
        const color = colors[index];
        parts.push(
            `<rect class="force-gauge__bar" x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${barWidth.toFixed(1)}" height="${Math.max(barHeight, 0.6).toFixed(1)}" rx="3" fill="${color}"></rect>`,
        );
        parts.push(
            `<text class="force-gauge__label" x="${centerX.toFixed(1)}" y="${labelY}" text-anchor="middle" fill="${color}">${labels[index]}</text>`,
        );
        // Value readout in a rounded "chip" directly under its bar.
        parts.push(
            `<rect class="force-gauge__chip" x="${(centerX - chipWidth / 2).toFixed(1)}" y="${chipY}" width="${chipWidth.toFixed(1)}" height="${chipHeight}" rx="8"></rect>`,
        );
        parts.push(
            `<text class="force-gauge__chip-text" x="${centerX.toFixed(1)}" y="${chipY + 16}" text-anchor="middle">${hasValue ? value.toFixed(1) : "--"}</text>`,
        );
    }

    return `<svg class="force-gauge__svg" viewBox="0 0 ${width} ${height}" aria-hidden="true">${parts.join("")}</svg>`;
}

function resolveFpsTone(fps, okMin) {
    if (!Number.isFinite(fps) || fps <= 0) {
        return "offline";
    }
    return fps < okMin ? "warning" : "ok";
}

// Telemetry polls at 10 Hz, but a number that reflows 10×/second is unreadable.
// Refresh the FPS text at most this often so the eye can actually catch it.
const FPS_BADGE_REFRESH_MS = 1000;
const fpsBadgeRenderState = new WeakMap();

function setFpsBadge(element, fps, okMin) {
    if (!element) {
        return;
    }
    const safeFps = Number.isFinite(fps) && fps > 0 ? fps : 0;
    const tone = resolveFpsTone(safeFps, okMin);
    const now = Date.now();
    const last = fpsBadgeRenderState.get(element);
    // Throttle the number itself, but repaint immediately when the tone changes
    // (e.g. camera goes offline/warning) so status stays responsive.
    if (last && last.tone === tone && now - last.renderedAt < FPS_BADGE_REFRESH_MS) {
        return;
    }
    fpsBadgeRenderState.set(element, { tone, renderedAt: now });
    element.className = `feed__fps feed__fps--${tone}`;
    element.innerHTML = `
        <span class="feed__fps-dot" aria-hidden="true"></span>
        <span>${safeFps.toFixed(1)} FPS</span>
    `;
}

function buildAwaitingDataMarkup() {
    return `
        <div class="telemetry-empty-state telemetry-empty-state--breathing">
            <span class="telemetry-empty-state__icon recording-status__icon recording-status__icon--loading" aria-hidden="true">
                <span class="loading-wheel">${LOADING_WHEEL_SEGMENTS}</span>
            </span>
            <span class="telemetry-empty-state__text">Awaiting data</span>
        </div>
    `;
}

function buildCameraFrameUrl(cameraName) {
    return `/teleop/cameras/${encodeURIComponent(cameraName)}/frame?ts=${Date.now()}`;
}

function stopCameraFeed(cameraName) {
    const feed = state.cameraFeeds[cameraName];
    if (feed) {
        feed.stopped = true;
        if (feed.retryTimer) {
            window.clearTimeout(feed.retryTimer);
        }
    }
    delete state.cameraFeeds[cameraName];
}

function stopAllCameraFeeds() {
    Object.keys(state.cameraFeeds).forEach((name) => stopCameraFeed(name));
}

function restoreAwaitingCameraPlaceholder(placeholder) {
    if (!placeholder) {
        return;
    }
    placeholder.classList.add("feed__placeholder--awaiting");
    placeholder.dataset.renderMode = "awaiting";
    delete placeholder.dataset.cameraName;
    placeholder.innerHTML = buildAwaitingDataMarkup();
}

function startCameraFeedPump(cameraName, image) {
    const feed = state.cameraFeeds[cameraName];
    if (!feed || feed.stopped) {
        return;
    }

    const nextUrl = buildCameraFrameUrl(cameraName);
    feed.lastUrl = nextUrl;

    image.onload = () => {
        if (feed.stopped || state.cameraFeeds[cameraName] !== feed) {
            return;
        }
        feed.failed = false;
        // Pump: request next frame immediately; actual FPS is limited by server response time
        window.setTimeout(() => startCameraFeedPump(cameraName, image), 0);
    };

    image.onerror = () => {
        if (feed.stopped || state.cameraFeeds[cameraName] !== feed) {
            return;
        }
        feed.failed = true;
        // Retry after a short delay — do NOT destroy the img element
        feed.retryTimer = window.setTimeout(() => startCameraFeedPump(cameraName, image), 1000);
    };

    image.src = nextUrl;
}

function ensureCameraFeedRunning(cameraName, image) {
    const existing = state.cameraFeeds[cameraName];
    if (existing && !existing.stopped && existing.image === image) {
        // Pump is already running for this image element
        return;
    }
    // Stop old feed if any
    stopCameraFeed(cameraName);
    const feed = { lastUrl: "", failed: false, stopped: false, retryTimer: null, image };
    state.cameraFeeds[cameraName] = feed;
    startCameraFeedPump(cameraName, image);
}

function renderCameraFps(elementId, cameraName, camera) {
    const element = byId(elementId);
    if (!element) {
        return;
    }

    const fps = Number(camera?.fps || 0);
    setFpsBadge(element, fps, 29);

    const feed = element.closest(".feed");
    const placeholder = feed?.querySelector(".feed__placeholder");
    if (!placeholder) {
        return;
    }

    if (!camera?.started) {
        stopCameraFeed(cameraName);
        if (placeholder.dataset.renderMode !== "awaiting") {
            restoreAwaitingCameraPlaceholder(placeholder);
        }
        return;
    }

    placeholder.classList.remove("feed__placeholder--awaiting");
    const cameraTitle = feed?.querySelector(".feed__title")?.textContent?.trim() || cameraName;
    if (placeholder.dataset.renderMode !== "live" || placeholder.dataset.cameraName !== cameraName) {
        placeholder.innerHTML = `<img class="feed__image" alt="${cameraTitle} live feed" />`;
        placeholder.dataset.renderMode = "live";
        placeholder.dataset.cameraName = cameraName;
    }

    const image = placeholder.querySelector(".feed__image");
    if (image) {
        ensureCameraFeedRunning(cameraName, image);
    }
}

function renderRecordingOptions(recording = {}) {
    const container = byId("recording-options");
    if (!container) {
        return;
    }

    const locked = !!recording.active || !!recording.awaiting_save || state.ui.recordingStartBusy;
    // Reconcile first so a gripper enabled after page load (or restored from a
    // saved config) adds its width/force entries to the default-on selection,
    // and a removed gripper retires them, before the checklist is derived.
    reconcileRecordingEntries();
    const options = recordingEntryOptions();
    const defaultIds = options.map((option) => option.id);
    const selected = new Set(state.recordingEntries);
    const allSelected = defaultIds.every((entryId) => selected.has(entryId));

    // renderTeleop runs on every poll tick (~10x/sec). Rebuilding these
    // controls unconditionally made the Select All button and checkboxes flash
    // and swallowed clicks landing mid-rerender. Only rebuild when something
    // that affects the rendered output actually changed -- including the set of
    // offered options, which grows/shrinks with the configured gripper sides.
    const renderKey = `${locked ? 1 : 0}|${getActiveSides().join("+")}|${defaultIds.join(",")}|${state.recordingEntries.join(",")}`;
    if (container.dataset.renderKey === renderKey) {
        return;
    }
    container.dataset.renderKey = renderKey;
    container.innerHTML = "";

    const selectAllButton = document.createElement("button");
    selectAllButton.className = "secondary-button toggle-all-button recording-select-all-button";
    selectAllButton.type = "button";
    selectAllButton.disabled = locked;
    setToggleAllButton(selectAllButton, allSelected);
    selectAllButton.onclick = () => {
        state.recordingEntries = allSelected ? [] : [...defaultIds];
        renderRecordingOptions(recording);
        renderRecordingStatusPanel(state.teleopStatus);
    };
    container.appendChild(selectAllButton);

    let currentGroup = null;
    options.forEach((option) => {
        // Section header whenever the feature group changes, so the list reads
        // as observation.images / observation.state / action.
        if (option.group && option.group !== currentGroup) {
            currentGroup = option.group;
            const heading = document.createElement("div");
            heading.className = "recording-option-group-title";
            heading.textContent = option.group;
            container.appendChild(heading);
        }
        const checked = selected.has(option.id);
        const label = document.createElement("label");
        label.className = `recording-option ${locked ? "recording-option--disabled" : ""}`;
        label.innerHTML = `
            <input type="checkbox" ${checked ? "checked" : ""} ${locked ? "disabled" : ""} />
            <span class="recording-option__text">
                <span class="recording-option__label">${option.label}</span>
            </span>
        `;
        const input = label.querySelector("input");
        input.onchange = () => {
            const nextSelected = new Set(state.recordingEntries);
            if (input.checked) {
                nextSelected.add(option.id);
            } else {
                nextSelected.delete(option.id);
            }
            state.recordingEntries = defaultIds.filter((entryId) => nextSelected.has(entryId));
            renderRecordingOptions(recording);
            renderRecordingStatusPanel(state.teleopStatus);
        };
        container.appendChild(label);
    });
}

function hasRecordingPayload(value) {
    if (Array.isArray(value)) {
        return value.some((item) => item !== null && item !== undefined);
    }
    if (value && typeof value === "object") {
        return Object.keys(value).length > 0;
    }
    return value !== null && value !== undefined;
}

function getConfiguredFollowerSerials() {
    return (state.summary?.robot_config?.follower_robot_serials || [])
        .map((serial) => String(serial || "").trim())
        .filter(Boolean);
}

function areSelectedRecordingEntriesAvailable(teleopStatus) {
    const selectedOptions = recordingEntryOptions().filter((option) => state.recordingEntries.includes(option.id));
    if (!selectedOptions.length) {
        return false;
    }

    const configuredFollowerSerials = getConfiguredFollowerSerials();
    return selectedOptions.every((option) => {
        if (option.bucket === "image") {
            const camera = teleopStatus?.cameras?.cameras?.[option.sourceField];
            return !!camera?.started;
        }

        // Each arm feature maps to one follower robot (side 0 = left, 1 = right).
        const serial = configuredFollowerSerials[option.side];
        if (!serial) {
            return false;
        }
        const robot = teleopStatus?.robot_data?.robots?.[serial];
        const payload = robot?.[option.payload];
        return hasRecordingPayload(payload?.[option.verifyField]);
    });
}

function recordingStatusIconMarkup(kind) {
    if (kind === "ready") {
        return `<span class="recording-status__icon recording-status__icon--ready" aria-hidden="true">${CHECK_ICON_SVG}</span>`;
    }
    if (kind === "recording") {
        return `
            <span class="recording-status__icon recording-status__icon--recording" aria-hidden="true">
                <span class="recording-live__dot recording-status__dot"></span>
            </span>
        `;
    }
    if (kind === "stopped") {
        return `
            <span class="recording-status__icon recording-status__icon--stopped" aria-hidden="true">
                ${STOP_SQUARE_ICON_SVG}
            </span>
        `;
    }
    return `
        <span class="recording-status__icon recording-status__icon--loading" aria-hidden="true">
            <span class="loading-wheel">${LOADING_WHEEL_SEGMENTS}</span>
        </span>
    `;
}

// Derive a smoothed realtime recording FPS from how many frames landed between
// status polls. Cumulative frames/elapsed would lag behind the current rate, so
// we track the delta since the previous poll and smooth it with an EMA. Returns
// 0 until enough time has elapsed to take a stable first measurement.
function updateRecordingFps(frames, seconds) {
    const prev = state.ui.recordingFpsSample;
    if (!prev || frames < prev.frames || seconds < prev.seconds) {
        // First sample of a (re)started recording — seed without a rate yet.
        state.ui.recordingFpsSample = { frames, seconds };
        state.ui.recordingFps = 0;
        return 0;
    }
    const deltaFrames = frames - prev.frames;
    const deltaSeconds = seconds - prev.seconds;
    if (deltaSeconds >= 0.25) {
        const instant = deltaFrames / deltaSeconds;
        const alpha = 0.4;
        state.ui.recordingFps =
            state.ui.recordingFps > 0
                ? alpha * instant + (1 - alpha) * state.ui.recordingFps
                : instant;
        state.ui.recordingFpsSample = { frames, seconds };
    }
    return state.ui.recordingFps;
}

function resetRecordingFps() {
    state.ui.recordingFpsSample = null;
    state.ui.recordingFps = 0;
}

function buildRecordingStatusModel(teleopStatus) {
    const recording = teleopStatus?.recording || {};
    const active = !!recording.active;
    const awaitingSave = !!recording.awaiting_save;
    const frames = Number(recording.frames_captured || 0);
    const fps = Number(recording.fps || 0);
    const elapsedCandidate = Number(recording.elapsed_s);
    const seconds = Number.isFinite(elapsedCandidate)
        ? Math.max(0, elapsedCandidate)
        : (fps > 0 ? frames / fps : 0);

    if (active) {
        // Surface capture-loop errors instead of silently showing "0 frames":
        // a failing add_frame keeps the timer ticking while no frames land.
        const captureError =
            typeof recording.error === "string" ? recording.error.trim() : "";
        const liveFps = updateRecordingFps(frames, seconds);
        return {
            kind: "recording",
            line1: formatElapsed(seconds),
            line2:
                captureError && frames === 0
                    ? `Capture error: ${captureError}`
                    : `${frames} frames captured`,
            fps: liveFps > 0 ? liveFps : null,
            animated: false,
            canStart: false,
            canStop: true,
        };
    }

    resetRecordingFps();

    if (awaitingSave) {
        return {
            kind: "stopped",
            line1: formatElapsed(seconds),
            line2: `${frames} frames captured`,
            animated: false,
            canStart: false,
            canStop: false,
        };
    }

    if (state.ui.recordingStartBusy || !state.teleopBootstrapped || !teleopStatus) {
        return {
            kind: "initializing",
            line1: "Initializing recorder",
            line2: null,
            animated: true,
            canStart: false,
            canStop: false,
        };
    }

    if (state.recordingEntries.length > 0 && areSelectedRecordingEntriesAvailable(teleopStatus)) {
        return {
            kind: "ready",
            line1: "Ready to record",
            line2: null,
            animated: false,
            canStart: true,
            canStop: false,
        };
    }

    return {
        kind: "awaiting-data",
        line1: "Awaiting data",
        line2: null,
        animated: true,
        canStart: false,
        canStop: false,
    };
}

function renderRecordingStatusPanel(teleopStatus) {
    const status = byId("recording-status");
    if (!status) {
        return;
    }

    const model = buildRecordingStatusModel(teleopStatus);
    const nextClassName = `recording-status ${model.animated ? "recording-status--breathing" : ""}`.trim();
    if (status.className !== nextClassName) {
        status.className = nextClassName;
    }
    // Show the realtime capture rate right next to the timer while recording.
    const fpsBadge =
        model.fps != null
            ? `<span class="recording-status__fps">${model.fps.toFixed(1)} FPS</span>`
            : "";
    const primaryLine = fpsBadge
        ? `<span class="recording-status__primary"><span>${model.line1}</span>${fpsBadge}</span>`
        : model.line1;
    const textMarkup = model.line2
        ? `
            <span class="recording-status__text recording-status__text--stacked">
                <span class="recording-status__line">${primaryLine}</span>
                <span class="recording-status__line recording-status__line--secondary">${model.line2}</span>
            </span>
        `
        : `<span class="recording-status__text">${primaryLine}</span>`;
    const fpsKey = model.fps != null ? model.fps.toFixed(1) : "";
    setMarkupIfChanged(
        status,
        `${model.kind}:${model.line1}:${fpsKey}:${model.line2 || ""}:${model.animated ? "animated" : "static"}`,
        `
            ${recordingStatusIconMarkup(model.kind)}
            ${textMarkup}
        `,
    );

    updateRecordingJobNameField(model);
    updateRecordingToggleButton(model);
}

// The Job name box names the training job the next episode is filed under
// (episodes/<job_name>/). It is locked while an episode is actively recording
// or awaiting save/discard so the job cannot change mid-session, and re-enabled
// once the recorder is idle.
function updateRecordingJobNameField(model) {
    const field = byId("record-job-name");
    if (!field) {
        return;
    }
    // "recording" => active capture; "stopped" => awaiting save/discard.
    const locked =
        model.canStop ||
        model.kind === "recording" ||
        model.kind === "stopped" ||
        !!state.ui.recordingStartBusy;
    field.disabled = locked;
}

function updateRecordingToggleButton(model) {
    // A single button toggles between Start and Stop based on whether a
    // recording is in progress, mirroring the teleop power control. While the
    // recorder is initializing, show a loading state.
    const toggle = byId("record-toggle");
    if (!toggle) {
        return;
    }
    const starting = !!state.ui.recordingStartBusy;
    const isRecording = !!model.canStop;
    // Only rewrite innerHTML when the rendered content actually changes;
    // renderTeleop runs on every poll tick, and unconditional rewrites make the
    // button flash and drop clicks that land mid-rerender.
    if (starting) {
        toggle.disabled = true;
        toggle.classList.add("start-button");
        toggle.classList.remove("stop-button");
        setMarkupIfChanged(toggle, "record-toggle:starting", RECORD_STARTING_MARKUP);
        return;
    }
    toggle.disabled = isRecording ? !model.canStop : !model.canStart;
    toggle.classList.toggle("start-button", !isRecording);
    toggle.classList.toggle("stop-button", isRecording);
    setMarkupIfChanged(
        toggle,
        isRecording ? "record-toggle:stop" : "record-toggle:start",
        isRecording ? RECORD_STOP_MARKUP : RECORD_START_MARKUP,
    );
}

function renderRecordingActionButtons(recording = {}) {
    const saveButton = byId("record-save");
    const discardButton = byId("record-discard");
    if (!saveButton || !discardButton) {
        return;
    }

    const backendBusy = !!recording.save_in_progress;
    const backendProgress = normalizePercent(recording.save_progress || 0);
    if (backendBusy) {
        state.ui.recordingSaveProgress = Math.max(state.ui.recordingSaveProgress, backendProgress);
    }

    const busy = state.ui.recordingSaveBusy || backendBusy;
    const progress = normalizePercent(Math.max(state.ui.recordingSaveProgress, backendProgress));

    saveButton.disabled = busy;
    discardButton.disabled = busy;

    if (busy) {
        saveButton.classList.add("record-save-button--progress");
        setMarkupIfChanged(
            saveButton,
            `save-progress:${progress}`,
            `
                <span class="record-save-button__track" aria-hidden="true">
                    <span class="record-save-button__fill" style="width: ${progress}%"></span>
                </span>
                <span class="record-save-button__label">Saving ${progress}%</span>
            `,
        );
    } else {
        saveButton.classList.remove("record-save-button--progress");
        setMarkupIfChanged(saveButton, "save-default", RECORD_SAVE_DEFAULT_MARKUP);
    }

    setMarkupIfChanged(discardButton, "discard-default", RECORD_DISCARD_DEFAULT_MARKUP);
}

function updateTeleopControlButtons(teleopStatus) {
    const teleop = teleopStatus?.teleop || {};
    // A fault (incl. soft fault where any_fault() is true) does NOT block Start:
    // Init() clears any clearable fault as part of bringing the loop up, so the
    // operator must be able to click Start to recover. Faults still gate Home
    // and Engage below, which run against an already-initialized robot.
    const teleopReady = !!teleop.initialized && !teleop.started && !teleop.error;
    const teleopResetBusy = !!state.ui.serviceResetBusy.teleop_service;
    const canStart = teleopReady && !state.ui.teleopHomeBusy && !teleopResetBusy;
    const canStop = !!teleop.started || state.ui.teleopHomeBusy;
    // Home is gated behind Stop: the loop must have been started and then
    // stopped (robots initialized, loop halted) before HomeAll() is allowed.
    const canHome = !!teleop.can_home && !teleop.fault && !state.ui.teleopHomeBusy && !teleopResetBusy;
    // Engaging requires the control loop to be running; the button toggles
    // between engage and disengage based on the current engagement state.
    const canEngage = !!teleop.started && !state.ui.teleopHomeBusy && !teleopResetBusy;

    // A single button toggles between Start and Stop based on whether the
    // control loop is running. While starting, Init() blocks (it zeroes the
    // F/T sensors), so show a loading state and an orange warning banner.
    const isRunning = !!teleop.started;
    const starting = !!state.ui.teleopStartBusy;
    const powerButton = byId("teleop-power");
    // Only rewrite innerHTML when the rendered content actually changes.
    // renderTeleop runs on every poll tick (~10x/sec); replacing innerHTML
    // unconditionally made the button flash and dropped clicks that landed
    // mid-rerender (the clicked node gets detached before the click fires).
    if (starting) {
        powerButton.disabled = true;
        powerButton.classList.add("start-button");
        powerButton.classList.remove("stop-button");
        setMarkupIfChanged(powerButton, "teleop-power:starting", TELEOP_STARTING_MARKUP);
    } else {
        powerButton.disabled = isRunning ? !canStop : !canStart;
        powerButton.classList.toggle("start-button", !isRunning);
        powerButton.classList.toggle("stop-button", isRunning);
        setMarkupIfChanged(
            powerButton,
            isRunning ? "teleop-power:stop" : "teleop-power:start",
            isRunning ? TELEOP_STOP_MARKUP : TELEOP_START_MARKUP,
        );
    }

    const startWarning = byId("teleop-start-warning");
    if (startWarning) {
        // The banner announces sensor zeroing, so only show it while starting
        // with zeroing actually requested.
        startWarning.classList.toggle("hidden", !(starting && state.ui.teleopZeroingSensors));
    }

    byId("teleop-home").disabled = !canHome;

    const engageButton = byId("teleop-engage");
    engageButton.disabled = !canEngage;
    setMarkupIfChanged(
        engageButton,
        teleop.engaged ? "teleop-engage:disengage" : "teleop-engage:engage",
        teleop.engaged ? TELEOP_DISENGAGE_MARKUP : TELEOP_ENGAGE_MARKUP,
    );
}

// The Gripper Control panel appears when any active arm has a gripper selected
// as its follower end effector (sides with other follower types get no panel).
// An Init button at the top enables every gripper, switches its tool, and
// triggers Gripper.Init() -- runnable only while teleop is NOT started, since
// Tool.Switch() requires the follower to be IDLE. Once initialized, each gripper
// side gets a grasping-velocity and grasping-force slider (read from that
// gripper's reported range, cached until disconnect). The leader-mirror thread
// runs with the teleop control loop (Start/Stop); a configured-but-uninitialized
// gripper is skipped by the thread and flagged with a warning here.
const GRIPPER_SIDE_LABELS = {
    left_arm: "Left",
    right_arm: "Right",
    single_arm: "",
};

// After Init triggers gripper.Init() on the backend, wait this long for the
// gripper hardware to finish initializing before unblocking the panel.
const GRIPPER_INIT_WAIT_MS = 10000;

function gripperConfiguredSides() {
    const eec = state.summary?.robot_config?.end_effector_config || {};
    return getActiveSides().filter((side) => eec[side]?.follower === "gripper");
}

function roundForInput(value, decimals) {
    const factor = 10 ** decimals;
    return Math.round(value * factor) / factor;
}

function clampNumber(value, min, max) {
    if (Number.isNaN(value)) return min;
    return Math.min(max, Math.max(min, value));
}

function renderGripperControl(teleopStatus) {
    const panel = byId("teleop-gripper-panel");
    const content = byId("teleop-gripper-content");
    if (!panel || !content) {
        return;
    }
    const sides = gripperConfiguredSides();
    panel.classList.toggle("hidden", sides.length === 0);
    if (sides.length === 0) {
        state.ui.gripperPanelSignature = null;
        return;
    }

    const grippers = teleopStatus?.gripper || {};
    // Rebuild the panel structure (and re-seed the sliders) only when the set of
    // sides or their params availability changes, so dragging a slider isn't
    // clobbered by the ~10Hz status poll.
    const signature = sides
        .map((side) => `${side}:${grippers[side] ? "params" : "none"}`)
        .join(",");
    if (state.ui.gripperPanelSignature !== signature) {
        state.ui.gripperPanelSignature = signature;
        buildGripperControlPanel(content, sides, grippers);
    }

    const teleop = teleopStatus?.teleop || {};
    updateGripperInitButton(teleop);
    sides.forEach((side) =>
        updateGripperControlBlock(side, grippers[side], teleop),
    );
}

function buildGripperControlPanel(content, sides, grippers) {
    content.innerHTML = "";

    // Panel-level Init button: enables/switches/inits/reads-params for all
    // grippers. Only valid while teleop is not started (Tool.Switch is IDLE-only).
    const initButton = document.createElement("button");
    initButton.type = "button";
    initButton.className = "secondary-button gripper-init";
    initButton.textContent = "Init";
    initButton.onclick = () => initGrippers();
    content.appendChild(initButton);

    const multiple = sides.length > 1;
    sides.forEach((side) => {
        const params = grippers[side];
        const label = GRIPPER_SIDE_LABELS[side] ?? "";
        const block = document.createElement("div");
        block.className = "gripper-control-block";
        block.dataset.side = side;
        block.innerHTML = `
            ${multiple && label ? `<div class="gripper-control-title">${label}</div>` : ""}
            <label class="gripper-input-group">
                <span>Grasping velocity: <strong class="gripper-velocity-value"></strong> m/s</span>
                <div class="gripper-input-row">
                    <input type="range" class="gripper-velocity" step="0.001" />
                    <input type="number" class="gripper-number-input gripper-velocity-number" step="0.001" aria-label="Grasping velocity (m/s)" />
                </div>
                <small class="gripper-input-note"></small>
            </label>
            <label class="gripper-input-group">
                <span>Grasping force: <strong class="gripper-force-value"></strong> N</span>
                <div class="gripper-input-row">
                    <input type="range" class="gripper-force" step="0.1" />
                    <input type="number" class="gripper-number-input gripper-force-number" step="0.1" aria-label="Grasping force (N)" />
                </div>
                <small class="gripper-input-note"></small>
            </label>
            <p class="gripper-control-hint"></p>
        `;
        content.appendChild(block);

        const velInput = block.querySelector(".gripper-velocity");
        const forceInput = block.querySelector(".gripper-force");
        const velNumber = block.querySelector(".gripper-velocity-number");
        const forceNumber = block.querySelector(".gripper-force-number");

        if (params) {
            const store = (state.gripperControl[side] ||= {});
            const cached = loadCachedGripperParams(side);
            // Seed from the cached last-used value, then the model default
            // (velocity = max_vel; grasping force = 1/4 of max_force). The clamp
            // below keeps a cached value valid if the gripper's range changed.
            if (store.velocity == null) {
                store.velocity = cached.velocity != null ? cached.velocity : params.max_vel;
            }
            if (store.force == null) {
                store.force = cached.force != null ? cached.force : params.max_force / 4;
            }
            // Clamp any carried-over value into the (possibly new) range.
            store.velocity = clampNumber(store.velocity, params.min_vel, params.max_vel);
            store.force = clampNumber(store.force, params.min_force, params.max_force);

            bindGripperControl(block, velInput, velNumber, side, "velocity", params.min_vel, params.max_vel, 3);
            bindGripperControl(block, forceInput, forceNumber, side, "force", params.min_force, params.max_force, 1);
        } else {
            // Params (and thus the valid range) are obtained on Init; leave the
            // controls disabled until then.
            velInput.disabled = true;
            forceInput.disabled = true;
            velNumber.disabled = true;
            forceNumber.disabled = true;
        }
    });
}

// Configure a range slider plus its number box for one gripper field: set their
// bounds, reflect the stored value, keep the two inputs in sync, live-update the
// readout while editing, and push to the backend when committed (change).
// ``decimals`` controls the displayed precision.
function bindGripperControl(block, slider, number, side, field, min, max, decimals) {
    const valueEl = block.querySelector(`.gripper-${field}-value`);
    const note = block.querySelector(`.gripper-${field}-number`).closest(".gripper-input-group").querySelector(".gripper-input-note");
    const store = (state.gripperControl[side] ||= {});
    const step = 1 / 10 ** decimals;
    slider.min = min;
    slider.max = max;
    number.min = roundForInput(min, decimals);
    number.max = roundForInput(max, decimals);
    number.step = step;

    // Reflect the stored value across the readout, slider, and number box.
    const render = () => {
        const rounded = roundForInput(store[field], decimals);
        if (valueEl) valueEl.textContent = rounded;
        slider.value = store[field];
        // Don't fight the field the user is typing in (e.g. a partial "0.").
        if (document.activeElement !== number) {
            number.value = rounded;
        }
    };
    store[field] = clampNumber(store[field], min, max);
    render();
    if (note) {
        note.textContent = `Range: ${roundForInput(min, decimals)}–${roundForInput(max, decimals)}`;
    }

    // Live-update (no backend push) while dragging the slider / typing a number.
    const onInput = (rawValue) => {
        const value = clampNumber(parseFloat(rawValue), min, max);
        store[field] = value;
        const rounded = roundForInput(value, decimals);
        if (valueEl) valueEl.textContent = rounded;
        return value;
    };
    slider.oninput = () => {
        const value = onInput(slider.value);
        if (document.activeElement !== number) {
            number.value = roundForInput(value, decimals);
        }
    };
    number.oninput = () => {
        // Allow intermediate, not-yet-parseable input without snapping it; only
        // mirror to the slider once it parses to a number.
        if (number.value.trim() === "" || Number.isNaN(parseFloat(number.value))) {
            return;
        }
        const value = onInput(number.value);
        slider.value = value;
    };

    // Commit: clamp, normalize the displayed value, persist, and push to backend.
    // ``store[field]`` is kept current by the oninput handlers above; fall back
    // to it when the number box holds an unparseable intermediate value.
    const commit = () => {
        const parsed = parseFloat(number.value);
        const base = Number.isNaN(parsed) ? store[field] : parsed;
        store[field] = clampNumber(base, min, max);
        render();
        persistGripperParams(side);
        // Push to the backend so the mirror loop's Move() uses the new value.
        sendGripperParams(side);
    };
    slider.onchange = commit;
    number.onchange = commit;
}

// Cache this side's last-used velocity/force so they persist across reloads.
function persistGripperParams(side) {
    const store = state.gripperControl[side] || {};
    if (store.velocity == null || store.force == null) {
        return;
    }
    saveCachedGripperParams(side, store.velocity, store.force);
}

// Persist this side's velocity/force on the backend (used by the mirror loop's
// Move() calls). Best-effort; surfaced on failure.
async function sendGripperParams(side) {
    const store = state.gripperControl[side] || {};
    if (store.velocity == null || store.force == null) {
        return;
    }
    try {
        await api(`/teleop/gripper/${side}/params`, {
            method: "POST",
            body: JSON.stringify({ velocity: store.velocity, force: store.force }),
        });
    } catch (error) {
        showToast(error.message, true);
    }
}

function updateGripperControlBlock(side, params, teleop) {
    const block = byId("teleop-gripper-content")?.querySelector(
        `.gripper-control-block[data-side="${side}"]`,
    );
    if (!block) {
        return;
    }
    const hint = block.querySelector(".gripper-control-hint");
    const started = !!teleop.started;
    const initializing = !!state.ui.gripperInitBusy;

    let message = "";
    let warn = false;
    if (initializing) {
        message = "Initializing gripper …";
    } else if (!params && started) {
        // Configured but not initialized while teleop runs: the mirror thread
        // skips it. Surface a warning -- Init requires stopping teleop first.
        message =
            "Gripper not initialized — it cannot be controlled. Stop teleop, then click Init.";
        warn = true;
    } else if (!params) {
        message = "Click Init to enable this gripper.";
    }
    hint.textContent = message;
    hint.classList.toggle("hidden", message === "");
    hint.classList.toggle("gripper-control-warning", warn);
}

function updateGripperInitButton(teleop) {
    const button = byId("teleop-gripper-content")?.querySelector(".gripper-init");
    if (!button) {
        return;
    }
    const initialized = !!teleop.initialized;
    const started = !!teleop.started;
    const busy = !!state.ui.gripperInitBusy;

    // Init must run after Connect but before Start (Tool.Switch is IDLE-only).
    button.disabled = !initialized || started || busy;
    button.classList.toggle("button--busy", busy);
    setMarkupIfChanged(
        button,
        `gripper-init:${busy ? "busy" : "idle"}`,
        busy
            ? '<span class="button-spinner" aria-hidden="true"></span><span>Initializing …</span>'
            : "Init",
    );
    button.title = !initialized
        ? "Connect teleoperation first"
        : started
          ? "Stop teleoperation before initializing grippers"
          : "";
}

async function initGrippers() {
    if (state.ui.gripperInitBusy) {
        return;
    }
    state.ui.gripperInitBusy = true;
    renderTeleop();
    try {
        await api("/teleop/gripper/init", { method: "POST" });
        // refreshTeleopStatus() rebuilds the panel and seeds the default
        // velocity/force; push those so the mirror loop uses them even before
        // the user touches a slider.
        await refreshTeleopStatus();
        await Promise.all(
            gripperConfiguredSides().map((side) => sendGripperParams(side)),
        );
        // The backend triggered gripper.Init(); wait for it to physically finish
        // before unblocking the panel (busy stays true, so the Init button keeps
        // spinning throughout).
        await new Promise((resolve) =>
            window.setTimeout(resolve, GRIPPER_INIT_WAIT_MS),
        );
    } catch (error) {
        showToast(error.message, true);
    } finally {
        state.ui.gripperInitBusy = false;
        renderTeleop();
    }
}

function renderForcePanel(side, robotEntry, telemetry, history, wrenchLabel) {
    const panel = byId(`${side}-force-panel`);
    if (!panel) {
        return;
    }

    const title = `${wrenchLabel || side.toUpperCase()} CARTESIAN WRENCH`;
    if (!telemetry.force) {
        setMarkupIfChanged(
            panel,
            `${side}:force:awaiting`,
            `
                <div class="telemetry-card__header">
                    <div>
                        <span class="eyebrow">${title}</span>
                    </div>
                </div>
                <div class="vector-panel vector-panel--empty">
                    ${buildAwaitingDataMarkup()}
                </div>
            `,
        );
        return;
    }

    // Per-axis signed bars for the full wrench: force (N) and moment (Nm) shown
    // as two side-by-side gauges, each with its own auto-scaling range.
    const forceRange = computeGaugeRange(history, "force", telemetry.force, FORCE_GAUGE_DEFAULT_RANGE_N, FORCE_GAUGE_STEP_N);
    const momentRange = computeGaugeRange(history, "moment", telemetry.moment, MOMENT_GAUGE_DEFAULT_RANGE_NM, MOMENT_GAUGE_STEP_NM);
    delete panel.dataset.renderKey;
    panel.innerHTML = `
        <div class="telemetry-card__header">
            <div>
                <span class="eyebrow">${title}</span>
            </div>
        </div>
        <div class="vector-panel vector-panel--live">
            <div class="wrench-gauges">
                <div class="wrench-gauge">
                    ${buildBarGauge(telemetry.force, forceRange, TELEMETRY_SERIES.force.colors, FORCE_BAR_LABELS, TELEMETRY_SERIES.force.units)}
                </div>
                <div class="wrench-gauge">
                    ${buildBarGauge(telemetry.moment, momentRange, TELEMETRY_SERIES.moment.colors, MOMENT_BAR_LABELS, TELEMETRY_SERIES.moment.units)}
                </div>
            </div>
        </div>
    `;
}

function computeTelemetryScale(history, kind) {
    const values = [];
    history.forEach((sample) => {
        if (!sample[kind]) {
            return;
        }
        sample[kind].forEach((value) => {
            if (Number.isFinite(value)) {
                values.push(value);
            }
        });
    });

    if (!values.length) {
        return { min: -1, max: 1, hasData: false };
    }

    let min = Math.min(...values);
    let max = Math.max(...values);
    if (Math.abs(max - min) < 1e-6) {
        const pad = Math.max(1, Math.abs(max) * 0.12 || 1);
        min -= pad;
        max += pad;
    } else {
        const pad = (max - min) * 0.12;
        min -= pad;
        max += pad;
    }

    return { min, max, hasData: true };
}

// Shared geometry for the wrench trend charts. Margins leave room on the left
// for the magnitude (y) axis labels and along the bottom for the time (x) axis
// labels; buildTrendGrid and buildTrendPath must use the same values.
const TREND_CHART_WIDTH = 960;
const TREND_CHART_HEIGHT = 540;
const TREND_CHART_MARGIN = { left: 64, right: 22, top: 20, bottom: 50 };

function buildTrendGrid(scale, units, spanSeconds) {
    const width = TREND_CHART_WIDTH;
    const height = TREND_CHART_HEIGHT;
    const { left, right, top, bottom } = TREND_CHART_MARGIN;
    const innerWidth = width - left - right;
    const innerHeight = height - top - bottom;
    const lines = [];
    const labels = [];

    // Vertical grid lines + time (x) axis ticks. The newest sample sits at the
    // right edge (0 s) and the window stretches back in time to the left.
    for (let index = 0; index <= 5; index += 1) {
        const x = left + (innerWidth * index) / 5;
        lines.push(`<line class="trend-chart__grid-line" x1="${x}" y1="${top}" x2="${x}" y2="${height - bottom}"></line>`);
        const secondsAgo = spanSeconds * (1 - index / 5);
        const tick = index === 5 ? "0" : `-${secondsAgo.toFixed(1)}`;
        labels.push(`<text class="trend-chart__axis" x="${x.toFixed(1)}" y="${height - bottom + 18}" text-anchor="middle">${tick}</text>`);
    }
    // Horizontal grid lines + magnitude (y) axis ticks (top = max, bottom = min).
    for (let index = 0; index <= 4; index += 1) {
        const y = top + (innerHeight * index) / 4;
        lines.push(`<line class="trend-chart__grid-line" x1="${left}" y1="${y}" x2="${width - right}" y2="${y}"></line>`);
        const value = scale.max - (scale.max - scale.min) * (index / 4);
        labels.push(`<text class="trend-chart__axis" x="${left - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end">${value.toFixed(1)}</text>`);
    }
    if (scale.hasData && scale.min < 0 && scale.max > 0) {
        const zeroY = top + (1 - ((0 - scale.min) / (scale.max - scale.min))) * innerHeight;
        lines.push(`<line class="trend-chart__zero" x1="${left}" y1="${zeroY}" x2="${width - right}" y2="${zeroY}"></line>`);
    }

    // Axis titles.
    labels.push(`<text class="trend-chart__axis-title" x="${(left + innerWidth / 2).toFixed(1)}" y="${height - 8}" text-anchor="middle">Time (s)</text>`);
    const titleX = 16;
    const titleY = top + innerHeight / 2;
    labels.push(`<text class="trend-chart__axis-title" x="${titleX}" y="${titleY.toFixed(1)}" text-anchor="middle" transform="rotate(-90 ${titleX} ${titleY.toFixed(1)})">Magnitude (${units})</text>`);

    return `<g>${lines.join("")}${labels.join("")}</g>`;
}

function buildTrendPath(history, kind, componentIndex, scale) {
    if (!scale.hasData || !history.length) {
        return "";
    }

    const width = TREND_CHART_WIDTH;
    const height = TREND_CHART_HEIGHT;
    const { left, right, top, bottom } = TREND_CHART_MARGIN;
    const innerWidth = width - left - right;
    const innerHeight = height - top - bottom;
    let drawing = false;
    let path = "";

    history.forEach((sample, index) => {
        const vector = sample[kind];
        const value = vector ? coerceFiniteNumber(vector[componentIndex]) : null;
        if (value === null) {
            drawing = false;
            return;
        }

        const ratio = history.length === 1 ? 1 : index / (history.length - 1);
        const x = left + ratio * innerWidth;
        const y = top + (1 - ((value - scale.min) / (scale.max - scale.min))) * innerHeight;
        path += `${drawing ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)} `;
        drawing = true;
    });

    return path.trim();
}

function renderTrendGraph(side, kind, history, currentVector, label) {
    const panel = byId(`${side}-${kind}-graph-panel`);
    if (!panel) {
        return;
    }

    const meta = TELEMETRY_SERIES[kind];
    const title = `${label || side.toUpperCase()} CARTESIAN ${kind === "force" ? "FORCE" : "MOMENT"}`;
    const hasLiveData = Array.isArray(currentVector);
    const scale = hasLiveData
        ? computeTelemetryScale(history, kind)
        : { min: -1, max: 1, hasData: false };
    const paths = meta.colors.map((color, index) => {
        const d = buildTrendPath(history, kind, index, scale);
        return d ? `<path class="trend-chart__line" style="--trend-color:${color}" d="${d}"></path>` : "";
    }).join("");
    // Time spanned by the plotted samples (oldest at the left edge, newest at
    // the right). Falls back to the full rolling-window duration before data
    // arrives so the awaiting-state axis still reads sensibly.
    const windowSeconds = ((TELEMETRY_HISTORY_LIMIT - 1) * TELEOP_POLL_INTERVAL_MS) / 1000;
    const spanSeconds = history.length > 1
        ? ((history.length - 1) * TELEOP_POLL_INTERVAL_MS) / 1000
        : windowSeconds;

    if (!scale.hasData) {
        setMarkupIfChanged(
            panel,
            `${side}:${kind}:awaiting`,
            `
                <div class="telemetry-card__header">
                    <div>
                        <span class="eyebrow">${title}</span>
                    </div>
                </div>
                <div class="trend-chart">
                    <svg class="trend-chart__svg" viewBox="0 0 ${TREND_CHART_WIDTH} ${TREND_CHART_HEIGHT}" aria-hidden="true">
                        ${buildTrendGrid(scale, meta.units, windowSeconds)}
                    </svg>
                    <div class="trend-chart__empty">${buildAwaitingDataMarkup()}</div>
                </div>
            `,
        );
        return;
    }

    delete panel.dataset.renderKey;
    panel.innerHTML = `
        <div class="telemetry-card__header">
            <div>
                <span class="eyebrow">${title}</span>
            </div>
        </div>
        <div class="trend-chart">
            <svg class="trend-chart__svg" viewBox="0 0 ${TREND_CHART_WIDTH} ${TREND_CHART_HEIGHT}" aria-hidden="true">
                ${buildTrendGrid(scale, meta.units, spanSeconds)}
                ${paths}
            </svg>
        </div>
    `;
}

async function fetchAndRenderTeleopStatus() {
    const seq = ++teleopStatusSeq;
    const status = await api("/teleop/status");
    // A newer status read (e.g. one issued right after an Engage/Disengage
    // click) has superseded this one while it was in flight -- drop its stale
    // snapshot so it can't revert the UI to the pre-click state.
    if (seq !== teleopStatusSeq) {
        return;
    }
    state.teleopStatus = status;
    if (state.summary) {
        state.summary.services = state.teleopStatus.services || state.summary.services;
        renderHomeStatus();
    }
    renderTeleop();
}

async function controlHomeService(serviceName, action, options = {}) {
    if (serviceName === "teleop" && action === "connect" && !allRobotSerialsConfigured()) {
        if (!options.silentToast) {
            showToast(ROBOT_SERIALS_REQUIRED_MESSAGE, true);
        }
        return;
    }
    const tracksBusy = action === "connect" && serviceName in state.ui.serviceConnectBusy;
    if (tracksBusy) {
        state.ui.serviceConnectBusy[serviceName] = true;
        renderHomeStatus();
    }
    let result;
    try {
        result = await api(`/system/services/${serviceName}/${action}`, { method: "POST" });
    } finally {
        if (tracksBusy) {
            state.ui.serviceConnectBusy[serviceName] = false;
            renderHomeStatus();
        }
    }
    state.summary.services = result.services;
    renderHomeStatus();
    const serviceKey = SERVICE_NAME_TO_KEY[serviceName];
    const serviceStatus = serviceKey ? result.services?.[serviceKey] : null;
    const connectSucceeded = action !== "connect" || serviceStatus?.tone === "ok";
    if (state.activeView === "teleoperation") {
        await refreshTeleopStatus();
        if (action === "connect") {
            if (connectSucceeded) {
                startTeleopPolling();
            } else {
                stopTeleopPolling();
            }
        }
    }
    if (serviceName === "cameras") {
        await loadTeleopCameraConfig();
    }
    const labels = {
        teleop: "Teleop service",
        cameras: "Cameras",
    };
    if (!options.silentToast) {
        if (action === "connect" && !connectSucceeded) {
            showToast(serviceStatus?.detail || `${labels[serviceName] || serviceName} connection failed.`, true);
        } else {
            showToast(`${labels[serviceName] || serviceName} ${action === "connect" ? "connected" : "disconnected"}.`);
        }
    }
}

function renderTeleop() {
    const teleopStatus = state.teleopStatus || {
        teleop: { started: false, initialized: false, error: null },
        robot_data: { robots: {}, errors: {} },
        cameras: { cameras: {}, errors: {} },
        services: state.summary?.services || {},
        recording: {
            active: false,
            awaiting_save: false,
            frames_captured: 0,
        },
    };

    renderRecordingOptions(teleopStatus.recording || {});

    const panels = activeArmPanels();
    const usedSlots = new Set(panels.map((panel) => panel.slot));
    ["left", "right"].forEach((slot) => {
        byId(`${slot}-wrist-column`)?.classList.toggle("hidden", !usedSlots.has(slot));
    });
    byId("wrist-feed-row")?.classList.toggle("feed-row--single", panels.length === 1);

    const cameras = teleopStatus.cameras?.cameras || {};
    renderCameraFps("ego-fps", "ego", cameras.ego);
    panels.forEach((panel) => {
        const titleEl = byId(`${panel.slot}-wrist-title`);
        if (titleEl) titleEl.textContent = panel.feedTitle;
        renderCameraFps(`${panel.slot}-wrist-fps`, panel.camera, cameras[panel.camera]);

        const robotEntry = getRobotTelemetryByIndex(panel.followerIndex, teleopStatus);
        const telemetry = readRobotTelemetry(robotEntry.robot);
        if (state.teleopStatus) appendTelemetrySample(panel.slot, telemetry);
        const history = state.telemetryHistory[panel.slot];
        renderForcePanel(panel.slot, robotEntry, telemetry, history, panel.wrenchLabel);
        renderTrendGraph(panel.slot, "force", history, telemetry.force, panel.wrenchLabel);
        renderTrendGraph(panel.slot, "moment", history, telemetry.moment, panel.wrenchLabel);
    });

    const grid = byId("teleop-status-grid");
    const services = teleopStatus.services || state.summary?.services || {};
    renderTeleopSystemCards(grid, services);

    renderRecordingStatusPanel(teleopStatus);
    renderRecordingActionButtons(teleopStatus.recording || {});
    updateTeleopControlButtons(teleopStatus);
    renderGripperControl(teleopStatus);

    const showRecordingSaveButtons = !!teleopStatus.recording.awaiting_save || !!state.ui.recordingSaveBusy || !!teleopStatus.recording.save_in_progress;
    byId("record-save").classList.toggle("hidden", !showRecordingSaveButtons);
    byId("record-discard").classList.toggle("hidden", !showRecordingSaveButtons);

    const issues = [];
    if (teleopStatus.teleop.error) {
        issues.push(teleopStatus.teleop.error);
    }
    issues.push(...Object.values(teleopStatus.robot_data.errors || {}));
    issues.push(...Object.values(teleopStatus.cameras.errors || {}));
    const issueSignature = issues.join(" | ");
    if (issueSignature && issueSignature !== state.notifications.lastTeleopIssueSignature) {
        pushNotification(issueSignature, "error");
    }
    state.notifications.lastTeleopIssueSignature = issueSignature;

    // Drive the floating fault widget off the same status snapshot.
    syncFaultCenter(teleopStatus);

    const message = byId("teleop-message");
    if (message) {
        message.classList.remove("panel--issue", "panel--ok");
        message.classList.add("hidden");
    }
}

async function refreshTeleopStatus() {
    if (teleopStatusRefreshPromise) {
        teleopStatusRefreshQueued = true;
        return teleopStatusRefreshPromise;
    }

    teleopStatusRefreshPromise = (async () => {
        try {
            do {
                teleopStatusRefreshQueued = false;
                await fetchAndRenderTeleopStatus();
            } while (teleopStatusRefreshQueued);
        } finally {
            teleopStatusRefreshPromise = null;
        }
    })();

    return teleopStatusRefreshPromise;
}

async function refreshTeleopStatusWithIndicator() {
    setTeleopRefreshBusy(true);
    try {
        await refreshTeleopStatus();
    } finally {
        setTeleopRefreshBusy(false);
    }
}

async function loadTrainingPreview(episodePath, options = {}) {
    try {
        state.preview = await api(`/datasets/preview?path=${encodeURIComponent(episodePath)}`);
        state.previewFrame = 0;
        state.previewPlaying = false;
        _stopDatasetPlayback("previewPlaying");
        try {
            state.previewSeries = await api(`/datasets/series?path=${encodeURIComponent(episodePath)}`);
        } catch (_) {
            state.previewSeries = null;
        }
    } catch (error) {
        state.preview = null;
        state.previewSeries = null;
        if (!options.silent) {
            showToast(error.message, true);
        }
    }
}

// Truncated job tags reveal their full text on hover by sliding ("rolling")
// the text to expose the clipped tail, then sliding back. Only tags whose text
// is actually clipped get the animation; the rest keep their static ellipsis.
function _setupJobTagMarquee(scope) {
    scope.querySelectorAll(".episode-row__job").forEach((tag) => {
        const text = tag.querySelector(".episode-row__job-text");
        if (!text) {
            return;
        }
        const overflow = text.scrollWidth - text.clientWidth;
        if (overflow > 1) {
            tag.classList.add("episode-row__job--overflowing");
            tag.style.setProperty("--job-tag-shift", `-${overflow}px`);
            // Pace the scroll by distance so longer names don't whip past.
            const seconds = Math.min(8, Math.max(2, overflow / 30));
            tag.style.setProperty("--job-tag-duration", `${seconds}s`);
        } else {
            tag.classList.remove("episode-row__job--overflowing");
            tag.style.removeProperty("--job-tag-shift");
            tag.style.removeProperty("--job-tag-duration");
        }
    });
}

function renderProcessing() {
    const container = byId("processing-content");
    container.classList.toggle("has-playback-bar", state.processingStep > 1);

    if (state.processingStep === 1) {
        container.innerHTML = `
            <div class="panel-header panel-header--training-step">
                <div>
                    <h2 class="training-step-title">Load Episodes</h2>
                </div>
            </div>
            <div class="episode-list" id="load-episode-list"></div>
            <div class="control-bar control-bar--episode-step">
                <button class="round-icon-button round-icon-button--add" id="training-add-episode" type="button" aria-label="Add episode dataset" title="Add episode dataset">
                    <span aria-hidden="true">+</span>
                </button>
                ${state.episodes.length ? `
                <button class="round-icon-button round-icon-button--clear" id="training-clear-episodes" type="button" aria-label="Clear all episodes" title="Clear all episodes">
                    <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                        <path d="M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6M10 11v6M14 11v6"></path>
                    </svg>
                </button>` : ""}
            </div>
            <div class="control-bar control-bar--floating-step-nav">
                <button id="training-next-step" type="button" ${state.episodes.length ? "" : "disabled"}>Next</button>
            </div>
        `;
        const list = byId("load-episode-list");
        if (!state.episodes.length) {
            list.innerHTML = `<div class="episode-empty-state"><span>No episodes selected.</span></div>`;
        } else {
            state.episodes.forEach((episode, index) => {
                const row = document.createElement("div");
                row.className = "episode-entry-row";
                const jobBadge = episode.job
                    ? `<span class="episode-entry-card__job">${escapeHtml(episode.job)}</span>`
                    : "";
                row.innerHTML = `
                    <div class="episode-entry-card">
                        <strong class="episode-entry-card__index">${index + 1}</strong>
                        <span class="episode-entry-card__divider" aria-hidden="true"></span>
                        <div class="episode-entry-card__text">
                            ${jobBadge}
                            <span class="episode-entry-card__name">${escapeHtml(episode.name)}</span>
                        </div>
                    </div>
                    <button class="round-icon-button round-icon-button--remove" data-remove-episode="${escapeHtml(episode.path)}" type="button" aria-label="Remove ${escapeHtml(episode.name)}" title="Remove ${escapeHtml(episode.name)}">
                        <span aria-hidden="true">&minus;</span>
                    </button>
        `;
                list.appendChild(row);
            });
        }
        byId("training-add-episode").onclick = () => openEpisodeBrowser();
        const clearButton = byId("training-clear-episodes");
        if (clearButton) {
            clearButton.onclick = () => {
                state.episodes = [];
                state.selectedEpisodes = [];
                renderProcessing();
            };
        }
        byId("training-next-step").onclick = () => {
            state.processingStep = 2;
            renderProcessing();
        };
        list.querySelectorAll("[data-remove-episode]").forEach((button) => {
            button.onclick = () => {
                const path = button.dataset.removeEpisode;
                state.episodes = state.episodes.filter((item) => item.path !== path);
                state.selectedEpisodes = state.selectedEpisodes.filter((item) => item !== path);
                renderProcessing();
            };
        });
        return;
    }

    if (state.processingStep === 2) {
        if (state.preview && !state.episodes.some((episode) => episode.path === state.preview.path)) {
            state.preview = null;
        }
        container.innerHTML = `
            <div class="training-layout">
                <aside class="panel">
                    <div class="panel-header">
                        <h2>Episodes</h2>
                        <button class="secondary-button toggle-all-button" id="training-select-all" type="button" title="${state.selectedEpisodes.length === state.episodes.length ? "Deselect all episodes" : "Select all episodes"}" aria-label="${state.selectedEpisodes.length === state.episodes.length ? "Deselect all episodes" : "Select all episodes"}">${state.selectedEpisodes.length === state.episodes.length ? DESELECT_ALL_ICON_SVG : SELECT_ALL_ICON_SVG}</button>
                    </div>
                    <div class="episode-list" id="training-episode-picker"></div>
                </aside>
                <div class="training-main">
                    <div id="episode-preview-block"></div>
                    <div class="control-bar control-bar--floating-step-nav">
                        <button class="secondary-button" id="training-prev-step" type="button">Previous Step</button>
                        <button id="training-merge" type="button" ${state.selectedEpisodes.length ? "" : "disabled"}>Merge Selected Episodes</button>
                    </div>
                </div>
            </div>
        `;
        const picker = byId("training-episode-picker");
        const previewPath = state.preview?.path || "";
        state.episodes.forEach((episode, index) => {
            const row = document.createElement("div");
            row.className = `episode-row episode-row--selectable ${previewPath === episode.path ? "episode-row--selected" : ""}`.trim();
            const jobBadge = episode.job
                ? `<span class="episode-row__job" title="${escapeHtml(episode.job)}"><span class="episode-row__job-text">${escapeHtml(episode.job)}</span></span>`
                : "";
            row.innerHTML = `
        <div class="episode-row__main">
          <input data-toggle-episode="${episode.path}" type="checkbox" ${state.selectedEpisodes.includes(episode.path) ? "checked" : ""} />
          <div class="episode-row__text">
            ${jobBadge}
            <span>${escapeHtml(episode.name)}</span>
          </div>
        </div>
      `;
            row.onclick = async () => {
                _showPreviewLoadingOverlay("episode-preview-block");
                await loadTrainingPreview(episode.path);
                renderProcessing();
            };
            const input = row.querySelector("[data-toggle-episode]");
            if (input instanceof HTMLInputElement) {
                input.onclick = (event) => event.stopPropagation();
                input.onchange = async () => {
                    const path = input.dataset.toggleEpisode;
                    if (!path) {
                        return;
                    }
                    if (state.selectedEpisodes.includes(path)) {
                        state.selectedEpisodes = state.selectedEpisodes.filter((item) => item !== path);
                    } else {
                        state.selectedEpisodes = [...state.selectedEpisodes, path];
                    }
                    _showPreviewLoadingOverlay("episode-preview-block");
                    await loadTrainingPreview(episode.path);
                    renderProcessing();
                };
            }
            picker.appendChild(row);
        });
        _setupJobTagMarquee(picker);
        const previewBlock = byId("episode-preview-block");
        if (!state.preview) {
            previewBlock.innerHTML = `<div class="panel panel--soft"><div class="feed__placeholder" style="min-height:200px">Select an episode to preview.</div></div>`;
        } else {
            renderDatasetPreviewBlock("episode-preview-block", state.preview, state.previewSeries?.series || null, "previewFrame", "previewPlaying");
        }
        byId("training-select-all").onclick = () => {
            state.selectedEpisodes = state.selectedEpisodes.length === state.episodes.length ? [] : state.episodes.map((episode) => episode.path);
            renderProcessing();
        };
        byId("training-prev-step").onclick = () => {
            state.processingStep = 1;
            renderProcessing();
        };
        byId("training-merge").onclick = async () => {
            if (!state.selectedEpisodes.length) return;
            // Stay on the episode-selection page and show a small progress
            // overlay; jump to the preview page only once the merge completes.
            state.mergeProgress = null;
            state.mergedPreview = null;
            state.mergedSeries = null;
            state.merging = true;
            _showMergeModal();
            try {
                await api("/datasets/merge", {
                    method: "POST",
                    body: JSON.stringify({ episode_paths: state.selectedEpisodes, output_name: `merged_${_timestampSuffix()}` }),
                });
                await _pollMergeProgress();
            } catch (error) {
                state.merging = false;
                _hideMergeModal();
                showToast(error.message, true);
            }
        };
        return;
    }

    if (state.processingStep === 3) {
        // Step 3 is only entered once the merge has finished and the preview is
        // loaded (merge progress is shown in an overlay on the previous step).
        if (!state.mergedPreview) return;

        // List "Whole Dataset" + each episode on the left; the main panel
        // previews the selected scope (whole dataset or one episode).
        const sel = state.mergedSelectedEpisode;
        const title = sel == null ? "Whole Dataset" : `Episode ${sel}`;
        container.innerHTML = `
            <div class="training-layout">
                <aside class="panel">
                    <div class="panel-header"><h2>Episodes</h2></div>
                    <div class="episode-list" id="merged-episode-picker"></div>
                </aside>
                <div class="training-main">
                    <div class="panel-header"><div><h2>${escapeHtml(title)}</h2></div></div>
                    <div id="merged-preview-block"></div>
                    <div class="control-bar control-bar--floating-step-nav"><button class="secondary-button" id="merge-prev" type="button">Previous Step</button><button id="merge-next" type="button">Next</button></div>
                </div>
            </div>
        `;

        _renderDatasetEpisodePicker(byId("merged-episode-picker"), "processing");

        renderDatasetPreviewBlock("merged-preview-block", state.mergedPreview, state.mergedSeries?.series || null, "mergedFrame", "mergedPlaying");
        byId("merge-prev").onclick = () => {
            state.processingStep = 2;
            renderProcessing();
        };
        byId("merge-next").onclick = () => {
            applyMergedDatasetToTraining();
            setActiveView("training");
            bootstrapTraining()
                .then(() => renderTraining())
                .catch((error) => showToast(error.message, true));
        };
        return;
    }
}

function renderTraining() {
    const container = byId("training-content");
    _captureTrainingLogView(container);
    // The playback bar (and its bottom-padding reservation) only appears on
    // the dataset-preview step once a dataset is loaded and previewed.
    container.classList.toggle(
        "has-playback-bar",
        state.trainingStep === 2 && !!state.mergedDatasetPreview
    );

    if (state.trainingStep === 1) {
        const loaded = !!state.mergedDatasetPath;
        const datasetName = loaded ? state.mergedDatasetPath.split("/").pop() : "";
        const preview = state.mergedDatasetPreview;
        const meta = preview
            ? `${(preview.num_episodes || 0).toLocaleString()} episodes · ${(preview.num_frames || 0).toLocaleString()} frames`
            : "Loading dataset metadata…";

        const bodyHtml = loaded
            ? `
            <div class="merged-dataset-entry" id="merged-dataset-entry">
                <div class="episode-entry-row">
                    <div class="episode-entry-card">
                        <strong class="episode-entry-card__index">1</strong>
                        <span class="episode-entry-card__divider" aria-hidden="true"></span>
                        <div>
                            <span class="episode-entry-card__name">${escapeHtml(datasetName || "Selected dataset")}</span>
                            <div class="episode-row__meta">${escapeHtml(meta)}</div>
                        </div>
                    </div>
                    <button class="round-icon-button round-icon-button--remove" id="training-remove-merged" type="button" aria-label="Remove ${escapeHtml(datasetName || "dataset")}" title="Remove ${escapeHtml(datasetName || "dataset")}">
                        <span aria-hidden="true">&minus;</span>
                    </button>
                </div>
            </div>`
            : `<div class="merged-dataset-entry" id="merged-dataset-entry"><div class="episode-empty-state"><span>No training dataset selected.</span></div></div>`;

        const controlHtml = `
            <div class="control-bar control-bar--episode-step">
                <button class="round-icon-button round-icon-button--add" id="training-browse-merged" type="button" aria-label="Browse datasets" title="Browse datasets"><span aria-hidden="true">+</span></button>
            </div>
            <div class="control-bar control-bar--floating-step-nav">
                <button id="training-next-step" type="button" ${loaded && preview ? "" : "disabled"}>Next</button>
            </div>`;

        const headerHtml = `<div class="panel-header panel-header--training-step"><div><h2 class="training-step-title">Load Training Dataset</h2></div></div>`;

        container.innerHTML = `
            ${headerHtml}
            ${bodyHtml}
            ${controlHtml}
        `;

        byId("training-browse-merged").onclick = () => openMergedDatasetBrowser();
        if (loaded) {
            byId("training-remove-merged").onclick = () => {
                state.mergedDatasetPath = "";
                state.mergedDatasetPreview = null;
                state.mergedDatasetSeries = null;
                state.mergedDatasetSelectedEpisode = null;
                state.mergedDatasetEpisodes = [];
                state.mergedDatasetScopeCache = {};
                _stopDatasetPlayback("mergedDatasetPlaying");
                renderTraining();
            };
        }
        byId("training-next-step").onclick = () => {
            state.trainingStep = 2;
            renderTraining();
        };
        return;
    }

    if (state.trainingStep === 2) {
        const tSel = state.mergedDatasetSelectedEpisode;
        const tTitle = tSel == null ? "Whole Dataset" : `Episode ${tSel}`;

        container.innerHTML = `
            <div class="training-layout">
                <aside class="panel">
                    <div class="panel-header"><h2>Episodes</h2></div>
                    <div class="episode-list" id="training-dataset-episode-picker"></div>
                </aside>
                <div class="training-main">
                    <div class="panel-header"><div><h2>${escapeHtml(tTitle)}</h2></div></div>
                    <div id="merged-dataset-preview-block"></div>
                    <div class="control-bar control-bar--floating-step-nav">
                        <button class="secondary-button" id="training-prev-dataset" type="button">Previous Step</button>
                        <button id="training-next-step" type="button">Next</button>
                    </div>
                </div>
            </div>`;

        _renderDatasetEpisodePicker(byId("training-dataset-episode-picker"), "training");
        if (state.mergedDatasetPreview) {
            renderDatasetPreviewBlock("merged-dataset-preview-block", state.mergedDatasetPreview, state.mergedDatasetSeries?.series || null, "mergedDatasetFrame", "mergedDatasetPlaying");
        } else {
            byId("merged-dataset-preview-block").innerHTML = `<div class="panel panel--soft"><div class="feed__placeholder" style="min-height:200px">Loading dataset preview…</div></div>`;
        }
        byId("training-prev-dataset").onclick = () => {
            state.trainingStep = 1;
            renderTraining();
        };
        byId("training-next-step").onclick = () => {
            state.trainingStep = 3;
            renderTraining();
        };
        return;
    }

    if (state.trainingStep === 3) {
        const catalog = state.trainingPolicies || { default: "diffusion", policies: {} };
        const policiesReady = !!state.trainingPolicies;
        const outputDir = getTrainingOutputDir();
        container.innerHTML = `
            <div class="panel-header"><div><h2>Choose Training Policy</h2></div></div>
            <div class="component-wrapper" id="policy-grid-wrap" style="min-height:100px">
                <div class="policy-grid" id="policy-grid"></div>
                ${!policiesReady ? `<div class="component-loading-overlay"><div class="mini-progress-bar"><span></span></div><span class="component-loading-overlay__label">Loading policies…</span></div>` : ""}
            </div>
            <div id="policy-config-panel"></div>
            <div class="output-picker"><div><p class="eyebrow">Training Output Directory</p><strong id="training-output-path">${escapeHtml(outputDir || "—")}</strong></div></div>
            <div class="control-bar control-bar--floating-step-nav"><button class="secondary-button" id="policy-prev" type="button">Previous Step</button><button id="policy-start" type="button" ${outputDir && policiesReady ? "" : "disabled"}>Next</button></div>
        `;
        const grid = byId("policy-grid");
        Object.entries(catalog.policies || {}).forEach(([key, policy]) => {
            const card = document.createElement("button");
            card.className = `policy-card ${state.selectedPolicy === key ? "policy-card--selected" : ""}`;
            card.type = "button";
            card.innerHTML = `<h3>${policy.label}</h3><p>${policy.description}</p>`;
            card.onclick = () => {
                state.selectedPolicy = key;
                renderTraining();
            };
            grid.appendChild(card);
        });
        renderPolicyConfigPanel(catalog.policies[state.selectedPolicy]);
        byId("policy-prev").onclick = () => {
            state.trainingStep = 2;
            renderTraining();
        };
        byId("policy-start").onclick = () => {
            state.trainingStep = 4;
            renderTraining();
        };
        return;
    }

    if (state.ui.trainingDeviceAutoTriggeredStep !== 4) {
        state.ui.trainingDeviceAutoTriggeredStep = 4;
        refreshTrainingDevices({ silent: true }).catch((error) => showToast(error.message, true));
    }

    const status = state.trainingStatus || { status: "ready", progress: 0, logs: [] };
    const progressDisplay = getTrainingProgressDisplay(status);
    const progress = progressDisplay.percent;
    const progressLabel = progressDisplay.label;
    // Pause/Resume only applies while the job is live. The backend keeps
    // status === "running" even while suspended; `paused` is a separate flag.
    const isRunning = status.status === "running";
    const isPaused = !!status.paused;
    const isStopped = status.status === "stopped";
    const isFailed = status.status === "failed";
    const isCompleted = status.status === "completed";
    const outputDir = getTrainingOutputDir();
    const canStart = !!outputDir;
    const showsRestart = isStopped || isFailed || isCompleted;
    const primaryMarkup = isRunning
        ? TELEOP_STOP_MARKUP
        : showsRestart
            ? TRAINING_RESTART_MARKUP
            : TRAINING_START_MARKUP;
    const primaryClass = isRunning ? "stop-button" : "start-button";
    const isDeviceEvalBusy = !!state.ui.trainingDeviceEvalBusy;
    const trainingDevices = state.trainingDevices || { configured: "auto", resolved: "cpu", devices: [] };
    const configuredDevice = trainingDevices.configured || "auto";
    const deviceOptions = isDeviceEvalBusy
        ? `<option value="" selected></option>`
        : (trainingDevices.devices || []).map((entry) => {
            const label = entry.name;
            const selected = entry.name === configuredDevice ? " selected" : "";
            const disabled = entry.name !== "auto" && !entry.available ? " disabled" : "";
            return `<option value="${escapeHtml(entry.name)}"${selected}${disabled}>${escapeHtml(label)}</option>`;
        }).join("");
    const deviceDetail = (() => {
        if (isDeviceEvalBusy) {
            return "Evaluating available devices...";
        }
        const selected = (trainingDevices.devices || []).find((entry) => entry.name === configuredDevice);
        if (selected?.detail) {
            return selected.detail;
        }
        if (configuredDevice === "auto") {
            return `Resolves to ${trainingDevices.resolved || "cpu"}`;
        }
        return "Select a device for training";
    })();
    const stateLabel = isPaused
        ? "Paused"
        : isRunning
            ? "Running"
            : isStopped
                ? "Stopped"
                : _formatTrainingStatusLabel(status.status);
    container.innerHTML = `
        <div class="training-layout">
            <div class="training-sidebar">
                <aside class="panel panel--soft control-panel training-device-panel">
                    <div class="panel-header"><h2>Computation Device</h2></div>
                    <div class="training-device-row">
                        <label class="training-device-field" for="training-device-select">
                            <select id="training-device-select" ${(isRunning || isDeviceEvalBusy) ? "disabled" : ""}>
                                ${deviceOptions || `<option value="auto" selected>auto</option><option value="cpu">cpu</option>`}
                            </select>
                        </label>
                        <button class="secondary-button icon-button training-device-reload" id="training-device-reload" type="button" aria-label="Evaluate training devices" title="Evaluate training devices" ${isDeviceEvalBusy ? "disabled" : ""}>
                            ${RESET_ICON_SVG}
                        </button>
                    </div>
                    <p class="training-device-detail">${escapeHtml(deviceDetail)}</p>
                </aside>
                <aside class="panel panel--soft control-panel training-controls">
                    <div class="panel-header"><h2>Training Control</h2></div>
                    <div class="control-stack">
                        <button id="training-primary-action" class="button-with-icon ${primaryClass}" type="button" ${canStart || isRunning ? "" : "disabled"}>
                            ${primaryMarkup}
                        </button>
                        <button id="training-pause-resume" class="button-with-icon" type="button" ${isRunning ? "" : "disabled"}>
                            ${isPaused ? TRAINING_RESUME_MARKUP : TRAINING_PAUSE_MARKUP}
                        </button>
                    </div>
                    <p class="training-controls__state">Status: <strong>${escapeHtml(stateLabel)}</strong></p>
                </aside>
            </div>
            <div class="training-main">
                <div class="progress-bar progress-bar--thick ${isFailed ? "progress-bar--error" : ""}"><span style="width: ${progress}%"></span><span class="progress-bar__text">${progressLabel}</span></div>
                <div class="log-pane">${renderTrainingTerminalLogs(status)}</div>
                <div class="control-bar control-bar--floating-step-nav"><button class="secondary-button" id="training-run-prev" type="button">Previous Step</button></div>
            </div>
        </div>
    `;
    _restoreTrainingLogView(container);
    const pauseResumeBtn = byId("training-pause-resume");
    if (pauseResumeBtn) {
        pauseResumeBtn.onclick = async () => {
            pauseResumeBtn.disabled = true;
            try {
                state.trainingStatus = await api(
                    isPaused ? "/training/resume" : "/training/pause",
                    { method: "POST" }
                );
            } catch (error) {
                showToast(error.message, true);
            }
            renderTraining();
        };
    }
    const deviceSelect = byId("training-device-select");
    if (deviceSelect) {
        deviceSelect.onchange = async () => {
            const nextValue = deviceSelect.value || "auto";
            try {
                state.trainingDevices = await api("/training/devices", {
                    method: "PUT",
                    body: JSON.stringify({ device: nextValue }),
                });
                if (state.trainingPolicies) {
                    state.trainingPolicies.device = state.trainingDevices.configured;
                }
            } catch (error) {
                showToast(error.message, true);
            }
            renderTraining();
        };
    }
    const deviceReloadBtn = byId("training-device-reload");
    if (deviceReloadBtn) {
        setTrainingDeviceEvalBusy(state.ui.trainingDeviceEvalBusy);
        deviceReloadBtn.onclick = () => {
            refreshTrainingDevices({ clearBeforeFetch: true, delayMs: 1000, force: true }).catch((error) => showToast(error.message, true));
        };
    }
    const primaryActionBtn = byId("training-primary-action");
    if (primaryActionBtn) {
        primaryActionBtn.onclick = async () => {
            primaryActionBtn.disabled = true;
            if (pauseResumeBtn) {
                pauseResumeBtn.disabled = true;
            }
            try {
                if (isRunning) {
                    state.trainingStatus = await api("/training/stop", { method: "POST" });
                    renderTraining();
                    return;
                }
                await startTrainingRun(outputDir);
                return;
            } catch (error) {
                showToast(error.message, true);
            }
            renderTraining();
        };
    }
    byId("training-run-prev").onclick = () => {
        state.trainingStep = 3;
        renderTraining();
    };
}

function _formatTrainingStatusLabel(status) {
    if (!status) {
        return "—";
    }
    const text = String(status).trim();
    if (!text) {
        return "—";
    }
    if (text.toLowerCase() === "waiting") {
        return "Ready";
    }
    if (text.toLowerCase() === "ready") {
        return "Ready";
    }
    return text.charAt(0).toUpperCase() + text.slice(1);
}

function _isScrolledToBottom(element, threshold = 8) {
    return element.scrollHeight - element.clientHeight - element.scrollTop <= threshold;
}

function _captureTrainingLogView(container) {
    const logPane = container ? container.querySelector(".log-pane") : null;
    if (!logPane) {
        return;
    }
    const distanceFromBottom = Math.max(
        0,
        logPane.scrollHeight - logPane.clientHeight - logPane.scrollTop,
    );
    state.trainingLogView = {
        stickToBottom: _isScrolledToBottom(logPane),
        distanceFromBottom,
        scrollLeft: logPane.scrollLeft,
    };
}

function _restoreTrainingLogView(container) {
    const logPane = container ? container.querySelector(".log-pane") : null;
    if (!logPane) {
        return;
    }

    const updateScrollState = () => {
        const distanceFromBottom = Math.max(
            0,
            logPane.scrollHeight - logPane.clientHeight - logPane.scrollTop,
        );
        state.trainingLogView = {
            stickToBottom: _isScrolledToBottom(logPane),
            distanceFromBottom,
            scrollLeft: logPane.scrollLeft,
        };
    };

    logPane.onscroll = updateScrollState;
    requestAnimationFrame(() => {
        const view = state.trainingLogView || { stickToBottom: true, distanceFromBottom: 0, scrollLeft: 0 };
        if (view.stickToBottom) {
            logPane.scrollTop = logPane.scrollHeight;
        } else {
            logPane.scrollTop = Math.max(
                0,
                logPane.scrollHeight - logPane.clientHeight - view.distanceFromBottom,
            );
        }
        logPane.scrollLeft = Math.max(0, view.scrollLeft || 0);
        updateScrollState();
    });
}

function resetTrainingRunViewState() {
    window.clearInterval(state.intervals.training);
    state.trainingStatus = null;
    state.trainingOutputStamp = "";
    state.ui.trainingDeviceAutoTriggeredStep = null;
    state.trainingLogView = {
        stickToBottom: true,
        distanceFromBottom: 0,
        scrollLeft: 0,
    };
}

async function refreshTrainingDevices(options = {}) {
    if (state.ui.trainingDeviceEvalBusy) {
        return state.trainingDevices;
    }
    const previousDevices = state.trainingDevices;
    setTrainingDeviceEvalBusy(true);
    try {
        if (options.clearBeforeFetch) {
            state.trainingDevices = { configured: "", resolved: "", devices: [] };
            if (state.activeView === "training" && state.trainingStep === 4) {
                renderTraining();
            }
        }
        await new Promise((resolve) => requestAnimationFrame(() => resolve()));
        if (options.delayMs) {
            await new Promise((resolve) => window.setTimeout(resolve, options.delayMs));
        }
        state.trainingDevices = await api(options.force ? "/training/devices?force=true" : "/training/devices");
        if (state.trainingPolicies) {
            state.trainingPolicies.device = state.trainingDevices.configured;
        }
        return state.trainingDevices;
    } catch (error) {
        state.trainingDevices = previousDevices;
        if (!options.silent) {
            throw error;
        }
        return state.trainingDevices;
    } finally {
        setTrainingDeviceEvalBusy(false);
        if (state.activeView === "training" && state.trainingStep === 4) {
            renderTraining();
        }
    }
}

function applyMergedDatasetToTraining() {
    resetTrainingRunViewState();
    state.mergedDatasetPath = state.mergedPath;
    state.mergedDatasetPreview = state.mergedPreview;
    state.mergedDatasetSeries = state.mergedSeries;
    state.mergedDatasetSelectedEpisode = state.mergedSelectedEpisode;
    state.mergedDatasetEpisodes = [...(state.mergedEpisodes || [])];
    state.mergedDatasetScopeCache = { ...(state.mergedScopeCache || {}) };
    state.mergedDatasetFrame = state.mergedFrame || 0;
    state.mergedDatasetPlaying = false;
    _stopDatasetPlayback("mergedDatasetPlaying");
    state.trainingStep = 1;
}

function getTrainingOutputDir() {
    const trainingRoot = (state.summary && state.summary.storage && state.summary.storage.training) || "";
    const datasetName = state.mergedDatasetPath ? state.mergedDatasetPath.split("/").pop() : "";
    if (!state.trainingOutputStamp) {
        state.trainingOutputStamp = _timestampSuffix();
    }
    return trainingRoot && datasetName
        ? `${trainingRoot}/${datasetName}-${state.selectedPolicy}_${state.trainingOutputStamp}`
        : trainingRoot;
}

// Value stored for a field: the user's edit if present, else the schema default.
function policyFieldValue(policy, field) {
    const edits = state.policyConfig[policy] || {};
    return field.name in edits ? edits[field.name] : field.default;
}

// steps = epochs * ceil(frames / batch_size); shown live under the epochs box.
function computeTrainingSteps(policy) {
    const frames = (state.mergedDatasetPreview && state.mergedDatasetPreview.num_frames) || 0;
    const edits = state.policyConfig[policy] || {};
    const batch = Number(edits.batch_size ?? 64);
    const epochs = Number(edits.epochs ?? 100);
    if (!frames || !batch || !epochs) return 0;
    return epochs * Math.ceil(frames / batch);
}

function renderPolicyConfigPanel(policy) {
    const panel = byId("policy-config-panel");
    if (!panel || !policy || !Array.isArray(policy.fields)) return;
    const key = state.selectedPolicy;
    const rows = policy.fields.map((field) => {
        const value = policyFieldValue(key, field);
        const hint = field.min != null && field.max != null
            ? `Range: ${field.min}–${field.max}` : "";
        let control;
        if (field.type === "enum") {
            const opts = (field.choices || []).map((c) =>
                `<option value="${c}" ${c === value ? "selected" : ""}>${c}</option>`).join("");
            control = `<select class="text-input" data-field="${field.name}">${opts}</select>`;
        } else if (field.type === "bool") {
            control = `<select class="text-input" data-field="${field.name}">
                <option value="true" ${value ? "selected" : ""}>true</option>
                <option value="false" ${!value ? "selected" : ""}>false</option></select>`;
        } else if (field.type === "tuple") {
            control = `<div class="config-tuple-row">` + Array.from({ length: field.arity }, (_, i) =>
                `<input class="text-input" type="number" data-field="${field.name}" data-index="${i}" value="${value[i]}" />`).join("") + `</div>`;
        } else {
            const step = field.type === "float" ? "any" : "1";
            control = `<input class="text-input" type="number" step="${step}" data-field="${field.name}" value="${value}" />`;
        }
        const readout = field.name === "epochs"
            ? `<small class="field-hint" id="steps-readout"></small>` : "";
        return `<div class="config-field">
            <label class="field-label">${field.name}</label>
            ${control}
            ${hint ? `<small class="field-hint">${hint}</small>` : ""}
            ${readout}
        </div>`;
    }).join("");
    panel.className = "policy-config-panel";
    panel.innerHTML = `<p class="eyebrow">Training Configuration</p>
        <div class="config-grid">${rows}</div>`;

    const edits = (state.policyConfig[key] = state.policyConfig[key] || {});
    const updateStepsReadout = () => {
        const el = byId("steps-readout");
        if (!el) return;
        const steps = computeTrainingSteps(key);
        const frames = (state.mergedDatasetPreview && state.mergedDatasetPreview.num_frames) || 0;
        el.textContent = steps
            ? `= ${steps.toLocaleString()} steps (${edits.epochs ?? 100} epochs × ⌈${frames.toLocaleString()} frames / batch⌉)`
            : "load a dataset to compute steps";
    };
    panel.querySelectorAll("[data-field]").forEach((el) => {
        el.oninput = el.onchange = () => {
            const field = el.dataset.field;
            const idxAttr = el.dataset.index;
            if (idxAttr != null) {
                const cur = Array.isArray(edits[field]) ? edits[field].slice()
                    : policy.fields.find((f) => f.name === field).default.slice();
                cur[Number(idxAttr)] = Number(el.value);
                edits[field] = cur;
            } else if (el.tagName === "SELECT" && (el.value === "true" || el.value === "false")) {
                edits[field] = el.value === "true";
            } else if (el.type === "number") {
                edits[field] = Number(el.value);
            } else {
                edits[field] = el.value;
            }
            if (field === "epochs" || field === "batch_size") updateStepsReadout();
        };
    });
    updateStepsReadout();
}

// Diff each field against its schema default; emit [flag, value] for changed ones.
// epochs -> --steps (converted); tuple -> "[a, b]"; bool/enum/number -> string.
function buildTrainingExtraArgs(policy) {
    const entry = state.trainingPolicies && state.trainingPolicies.policies[policy];
    if (!entry || !Array.isArray(entry.fields)) return [];
    const edits = state.policyConfig[policy] || {};
    const args = [];
    entry.fields.forEach((field) => {
        // epochs always maps to a concrete --steps (default 100 epochs is intentional,
        // not LeRobot's 100k step default), so always emit it when computable.
        if (field.name === "epochs") {
            const steps = computeTrainingSteps(policy);
            if (steps) args.push(field.flag, String(steps));
            return;
        }
        if (!(field.name in edits)) return;
        const value = edits[field.name];
        if (JSON.stringify(value) === JSON.stringify(field.default)) return;
        if (field.type === "tuple") {
            args.push(field.flag, `[${value.join(", ")}]`);
        } else {
            args.push(field.flag, String(value));
        }
    });
    return args;
}

async function startTrainingRun(outputDir) {
    state.trainingOutputStamp = "";
    try {
        state.trainingStep = 4;
        renderTraining();
        state.trainingStatus = await api("/training/start", {
            method: "POST",
            body: JSON.stringify({
                dataset_path: state.mergedDatasetPath,
                output_dir: outputDir,
                policy_type: state.selectedPolicy,
                extra_args: buildTrainingExtraArgs(state.selectedPolicy),
            }),
        });
        renderTraining();
        window.clearInterval(state.intervals.training);
        state.intervals.training = window.setInterval(async () => {
            if (state.activeView !== "training" || state.trainingStep !== 4) {
                return;
            }
            state.trainingStatus = await api("/training/status");
            renderTraining();
        }, 2000);
    } catch (error) {
        showToast(error.message, true);
        state.trainingStep = 3;
        renderTraining();
        throw error;
    }
}

function _latestTrainingTrackerLine(status) {
    const logs = Array.isArray(status.logs) ? status.logs : [];
    for (let i = logs.length - 1; i >= 0; i -= 1) {
        const parsed = _parseTrainingLogEntry(logs[i], status.job_id);
        const line = String(parsed.message || "").replace(/\u001b\[[0-9;]*m/g, "");
        const parts = line.split("\r").map((part) => part.trim()).filter(Boolean);
        for (let j = parts.length - 1; j >= 0; j -= 1) {
            const part = parts[j];
            const idx = part.indexOf("Training:");
            if (idx !== -1) {
                return part.slice(idx);
            }
        }
    }
    return "";
}

function _parseTrainingTrackerLine(line) {
    if (!line) {
        return { step: null, total: null, percent: null, speed: null };
    }

    const percentMatch = line.match(/Training:\s*([0-9]+(?:\.[0-9]+)?)%/i);
    const stepTotalMatch = line.match(/(?:\||\s)(\d+)\s*\/\s*(\d+)(?:\s|\[|$)/);
    const speedMatch = line.match(/([0-9]+(?:\.[0-9]+)?)\s*(?:it|step)\/s/i);

    const percent = percentMatch ? Number(percentMatch[1]) : null;
    const step = stepTotalMatch ? Number(stepTotalMatch[1]) : null;
    const total = stepTotalMatch ? Number(stepTotalMatch[2]) : null;
    const speed = speedMatch ? Number(speedMatch[1]) : null;

    return {
        percent: Number.isFinite(percent) ? percent : null,
        step: Number.isFinite(step) ? step : null,
        total: Number.isFinite(total) ? total : null,
        speed: Number.isFinite(speed) ? speed : null,
    };
}

function getTrainingProgressDisplay(status) {
    if (status.status === "completed") {
        return { percent: 100, label: "100% · completed" };
    }
    if (status.status === "failed") {
        return {
            percent: Number.isFinite(status.progress) ? Math.max(0, Math.min(100, Math.round(status.progress))) : 0,
            label: escapeHtml(formatValue(status.error || "Training failed")),
        };
    }

    const latestTrackerLine = _latestTrainingTrackerLine(status);
    const tracker = _parseTrainingTrackerLine(latestTrackerLine);

    const step = Number(status.metrics?.step);
    const total = Number(status.total_steps);
    const safeStep = Number.isFinite(step) ? Math.max(0, Math.trunc(step)) : null;
    const safeTotal = Number.isFinite(total) && total > 0 ? Math.trunc(total) : null;

    const resolvedStep = safeStep != null ? safeStep : (tracker.step != null ? Math.max(0, Math.trunc(tracker.step)) : null);
    const resolvedTotal = safeTotal != null ? safeTotal : (tracker.total != null && tracker.total > 0 ? Math.trunc(tracker.total) : null);

    let percent = resolvedStep != null && resolvedTotal != null
        ? Math.max(0, Math.min(100, Math.round((resolvedStep / resolvedTotal) * 100)))
        : null;
    if (percent == null && tracker.percent != null) {
        percent = Math.max(0, Math.min(100, Math.round(tracker.percent)));
    }
    if (percent == null) {
        percent = Number.isFinite(status.progress) ? Math.max(0, Math.min(100, Math.round(status.progress))) : 0;
    }

    let speed = tracker.speed;
    if (speed == null) {
        const elapsed = String(status.elapsed || "");
        const m = elapsed.match(/(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?/);
        if (m && resolvedStep != null) {
            const hours = Number(m[1] || 0);
            const minutes = Number(m[2] || 0);
            const seconds = Number(m[3] || 0);
            const totalSeconds = (hours * 3600) + (minutes * 60) + seconds;
            if (totalSeconds > 0) {
                speed = resolvedStep / totalSeconds;
            }
        }
    }

    const stepText = resolvedStep != null
        ? `${resolvedStep}/${resolvedTotal != null ? resolvedTotal : "?"}`
        : "0/?";
    const speedText = speed != null && Number.isFinite(speed)
        ? `${speed.toFixed(2)} steps/s`
        : "-- steps/s";
    const normalizedStatus = String(status.status || "").trim().toLowerCase();

    if (normalizedStatus === "waiting" || normalizedStatus === "ready") {
        return { percent, label: "Ready" };
    }

    if (status.status === "running") {
        return { percent, label: escapeHtml(`${percent}% · ${stepText} · ${speedText}`) };
    }
    return {
        percent,
        label: escapeHtml(`${formatValue(status.status)} · ${stepText} · ${speedText}`),
    };
}

function _terminalClock() {
    const d = new Date();
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    return `${hh}:${mm}:${ss}`;
}

function _logLevelClass(level) {
    const normalized = String(level || "INFO").toLowerCase();
    return `terminal-log__level terminal-log__level--${normalized}`;
}

function _inferTrainingLogLevel(message) {
    const lowered = String(message || "").toLowerCase();
    if (["traceback", "exception", " fatal", "failed", "error"].some((token) => lowered.includes(token))) {
        return "ERROR";
    }
    if (["warning", "warn", "deprecated"].some((token) => lowered.includes(token))) {
        return "WARN";
    }
    return "INFO";
}

function _parseTrainingLogEntry(rawLine, jobId = "") {
    const line = String(rawLine || "");
    if (line.startsWith("@@TRAIN_LOG@@")) {
        try {
            const entry = JSON.parse(line.slice("@@TRAIN_LOG@@".length));
            return {
                level: entry.level || "INFO",
                source: entry.source || "TRAIN",
                message: entry.message || "",
                detail: entry.detail || "",
            };
        } catch (_) {
            // Fall back to raw rendering below if a stored entry is malformed.
        }
    }
    return {
        level: _inferTrainingLogLevel(line),
        source: "TRAIN",
        message: line,
        detail: jobId ? `job_id=${jobId}` : "",
    };
}

function _terminalLogRow(level, source, message, detail = "") {
    return `
        <div class="terminal-log__row">
            <span class="terminal-log__stamp">[${_terminalClock()}]</span>
            <span class="${_logLevelClass(level)}">${escapeHtml(level)}</span>
            <span class="terminal-log__source">${escapeHtml(source)}</span>
            <span class="terminal-log__message">${escapeHtml(message)}</span>
            ${detail ? `<span class="terminal-log__detail">${escapeHtml(detail)}</span>` : ""}
        </div>
    `;
}

function renderTrainingTerminalLogs(status) {
    const logs = status.logs || [];
    if (!logs.length) {
        return `<div class="terminal-log__empty">Training logs will appear here.</div>`;
    }

    let html = "";
    if (status.status === "running") {
        const metrics = status.metrics || {};
        const parts = [];
        if (status.job_id) parts.push(`job_id=${status.job_id}`);
        if (status.elapsed) parts.push(`elapsed=${status.elapsed}`);
        if (metrics.step !== undefined) parts.push(`step=${metrics.step}/${status.total_steps ?? "?"}`);
        if (metrics.loss !== undefined) parts.push(`loss=${Number(metrics.loss).toFixed(3)}`);
        if (metrics.grad_norm !== undefined) parts.push(`grdn=${Number(metrics.grad_norm).toFixed(3)}`);
        if (metrics.lr !== undefined) parts.push(`lr=${Number(metrics.lr).toExponential(2)}`);
        if (status.log_lines !== undefined) parts.push(`lines=${status.log_lines}`);
        html += _terminalLogRow("INFO", "·", "Training job running", parts.join(" "));
    }

    for (const line of logs) {
        const entry = _parseTrainingLogEntry(line, status.job_id);
        html += _terminalLogRow(entry.level, entry.source, entry.message, entry.detail);
    }
    return html;
}

function isEpisodeBrowserMode() {
    return state.pathBrowser.mode === "episodes";
}

function getBrowserSelectablePaths() {
    return (state.pathBrowser.items || [])
        .filter((item) => {
            if (isEpisodeBrowserMode()) {
                return !!item.is_valid_episode;
            }
            return state.pathBrowser.directoriesOnly ? !!item.is_dir : true;
        })
        .map((item) => item.path);
}

// Selectable episode paths belonging to a single job group, in list order.
function getBrowserSelectablePathsForJob(job) {
    return (state.pathBrowser.items || [])
        .filter((item) => !!item.is_valid_episode && (item.job || null) === job)
        .map((item) => item.path);
}

function updateBrowserConfirmState() {
    const confirmButton = byId("browser-confirm");
    if (!confirmButton) {
        return;
    }
    const selectionRequired = !!state.pathBrowser.requireSelection || !!state.pathBrowser.multiSelect;
    confirmButton.disabled = selectionRequired && !state.pathBrowser.selected.length;
}

function updateBrowserSelectionUi() {
    const list = byId("browser-list");
    if (!list) {
        return;
    }

    const selected = new Set(state.pathBrowser.selected);
    list.querySelectorAll("[data-browser-path]").forEach((element) => {
        const path = element.dataset.browserPath;
        const isSelected = !!path && selected.has(path);
        element.classList.toggle("browser-item--selected", isSelected);
        const checkbox = element.querySelector("input[type='checkbox']");
        if (checkbox instanceof HTMLInputElement) {
            checkbox.checked = isSelected;
        }
    });

    const selectAllButton = byId("browser-select-all");
    const selectablePaths = getBrowserSelectablePaths();
    if (selectAllButton) {
        selectAllButton.disabled = selectablePaths.length === 0;
        const allSelected = selectablePaths.length > 0
            && selectablePaths.every((path) => state.pathBrowser.selected.includes(path));
        setToggleAllButton(selectAllButton, allSelected);
    }

    list.querySelectorAll("[data-job-toggle]").forEach((button) => {
        const job = button.dataset.jobToggle || null;
        const jobPaths = getBrowserSelectablePathsForJob(job);
        const allSelected = jobPaths.length > 0
            && jobPaths.every((path) => selected.has(path));
        setToggleAllButton(button, allSelected, "episodes in this job");
    });

    updateBrowserConfirmState();
}

function setBrowserSelection(paths) {
    const allowed = new Set(getBrowserSelectablePaths());
    state.pathBrowser.selected = paths.filter((path) => allowed.has(path));
    updateBrowserSelectionUi();
}

function toggleBrowserSelection(path) {
    if (!path) {
        return;
    }
    const nextSelected = new Set(state.pathBrowser.selected);
    if (nextSelected.has(path)) {
        nextSelected.delete(path);
    } else {
        if (!state.pathBrowser.multiSelect) {
            nextSelected.clear();
        }
        nextSelected.add(path);
    }
    setBrowserSelection([...nextSelected]);
}

// Select every episode under a job, or deselect them all if they are already
// fully selected. Mirrors the global select-all toggle, scoped to one job.
function toggleBrowserJobSelection(job) {
    const jobPaths = getBrowserSelectablePathsForJob(job);
    if (!jobPaths.length) {
        return;
    }
    const nextSelected = new Set(state.pathBrowser.selected);
    const allSelected = jobPaths.every((path) => nextSelected.has(path));
    jobPaths.forEach((path) => {
        if (allSelected) {
            nextSelected.delete(path);
        } else {
            nextSelected.add(path);
        }
    });
    setBrowserSelection([...nextSelected]);
}

function syncBrowserDialogChrome() {
    const header = byId("browser-header");
    const eyebrow = byId("browser-eyebrow");
    const closeButton = byId("browser-close");
    const upButton = byId("browser-up");
    const selectAllButton = byId("browser-select-all");
    const confirmButton = byId("browser-confirm");
    const pathNote = byId("browser-path-note");

    if (header) {
        header.classList.toggle("hidden", !!state.pathBrowser.hideHeader);
    }
    if (eyebrow) {
        eyebrow.textContent = state.pathBrowser.eyebrow || "";
        eyebrow.classList.toggle("hidden", !!state.pathBrowser.hideEyebrow || !state.pathBrowser.eyebrow);
        resetTrainingRunViewState();
    }
    if (closeButton) {
        closeButton.classList.toggle("hidden", !!state.pathBrowser.hideClose);
    }
    if (upButton) {
        upButton.classList.toggle("hidden", !!state.pathBrowser.hideUp);
    }
    if (selectAllButton) {
        selectAllButton.classList.toggle("hidden", !state.pathBrowser.showSelectAll);
    }
    const sortButton = byId("browser-sort");
    if (sortButton) {
        // The time-created sort only makes sense for the episode picker.
        sortButton.classList.toggle("hidden", !isEpisodeBrowserMode());
        const ascending = state.pathBrowser.sortOrder === "asc";
        const label = byId("browser-sort-label");
        if (label) {
            label.textContent = ascending ? "Oldest" : "Newest";
        }
        sortButton.classList.toggle("browser-sort-button--asc", ascending);
        const title = ascending
            ? "Sorted oldest first — click for newest first"
            : "Sorted newest first — click for oldest first";
        sortButton.setAttribute("title", title);
        sortButton.setAttribute("aria-label", title);
    }
    if (confirmButton) {
        confirmButton.textContent = state.pathBrowser.confirmLabel || "Select";
    }
    if (pathNote) {
        pathNote.textContent = state.pathBrowser.pathNote || "";
        pathNote.classList.toggle("hidden", !state.pathBrowser.pathNote);
    }
    byId("browser-title").textContent = state.pathBrowser.title || "Select Path";
}

async function refreshBrowser(path) {
    const params = new URLSearchParams();
    params.set("path", path);
    params.set("directories_only", String(state.pathBrowser.directoriesOnly));
    if (state.pathBrowser.rootPath) {
        params.set("root_path", state.pathBrowser.rootPath);
    }
    if (state.pathBrowser.annotateEpisodeDirs) {
        params.set("annotate_episode_dirs", "true");
    }
    const result = await api(`/datasets/browse?${params.toString()}`);
    state.pathBrowser.currentPath = result.path;
    state.pathBrowser.rootPath = result.root_path || state.pathBrowser.rootPath || result.path;
    state.pathBrowser.items = sortBrowserItems(result.items || []);
    state.pathBrowser.selected = state.pathBrowser.selected.filter((selectedPath) =>
        state.pathBrowser.items.some((item) => item.path === selectedPath),
    );
    syncBrowserDialogChrome();
    byId("browser-path").textContent = result.path;
    renderBrowserList();
    byId("browser-modal").classList.remove("hidden");
}

// Render the current ``state.pathBrowser.items`` into the browser list. Split
// out of refreshBrowser so the sort-order toggle can re-render the already
// fetched items without another round-trip.
function renderBrowserList() {
    const list = byId("browser-list");
    list.innerHTML = "";

    if (!state.pathBrowser.items.length) {
        list.innerHTML = `<div class="dialog-empty-state">${escapeHtml(state.pathBrowser.emptyMessage || "No entries available.")}</div>`;
        updateBrowserSelectionUi();
        return;
    }

    // Job heading currently rendered, so consecutive episodes sharing a job get
    // one header. Episodes are listed grouped by job (see _expand_job_episodes).
    let currentJobHeading = null;
    // Per-job running sequence number, so episodes are numbered 1, 2, 3, … within
    // each job independently. Ungrouped episodes (job == null) share their own run.
    const jobSequence = new Map();
    state.pathBrowser.items.forEach((item) => {
        if (isEpisodeBrowserMode()) {
            const selectable = !!item.is_valid_episode;
            // Insert a job group header whenever the job changes.
            const job = item.job || null;
            const sequenceKey = job || "";
            const sequenceNumber = (jobSequence.get(sequenceKey) || 0) + 1;
            jobSequence.set(sequenceKey, sequenceNumber);
            if (selectable && job && job !== currentJobHeading) {
                currentJobHeading = job;
                const heading = document.createElement("div");
                heading.className = "browser-item__job-heading";
                const jobLabel = document.createElement("span");
                jobLabel.className = "browser-item__job-heading-text";
                jobLabel.textContent = job;
                heading.appendChild(jobLabel);
                if (state.pathBrowser.multiSelect) {
                    const jobToggle = document.createElement("button");
                    jobToggle.type = "button";
                    jobToggle.className = "secondary-button toggle-all-button browser-item__job-toggle";
                    jobToggle.dataset.jobToggle = job;
                    setToggleAllButton(jobToggle, false, "episodes in this job");
                    jobToggle.onclick = (event) => {
                        event.preventDefault();
                        toggleBrowserJobSelection(job);
                    };
                    heading.appendChild(jobToggle);
                }
                list.appendChild(heading);
            } else if (!job) {
                currentJobHeading = null;
            }
            const row = document.createElement("label");
            row.className = `browser-item browser-item--episode ${selectable ? "" : "browser-item--invalid"} ${job ? "browser-item--in-job" : ""}`.trim();
            row.dataset.browserPath = item.path;
            row.innerHTML = `
                <span class="browser-item__main">
                    <span class="browser-item__seq">${sequenceNumber}</span>
                    <input class="browser-item__checkbox" type="checkbox" ${selectable ? "" : "disabled"} />
                    <strong class="browser-item__name">${escapeHtml(item.name)}</strong>
                </span>
                <span class="browser-item__meta ${selectable ? "hidden" : ""}">Invalid episode</span>
            `;
            const checkbox = row.querySelector("input");
            if (checkbox instanceof HTMLInputElement) {
                checkbox.checked = state.pathBrowser.selected.includes(item.path);
                checkbox.onchange = () => toggleBrowserSelection(item.path);
            }
            list.appendChild(row);
            return;
        }

        const button = document.createElement("button");
        button.className = "browser-item";
        button.type = "button";
        button.dataset.browserPath = item.path;
        button.innerHTML = `<strong>${escapeHtml(item.name)}</strong><span>${item.is_dir ? "Directory" : "File"}</span>`;
        button.onclick = () => {
            if (!item.is_dir && state.pathBrowser.directoriesOnly) {
                return;
            }
            if (state.pathBrowser.multiSelect) {
                toggleBrowserSelection(item.path);
            } else {
                setBrowserSelection([item.path]);
            }
        };
        if (state.pathBrowser.allowNavigation) {
            button.ondblclick = () => {
                if (item.is_dir) {
                    refreshBrowser(item.path).catch((error) => showToast(error.message, true));
                }
            };
        }
        list.appendChild(button);
    });

    updateBrowserSelectionUi();
}

// Sort browser items by creation time in the current sort order. Episodes are
// kept grouped by job: episodes within a job are ordered by time, and the job
// groups themselves are ordered by their most recent (desc) / oldest (asc)
// episode. Ungrouped entries (job == null) sort purely by time. Non-episode
// browsing keeps the server's name order untouched.
function sortBrowserItems(items) {
    if (!isEpisodeBrowserMode()) {
        return items;
    }
    const order = state.pathBrowser.sortOrder === "asc" ? "asc" : "desc";
    const dir = order === "asc" ? 1 : -1;
    const created = (item) => (typeof item.created === "number" ? item.created : 0);

    // Bucket by job, preserving a representative time per job for group ordering.
    const groups = new Map();
    items.forEach((item) => {
        const key = item.job || "";
        if (!groups.has(key)) {
            groups.set(key, []);
        }
        groups.get(key).push(item);
    });

    const groupTime = (entries) => {
        const times = entries.map(created);
        return order === "asc" ? Math.min(...times) : Math.max(...times);
    };

    const orderedGroups = [...groups.entries()].sort(
        ([, a], [, b]) => (groupTime(a) - groupTime(b)) * dir,
    );

    const sorted = [];
    orderedGroups.forEach(([, entries]) => {
        entries.sort((a, b) => (created(a) - created(b)) * dir);
        sorted.push(...entries);
    });
    return sorted;
}

async function openBrowser(options) {
    state.pathBrowser = {
        ...state.pathBrowser,
        mode: "generic",
        title: "Select Path",
        currentPath: "/",
        rootPath: "",
        directoriesOnly: true,
        multiSelect: false,
        selected: [],
        items: [],
        allowNavigation: true,
        annotateEpisodeDirs: false,
        showSelectAll: false,
        hideHeader: false,
        hideEyebrow: false,
        hideClose: false,
        hideUp: false,
        requireSelection: false,
        fallbackToCurrentPath: true,
        emptyMessage: "No entries available.",
        confirmLabel: "Select",
        pathNote: "Current path",
        eyebrow: "Server Path Browser",
        onConfirm: null,
        // Sort order for episode browsing, by creation time. "desc" lists the
        // newest episodes first; the episode browser exposes a toggle for it.
        sortOrder: "desc",
        ...options,
        selected: [],
        items: [],
    };
    await refreshBrowser(options.startPath);
}

function closeBrowser() {
    byId("browser-modal").classList.add("hidden");
}

function openEpisodeBrowser() {
    openBrowser({
        mode: "episodes",
        title: "",
        startPath: state.summary.storage.episodes,
        rootPath: state.summary.storage.episodes,
        directoriesOnly: true,
        multiSelect: true,
        allowNavigation: false,
        annotateEpisodeDirs: true,
        showSelectAll: true,
        hideHeader: true,
        hideEyebrow: true,
        hideClose: true,
        hideUp: true,
        requireSelection: true,
        fallbackToCurrentPath: false,
        emptyMessage: "No episode datasets found under this directory.",
        confirmLabel: "Load",
        pathNote: "Episodes directory",
        onConfirm: (paths) => {
            const orderedItems = (state.pathBrowser.items || [])
                .filter((item) => paths.includes(item.path));
            orderedItems.forEach((item) => {
                if (!state.episodes.some((existing) => existing.path === item.path)) {
                    state.episodes.push({
                        name: item.path.split("/").pop(),
                        path: item.path,
                        job: item.job || null,
                    });
                }
            });
            closeBrowser();
            renderProcessing();
        },
    }).catch((error) => showToast(error.message, true));
}

function openMergedDatasetBrowser() {
    openBrowser({
        mode: "episodes",
        title: "",
        startPath: state.summary.storage.merged,
        rootPath: state.summary.storage.merged,
        directoriesOnly: true,
        multiSelect: false,
        allowNavigation: false,
        annotateEpisodeDirs: true,
        showSelectAll: false,
        hideHeader: true,
        hideEyebrow: true,
        hideClose: true,
        hideUp: true,
        requireSelection: true,
        fallbackToCurrentPath: false,
        emptyMessage: "No datasets found.",
        confirmLabel: "Load",
        pathNote: "Datasets directory",
        onConfirm: async (paths) => {
            const path = paths[0];
            if (!path) return;
            if (path !== state.mergedDatasetPath) {
                resetTrainingRunViewState();
            }
            state.mergedDatasetPath = path;
            state.mergedDatasetPreview = null;
            state.mergedDatasetSeries = null;
            state.mergedDatasetSelectedEpisode = null;
            state.mergedDatasetEpisodes = [];
            state.mergedDatasetScopeCache = {};
            state.mergedDatasetFrame = 0;
            state.mergedDatasetPlaying = false;
            _stopDatasetPlayback("mergedDatasetPlaying");
            closeBrowser();
            renderTraining();
            _showPreviewLoadingOverlay("merged-dataset-preview-block");
            try {
                const preview = await api(`/datasets/preview?path=${encodeURIComponent(path)}`);
                let series = null;
                try {
                    series = await api(`/datasets/series?path=${encodeURIComponent(path)}`);
                } catch (_) {
                    series = null;
                }
                state.mergedDatasetEpisodes = preview.episodes || [];
                state.mergedDatasetScopeCache = { all: { preview, series } };
                state.mergedDatasetPreview = preview;
                state.mergedDatasetSeries = series;
            } catch (error) {
                state.mergedDatasetPreview = null;
                state.mergedDatasetSeries = null;
                showToast(error.message, true);
            }
            renderTraining();
        },
    }).catch((error) => showToast(error.message, true));
}

async function bootstrapTeleoperation() {
    loadTeleopCameraConfig();
    // Do not auto-initialize the system modules when entering the data
    // collection page. Modules are connected manually from the home page,
    // or via the per-service reload buttons on this page. Here we only
    // reflect whatever is already running.
    try {
        await refreshTeleopStatus();
    } catch (error) {
        // Non-fatal: the reload buttons and polling will retry.
        showToast(error.message, true);
    }
    state.teleopBootstrapped = true;
    // Reload so the panel reflects the current camera assignment.
    loadTeleopCameraConfig();
    if (hasActiveTeleopServices(state.teleopStatus)) {
        startTeleopPolling();
    }
}

async function bootstrapProcessing() {
    if (state.processingBootstrapped) {
        return;
    }
    state.processingBootstrapped = true;
    renderProcessing();
}

async function bootstrapTraining() {
    if (state.trainingBootstrapped) {
        return;
    }
    await api("/training/bootstrap", { method: "POST" });
    state.trainingPolicies = await api("/training/policies");
    state.selectedPolicy = state.trainingPolicies.default;
    state.trainingBootstrapped = true;
    if (state.activeView === "training") {
        renderTraining();
    }
    state.ui.trainingDeviceAutoTriggeredStep = 4;
    refreshTrainingDevices({ silent: true }).catch((error) => showToast(error.message, true));
}

function bindGlobalEvents() {
    document.querySelectorAll("[data-nav]").forEach((button) => {
        button.addEventListener("click", () => {
            const target = button.dataset.nav;
            if (!target) {
                return;
            }
            setActiveView(target);
            if (target === "teleoperation") {
                bootstrapTeleoperation().catch((error) => showToast(error.message, true));
            }
            if (target === "processing") {
                bootstrapProcessing().catch((error) => showToast(error.message, true));
            }
            if (target === "training") {
                bootstrapTraining().catch((error) => showToast(error.message, true));
            }
        });
    });

    const notificationCenter = byId("notification-center");
    const notificationToggle = byId("notification-toggle");
    if (notificationToggle) {
        notificationToggle.onclick = (event) => {
            event.stopPropagation();
            toggleNotificationCenter();
        };
    }
    if (notificationCenter) {
        notificationCenter.onclick = (event) => event.stopPropagation();
    }

    const faultCenter = byId("fault-center");
    const faultToggle = byId("fault-toggle");
    if (faultToggle) {
        faultToggle.onclick = (event) => {
            event.stopPropagation();
            toggleFaultCenter();
        };
    }
    if (faultCenter) {
        faultCenter.onclick = (event) => event.stopPropagation();
    }

    document.addEventListener("click", () => {
        if (state.notifications.open) {
            toggleNotificationCenter(false);
        }
        // Don't collapse mid-clear or while showing the cleared confirmation;
        // the user dismisses those via the spinner finishing / the OK button.
        if (state.fault.open && !state.fault.busy && !state.fault.cleared) {
            toggleFaultCenter(false);
        }
    });

    const refreshButton = byId("teleop-refresh");
    if (refreshButton) {
        refreshButton.onclick = () => refreshTeleopStatusWithIndicator().catch((error) => showToast(error.message, true));
    }
    byId("teleop-power").onclick = async () => {
        if (state.teleopStatus?.teleop?.started) {
            try {
                await api("/teleop/stop", { method: "POST" });
                await refreshTeleopStatus();
            } catch (error) {
                showToast(error.message, true);
            }
            return;
        }
        // The checkbox maps to Init()'s zero_ft_sensor flag. When enabled the
        // blocking Init() zeroes the F/T sensors, so confirm first since the
        // robots must be free of unexpected contact for the zero to be valid.
        const zeroFtSensor = !!byId("teleop-zero-sensors")?.checked;
        if (zeroFtSensor) {
            const proceed = window.confirm(
                "Force sensors will be zeroed, please make sure the robots are NOT in contact with anything other than the configured tools. Click OK to proceed.",
            );
            if (!proceed) {
                return;
            }
        }
        // Show the loading state and orange warning until the request returns.
        // The warning only makes sense while zeroing is actually requested.
        state.ui.teleopZeroingSensors = zeroFtSensor;
        setTeleopStartBusy(true);
        try {
            await api("/teleop/start", {
                method: "POST",
                body: JSON.stringify({ zero_ft_sensor: zeroFtSensor }),
            });
            await refreshTeleopStatus();
        } catch (error) {
            showToast(error.message, true);
        } finally {
            setTeleopStartBusy(false);
        }
    };
    byId("teleop-engage").onclick = async () => {
        const engaged = !!state.teleopStatus?.teleop?.engaged;
        const endpoint = engaged ? "/teleop/disengage" : "/teleop/engage";
        try {
            // The POST returns the authoritative post-action snapshot. Claim a
            // fresh sequence number when it lands and apply it unconditionally:
            // claiming the newest number invalidates every /teleop/status poll
            // that started earlier (including ones in flight during the POST),
            // so none of their older snapshots can write state.teleopStatus and
            // revert the button afterward.
            const teleop = await api(endpoint, { method: "POST" });
            ++teleopStatusSeq;
            if (state.teleopStatus) {
                state.teleopStatus = { ...state.teleopStatus, teleop };
                renderTeleop();
            }
            await refreshTeleopStatus();
        } catch (error) {
            showToast(error.message, true);
        }
    };
    byId("teleop-home").onclick = async () => {
        try {
            setTeleopHomeBusy(true);
            const result = await api("/teleop/home", { method: "POST" });
            if (result.error) {
                showToast(result.error, true);
            } else if (result.warnings?.length) {
                showToast(result.warnings.join(" | "), true);
            } else {
                showToast("Home reset command sent.");
            }
        } catch (error) {
            showToast(error.message, true);
        } finally {
            try {
                await refreshTeleopStatus();
            } catch (error) {
                showToast(error.message, true);
            }
            setTeleopHomeBusy(false);
        }
    };
    const jobNameField = byId("record-job-name");
    if (jobNameField && !jobNameField.value) {
        // Prefill the last-used job name (cached across reloads) so a session
        // resumes the operator's job; fall back to the default when none is
        // cached, so the very first episode is still grouped.
        jobNameField.value = loadCachedJobName() || DEFAULT_JOB_NAME;
    }
    byId("record-toggle").onclick = async () => {
        // The single toggle button stops an in-progress recording and otherwise
        // starts a new one, mirroring the teleop power control.
        const recording = state.teleopStatus?.recording || {};
        if (recording.active) {
            try {
                await api("/teleop/recording/stop", { method: "POST" });
                await refreshTeleopStatus();
            } catch (error) {
                showToast(error.message, true);
            }
            return;
        }
        try {
            if (!state.recordingEntries.length) {
                showToast("Select at least one recording entry.", true);
                return;
            }
            setRecordingStartBusy(true);
            // The job name groups episodes on disk under episodes/<job_name>/.
            // Fall back to the default when the box is blank; the server
            // sanitizes it into a single safe path segment.
            const jobName = (byId("record-job-name")?.value || "").trim() || DEFAULT_JOB_NAME;
            // Remember this job name so the next session resumes it.
            saveCachedJobName(jobName);
            await api("/teleop/recording/start", {
                method: "POST",
                body: JSON.stringify({
                    task: "Dual-arm Flexiv teleoperation demonstration",
                    recording_entries: state.recordingEntries,
                    job_name: jobName,
                }),
            });
        } catch (error) {
            showToast(error.message, true);
        } finally {
            try {
                await refreshTeleopStatus();
            } catch (error) {
                showToast(error.message, true);
            }
            setRecordingStartBusy(false);
        }
    };
    byId("record-save").onclick = async () => {
        if (state.ui.recordingSaveBusy) {
            return;
        }
        try {
            setRecordingSaveBusy(true);
            state.ui.recordingSaveProgress = 0;
            startRecordingSavePolling();
            const result = await api("/teleop/recording/save", { method: "POST" });
            state.ui.recordingSaveProgress = 100;
            renderTeleop();
            showToast(`Saved ${result.episode_name}`);
        } catch (error) {
            showToast(error.message, true);
        } finally {
            stopRecordingSavePolling();
            try {
                await refreshTeleopStatus();
            } catch (error) {
                showToast(error.message, true);
            }
            setRecordingSaveBusy(false);
        }
    };
    byId("record-discard").onclick = async () => {
        const confirmed = window.confirm("Discard this recording? This cannot be undone.");
        if (!confirmed) {
            return;
        }
        try {
            const result = await api("/teleop/recording/discard", { method: "POST" });
            showToast(`Discarded ${result.episode_name}`);
            await refreshTeleopStatus();
        } catch (error) {
            showToast(error.message, true);
        }
    };

    byId("browser-close").onclick = closeBrowser;
    byId("browser-cancel").onclick = closeBrowser;
    byId("browser-select-all").onclick = () => {
        const selectablePaths = getBrowserSelectablePaths();
        const allSelected = selectablePaths.length > 0
            && selectablePaths.every((path) => state.pathBrowser.selected.includes(path));
        setBrowserSelection(allSelected ? [] : selectablePaths);
    };
    byId("browser-sort").onclick = () => {
        // Flip the time-created order and re-render the already-fetched items.
        state.pathBrowser.sortOrder =
            state.pathBrowser.sortOrder === "asc" ? "desc" : "asc";
        state.pathBrowser.items = sortBrowserItems(state.pathBrowser.items);
        syncBrowserDialogChrome();
        renderBrowserList();
    };
    byId("browser-up").onclick = () => {
        const parts = state.pathBrowser.currentPath.split("/").filter(Boolean);
        const parentPath = parts.length ? `/${parts.slice(0, -1).join("/")}` || "/" : "/";
        const rootPath = state.pathBrowser.rootPath || "/";
        const parent = parentPath.startsWith(rootPath) ? parentPath : rootPath;
        refreshBrowser(parent).catch((error) => showToast(error.message, true));
    };
    byId("browser-confirm").onclick = () => {
        if (!state.pathBrowser.onConfirm) {
            return;
        }
        const selection = state.pathBrowser.selected.length
            ? state.pathBrowser.selected
            : state.pathBrowser.fallbackToCurrentPath
                ? [state.pathBrowser.currentPath]
                : [];
        if (!selection.length) {
            return;
        }
        state.pathBrowser.onConfirm(selection);
    };
}

async function init() {
    bindGlobalEvents();
    renderNotificationCenter();
    await refreshSummary();
    renderTeleop();
    renderTraining();
    setActiveView("home");
    // Watch for robot faults on every page, independent of the active view.
    startFaultPolling();
}

init().catch((error) => showToast(error.message, true));
