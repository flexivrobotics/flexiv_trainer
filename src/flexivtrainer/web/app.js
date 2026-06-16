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
    teleopStatus: null,
    cameraConfig: null,
    trainingPolicies: null,
    trainingStatus: null,
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
    previewSeries: null,
    mergedSeries: null,
    previewFrame: 0,
    mergedFrame: 0,
    previewPlaying: false,
    mergedPlaying: false,
    // Training page state
    mergedDatasetPath: "",
    mergedDatasetPreview: null,
    mergedDatasetSeries: null,
    mergedDatasetFrame: 0,
    mergedDatasetPlaying: false,
    outputDir: "",
    selectedPolicy: "diffusion",
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
        // Smoothed realtime recording FPS, derived from the change in captured
        // frames between status polls. Reset whenever a recording (re)starts.
        recordingFps: 0,
        recordingFpsSample: null,
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
    datasetPlotScope: {},
    notifications: {
        items: [],
        unreadCount: 0,
        open: false,
        nextId: 1,
        lastTeleopIssueSignature: "",
    },
};

const TELEMETRY_HISTORY_LIMIT = 90;
// How often the teleop view polls /teleop/status. This sets the telemetry
// refresh rate (e.g. 100ms ≈ 10 FPS); refreshTeleopStatus() self-throttles via
// a queue, so requests never pile up if the backend is briefly slower.
const TELEOP_POLL_INTERVAL_MS = 100;
// Arms in capture order (index 0 = left follower, 1 = right follower) and the
// per-metric vectors each arm exposes. The checklist offers one toggle per
// (side, metric); selected metrics are concatenated into that arm's grouped
// state/action feature in the dataset.
const RECORDING_ARM_SIDES = [
    { side: "left_arm", index: 0 },
    { side: "right_arm", index: 1 },
];
const RECORDING_METRICS = [
    { metric: "tcp_pose", stateField: "tcp_pose", actionField: "tcp_pose_d" },
    { metric: "tcp_twist", stateField: "tcp_vel", actionField: "tcp_vel_d" },
    { metric: "tcp_wrench", stateField: "ext_wrench_in_world", actionField: "ext_wrench_d" },
];

function buildArmMetricRecordingOptions() {
    const options = [];
    for (const { side, index } of RECORDING_ARM_SIDES) {
        for (const { metric, stateField } of RECORDING_METRICS) {
            options.push({
                id: `observation.state.${side}.${metric}`,
                label: `observation.state.${side}.${metric}`,
                bucket: "observation",
                payload: "states",
                side: index,
                verifyField: stateField,
            });
        }
    }
    for (const { side, index } of RECORDING_ARM_SIDES) {
        for (const { metric, actionField } of RECORDING_METRICS) {
            options.push({
                id: `action.${side}.${metric}`,
                label: `action.${side}.${metric}`,
                bucket: "action",
                payload: "actions",
                side: index,
                verifyField: actionField,
            });
        }
    }
    return options;
}

const RECORDING_ENTRY_OPTIONS = [
    { id: "observation.images.ego", label: "observation.images.ego", bucket: "image", sourceField: "ego" },
    { id: "observation.images.left_wrist", label: "observation.images.left_wrist", bucket: "image", sourceField: "left_wrist" },
    { id: "observation.images.right_wrist", label: "observation.images.right_wrist", bucket: "image", sourceField: "right_wrist" },
    ...buildArmMetricRecordingOptions(),
];
const DEFAULT_RECORDING_ENTRY_IDS = RECORDING_ENTRY_OPTIONS.map((option) => option.id);
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
state.recordingEntries = [...DEFAULT_RECORDING_ENTRY_IDS];

let teleopStatusRefreshPromise = null;
let teleopStatusRefreshQueued = false;

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
};
const DATASET_METRIC_ORDER = { tcp_pose: 0, tcp_twist: 1, tcp_wrench: 2 };
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
        const match = key.match(/^(observation\.state|action)\.([a-z0-9_]+)\.(tcp_pose|tcp_twist|tcp_wrench)\.(.+)$/);
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
        ? value.toFixed(2)
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

async function _pollMergeProgress() {
    while (true) {
        await new Promise((r) => setTimeout(r, 400));
        if (state.processingStep !== 3) return;
        try {
            const prog = await api("/datasets/merge-progress");
            state.mergeProgress = prog;
            if (prog.status === "done") {
                const result = prog.result;
                state.mergedPath = result.root;
                // Hide progress block and show loading overlay
                const progressBlock = document.querySelector(".merge-progress-block");
                if (progressBlock) progressBlock.classList.add("hidden");
                const previewBlock = byId("merged-preview-block");
                if (previewBlock) {
                    previewBlock.classList.remove("hidden");
                    _showPreviewLoadingOverlay("merged-preview-block");
                }
                state.mergedPreview = await api(`/datasets/preview?path=${encodeURIComponent(result.root)}`);
                state.mergedFrame = 0;
                state.mergedPlaying = false;
                _stopDatasetPlayback("mergedPlaying");
                try {
                    state.mergedSeries = await api(`/datasets/series?path=${encodeURIComponent(result.root)}`);
                } catch (_) {
                    state.mergedSeries = null;
                }
                renderProcessing();
                return;
            } else if (prog.status === "error") {
                showToast(prog.error || "Merge failed", true);
                state.processingStep = 2;
                renderProcessing();
                return;
            }
            // Update progress bars in place without full re-render
            _updateMergeProgressBars(prog);
        } catch (_) {
            // endpoint not available yet, keep polling
        }
    }
}

function _updateMergeProgressBars(prog) {
    const overallPercent = prog.total_episodes ? Math.round((prog.episode_index / prog.total_episodes) * 100) : 0;
    const overallLabel = `${prog.episode_index}/${prog.total_episodes}`;
    const block = document.querySelector(".merge-progress-block");
    if (!block) return;
    const bar = block.querySelector(".progress-bar");
    if (bar) {
        bar.querySelector("span:first-child").style.width = `${overallPercent}%`;
        const txt = bar.querySelector(".progress-bar__text");
        if (txt) txt.textContent = overallLabel;
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
    const feedsHtml = (preview.camera_keys || []).map((cameraKey) => `
        <div class="feed">
            <div class="feed__header"><span>${cameraKey}</span></div>
            <div class="feed__placeholder" data-render-mode="live">
                <video class="feed__image dataset-frame-video" data-camera-key="${cameraKey}" src="/datasets/video?path=${encodeURIComponent(datasetPath)}&key=${encodeURIComponent(cameraKey)}" muted playsinline preload="auto" loop></video>
            </div>
        </div>
    `).join("");

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

// Seek every camera <video> to the given frame (time = frame / fps). Frames are
// concatenated at constant fps, so the frame index maps linearly to video time.
function _seekDatasetVideos(container, preview, frameIndex) {
    const t = frameIndex / _datasetPreviewFps(preview);
    container.querySelectorAll(".dataset-frame-video").forEach((video) => {
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
        // Refresh the per-legend current-frame values so they track the playhead.
        const legendDiv = card.querySelector(".trend-chart__legend");
        if (legendDiv) {
            legendDiv.innerHTML = _buildDatasetPlotLegend(seriesData, group, currentFrame, scope, gi);
        }
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
        const frame = Math.min(
            numFrames - 1,
            Math.max(0, Math.round(master.currentTime * fps))
        );
        if (frame !== state[frameKey]) {
            state[frameKey] = frame;
            _renderDatasetFrameMeta(container, preview, seriesData, frameKey);
            // Correct any drift between feeds without disrupting the master.
            for (let i = 1; i < videos.length; i++) {
                if (Math.abs(videos[i].currentTime - master.currentTime) > 0.15) {
                    videos[i].currentTime = master.currentTime;
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
    "Enter all four robot serial numbers (LEFT/RIGHT leader and follower) before connecting the teleop service.";

// Teleoperation needs both leader and both follower serials. Require all four
// before the teleop service can be connected or reconnected, so a half-typed
// configuration can't start a pairing with a missing arm.
function allRobotSerialsConfigured() {
    const config = state.summary?.robot_config;
    if (!config) return false;
    const required = [
        config.leader_robot_serials?.[0],
        config.leader_robot_serials?.[1],
        config.follower_robot_serials?.[0],
        config.follower_robot_serials?.[1],
    ];
    return required.every((serial) => typeof serial === "string" && serial.trim() !== "");
}

function renderHomeRobotConfigInputs() {
    if (!state.summary) {
        return;
    }
    const sideLabels = ["LEFT", "RIGHT"];
    const robotConfig = state.summary.robot_config || {
        leader_robot_serials: ["", ""],
        follower_robot_serials: ["", ""],
    };
    const configs = [
        ["home-leader-robots", "leader_robot_serials", robotConfig.leader_robot_serials || ["", ""]],
        ["home-follower-robots", "follower_robot_serials", robotConfig.follower_robot_serials || ["", ""]],
    ];
    configs.forEach(([containerId, key, serials]) => {
        const container = byId(containerId);
        container.innerHTML = "";
        serials.forEach((serial, index) => {
            const field = document.createElement("label");
            field.className = "robot-input-group";
            field.innerHTML = `
                <span>${sideLabels[index] || `Robot ${index + 1}`}</span>
                <input type="text" value="${serial}" placeholder="Enter robot serial number" />
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
        });
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

function renderHome() {
    if (!state.summary) {
        return;
    }
    renderHomeRobotConfigInputs();
    renderHomeStatus();
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

function getRobotTelemetryForSide(side, teleopStatus) {
    const robots = teleopStatus.robot_data?.robots || {};
    const sideIndex = side === "left" ? 0 : 1;
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
    const selected = new Set(state.recordingEntries);
    const allSelected = DEFAULT_RECORDING_ENTRY_IDS.every((entryId) => selected.has(entryId));

    // renderTeleop runs on every poll tick (~10x/sec). Rebuilding these
    // controls unconditionally made the Select All button and checkboxes flash
    // and swallowed clicks landing mid-rerender. Only rebuild when something
    // that affects the rendered output actually changed.
    const renderKey = `${locked ? 1 : 0}|${state.recordingEntries.join(",")}`;
    if (container.dataset.renderKey === renderKey) {
        return;
    }
    container.dataset.renderKey = renderKey;
    container.innerHTML = "";

    const selectAllButton = document.createElement("button");
    selectAllButton.className = "secondary-button recording-select-all-button";
    selectAllButton.type = "button";
    selectAllButton.disabled = locked;
    selectAllButton.textContent = allSelected ? "Deselect All" : "Select All";
    selectAllButton.onclick = () => {
        state.recordingEntries = allSelected ? [] : [...DEFAULT_RECORDING_ENTRY_IDS];
        renderRecordingOptions(recording);
        renderRecordingStatusPanel(state.teleopStatus);
    };
    container.appendChild(selectAllButton);

    RECORDING_ENTRY_OPTIONS.forEach((option) => {
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
            state.recordingEntries = DEFAULT_RECORDING_ENTRY_IDS.filter((entryId) => nextSelected.has(entryId));
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
    const selectedOptions = RECORDING_ENTRY_OPTIONS.filter((option) => state.recordingEntries.includes(option.id));
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

    updateRecordingToggleButton(model);
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
    const teleopReady = !!teleop.initialized && !teleop.started && !teleop.error && !teleop.fault;
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

function renderForcePanel(side, robotEntry, telemetry, history) {
    const panel = byId(`${side}-force-panel`);
    if (!panel) {
        return;
    }

    const title = `${side.toUpperCase()} CARTESIAN WRENCH`;
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

function renderTrendGraph(side, kind, history, currentVector) {
    const panel = byId(`${side}-${kind}-graph-panel`);
    if (!panel) {
        return;
    }

    const meta = TELEMETRY_SERIES[kind];
    const title = `${side.toUpperCase()} CARTESIAN ${kind === "force" ? "FORCE" : "MOMENT"}`;
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
    state.teleopStatus = await api("/teleop/status");
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

    const cameras = teleopStatus.cameras?.cameras || {};
    renderCameraFps("ego-fps", "ego", cameras.ego);
    renderCameraFps("left-wrist-fps", "left_wrist", cameras.left_wrist);
    renderCameraFps("right-wrist-fps", "right_wrist", cameras.right_wrist);

    const leftRobotEntry = getRobotTelemetryForSide("left", teleopStatus);
    const rightRobotEntry = getRobotTelemetryForSide("right", teleopStatus);
    const leftTelemetry = readRobotTelemetry(leftRobotEntry.robot);
    const rightTelemetry = readRobotTelemetry(rightRobotEntry.robot);
    if (state.teleopStatus) {
        appendTelemetrySample("left", leftTelemetry);
        appendTelemetrySample("right", rightTelemetry);
    }
    renderForcePanel("left", leftRobotEntry, leftTelemetry, state.telemetryHistory.left);
    renderForcePanel("right", rightRobotEntry, rightTelemetry, state.telemetryHistory.right);
    renderTrendGraph("left", "force", state.telemetryHistory.left, leftTelemetry.force);
    renderTrendGraph("left", "moment", state.telemetryHistory.left, leftTelemetry.moment);
    renderTrendGraph("right", "force", state.telemetryHistory.right, rightTelemetry.force);
    renderTrendGraph("right", "moment", state.telemetryHistory.right, rightTelemetry.moment);

    const grid = byId("teleop-status-grid");
    const services = teleopStatus.services || state.summary?.services || {};
    renderTeleopSystemCards(grid, services);

    renderRecordingStatusPanel(teleopStatus);
    renderRecordingActionButtons(teleopStatus.recording || {});
    updateTeleopControlButtons(teleopStatus);

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
                <button id="training-next-step" type="button" ${state.episodes.length ? "" : "disabled"}>Next</button>
            </div>
        `;
        const list = byId("load-episode-list");
        if (!state.episodes.length) {
            list.innerHTML = `<div class="episode-empty-state"><span>No episode datasets selected yet.</span></div>`;
        } else {
            state.episodes.forEach((episode, index) => {
                const row = document.createElement("div");
                row.className = "episode-entry-row";
                row.innerHTML = `
                    <div class="episode-entry-card">
                        <strong class="episode-entry-card__index">${index + 1}</strong>
                        <span class="episode-entry-card__divider" aria-hidden="true"></span>
                        <span class="episode-entry-card__name">${escapeHtml(episode.name)}</span>
                    </div>
                    <button class="round-icon-button round-icon-button--remove" data-remove-episode="${escapeHtml(episode.path)}" type="button" aria-label="Remove ${escapeHtml(episode.name)}" title="Remove ${escapeHtml(episode.name)}">
                        <span aria-hidden="true">&minus;</span>
                    </button>
        `;
                list.appendChild(row);
            });
        }
        byId("training-add-episode").onclick = () => openEpisodeBrowser();
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
                        <button class="secondary-button" id="training-select-all" type="button">${state.selectedEpisodes.length === state.episodes.length ? "Deselect All" : "Select All"}</button>
                    </div>
                    <div class="episode-list" id="training-episode-picker"></div>
                </aside>
                <div class="training-main">
                    <div id="episode-preview-block"></div>
                    <div class="control-bar">
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
            row.innerHTML = `
        <div class="episode-row__main">
          <input data-toggle-episode="${episode.path}" type="checkbox" ${state.selectedEpisodes.includes(episode.path) ? "checked" : ""} />
          <span>${escapeHtml(episode.name)}</span>
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
            try {
                state.processingStep = 3;
                state.mergeProgress = null;
                state.mergedPreview = null;
                state.mergedSeries = null;
                renderProcessing();
                await api("/datasets/merge", {
                    method: "POST",
                    body: JSON.stringify({ episode_paths: state.selectedEpisodes, output_name: `merged-${Date.now()}` }),
                });
                // Poll for progress
                await _pollMergeProgress();
            } catch (error) {
                showToast(error.message, true);
                state.processingStep = 2;
                renderProcessing();
            }
        };
        return;
    }

    if (state.processingStep === 3) {
        const prog = state.mergeProgress;
        const merging = !state.mergedPreview;
        const overallPercent = prog && prog.total_episodes ? Math.round((prog.episode_index / prog.total_episodes) * 100) : 0;
        const overallLabel = prog ? `${prog.episode_index}/${prog.total_episodes}` : "";
        container.innerHTML = `
            <div class="panel-header"><div><h2>Merged Dataset</h2></div></div>
            <div class="merge-progress-block ${merging ? "" : "hidden"}">
                <p class="merge-progress-block__title">Merging selected episodes...</p>
                <div class="merge-progress-item">
                    <div class="progress-bar progress-bar--thick"><span style="width: ${overallPercent}%"></span><span class="progress-bar__text">${overallLabel}</span></div>
                </div>
            </div>
            <div class="${state.mergedPreview ? "" : "hidden"}" id="merged-preview-block"></div>
            <div class="control-bar"><button class="secondary-button" id="merge-prev" type="button">Previous Step</button></div>
        `;
        if (state.mergedPreview) {
            renderDatasetPreviewBlock("merged-preview-block", state.mergedPreview, state.mergedSeries?.series || null, "mergedFrame", "mergedPlaying");
        }
        byId("merge-prev").onclick = () => {
            state.processingStep = 2;
            renderProcessing();
        };
        return;
    }
}

function renderTraining() {
    const container = byId("training-content");
    container.classList.toggle("has-playback-bar", state.trainingStep > 1);

    if (state.trainingStep === 1) {
        container.innerHTML = `
            <div class="panel-header panel-header--training-step">
                <div>
                    <h2 class="training-step-title">Select Training Dataset</h2>
                </div>
            </div>
            <div class="merged-dataset-entry" id="merged-dataset-entry">
                ${state.mergedDatasetPath
                ? `<div class="episode-entry-row"><div class="episode-entry-card"><strong class="episode-entry-card__index">1</strong><span class="episode-entry-card__divider" aria-hidden="true"></span><span class="episode-entry-card__name">${escapeHtml(state.mergedDatasetPath.split("/").pop())}</span></div><button class="round-icon-button round-icon-button--remove" id="training-remove-dataset" type="button" aria-label="Remove dataset" title="Remove dataset"><span aria-hidden="true">&minus;</span></button></div>`
                : `<div class="episode-empty-state"><span>No training dataset selected yet.</span></div>`
            }
            </div>
            <div id="merged-dataset-preview-block"></div>
            <div class="control-bar control-bar--episode-step">
                <button class="round-icon-button round-icon-button--add" id="training-browse-merged" type="button" aria-label="Browse datasets" title="Browse datasets">
                    <span aria-hidden="true">+</span>
                </button>
                <button id="training-next-step" type="button" ${state.mergedDatasetPath ? "" : "disabled"}>Next</button>
            </div>
        `;
        if (state.mergedDatasetPreview) {
            renderDatasetPreviewBlock("merged-dataset-preview-block", state.mergedDatasetPreview, state.mergedDatasetSeries?.series || null, "mergedDatasetFrame", "mergedDatasetPlaying");
        }
        byId("training-browse-merged").onclick = () => openMergedDatasetBrowser();
        const removeBtn = byId("training-remove-dataset");
        if (removeBtn) {
            removeBtn.onclick = () => {
                state.mergedDatasetPath = "";
                state.mergedDatasetPreview = null;
                state.mergedDatasetSeries = null;
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
        const catalog = state.trainingPolicies || { default: "diffusion", policies: {} };
        const policiesReady = !!state.trainingPolicies;
        container.innerHTML = `
            <div class="panel-header"><div><h2>Choose Training Policy</h2></div></div>
            <div class="component-wrapper" id="policy-grid-wrap" style="min-height:100px">
                <div class="policy-grid" id="policy-grid"></div>
                ${!policiesReady ? `<div class="component-loading-overlay"><div class="mini-progress-bar"><span></span></div><span class="component-loading-overlay__label">Loading policies…</span></div>` : ""}
            </div>
            <div class="output-picker"><div><p class="eyebrow">Training Output Directory</p><strong id="training-output-path">${state.outputDir || "No directory selected"}</strong></div><button class="secondary-button" id="training-pick-output" type="button">Choose Directory</button></div>
            <div class="control-bar"><button class="secondary-button" id="policy-prev" type="button">Previous Step</button><button id="policy-start" type="button" ${state.outputDir && policiesReady ? "" : "disabled"}>Start Training</button></div>
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
        byId("training-pick-output").onclick = () => openOutputBrowser();
        byId("policy-prev").onclick = () => {
            state.trainingStep = 1;
            renderTraining();
        };
        byId("policy-start").onclick = async () => {
            try {
                state.trainingStep = 3;
                renderTraining();
                state.trainingStatus = await api("/training/start", {
                    method: "POST",
                    body: JSON.stringify({ dataset_path: state.mergedDatasetPath, output_dir: state.outputDir, policy_type: state.selectedPolicy }),
                });
                renderTraining();
                window.clearInterval(state.intervals.training);
                state.intervals.training = window.setInterval(async () => {
                    if (state.activeView !== "training" || state.trainingStep !== 3) {
                        return;
                    }
                    state.trainingStatus = await api("/training/status");
                    renderTraining();
                }, 2000);
            } catch (error) {
                showToast(error.message, true);
                state.trainingStep = 2;
                renderTraining();
            }
        };
        return;
    }

    const status = state.trainingStatus || { status: "waiting", progress: 0, logs: [] };
    const done = status.status === "completed";
    const failed = status.status === "failed";
    container.innerHTML = `
        <div class="panel-header"><div><h2>Training Run</h2></div></div>
        <div class="progress-bar"><span style="width: ${status.progress || 0}%"></span></div>
        <div class="result-pill ${done ? "result-pill--success" : failed ? "result-pill--error" : ""}">${done ? "Training completed" : failed ? formatValue(status.error || "Training failed") : formatValue(status.status)}</div>
        <pre class="log-pane">${(status.logs || []).join("\n") || "Training logs will appear here."}</pre>
        <div class="control-bar"><button class="secondary-button" id="training-run-prev" type="button">Previous Step</button></div>
    `;
    byId("training-run-prev").onclick = () => {
        state.trainingStep = 2;
        renderTraining();
    };
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
        selectAllButton.textContent = allSelected ? "Deselect All" : "Select All";
    }

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
    state.pathBrowser.items = result.items || [];
    state.pathBrowser.selected = state.pathBrowser.selected.filter((selectedPath) =>
        state.pathBrowser.items.some((item) => item.path === selectedPath),
    );
    syncBrowserDialogChrome();
    byId("browser-path").textContent = result.path;
    const list = byId("browser-list");
    list.innerHTML = "";

    if (!state.pathBrowser.items.length) {
        list.innerHTML = `<div class="dialog-empty-state">${escapeHtml(state.pathBrowser.emptyMessage || "No entries available.")}</div>`;
        updateBrowserSelectionUi();
        byId("browser-modal").classList.remove("hidden");
        return;
    }

    state.pathBrowser.items.forEach((item) => {
        if (isEpisodeBrowserMode()) {
            const selectable = !!item.is_valid_episode;
            const row = document.createElement("label");
            row.className = `browser-item browser-item--episode ${selectable ? "" : "browser-item--invalid"}`.trim();
            row.dataset.browserPath = item.path;
            row.innerHTML = `
                <span class="browser-item__main">
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
    byId("browser-modal").classList.remove("hidden");
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
        pathNote: "Episodes stored under:",
        onConfirm: (paths) => {
            const orderedPaths = (state.pathBrowser.items || [])
                .map((item) => item.path)
                .filter((itemPath) => paths.includes(itemPath));
            orderedPaths.forEach((path) => {
                if (!state.episodes.some((item) => item.path === path)) {
                    state.episodes.push({ name: path.split("/").pop(), path });
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
        pathNote: "Datasets stored under:",
        onConfirm: async (paths) => {
            const path = paths[0];
            if (!path) return;
            state.mergedDatasetPath = path;
            closeBrowser();
            _showPreviewLoadingOverlay("merged-dataset-preview-block");
            try {
                state.mergedDatasetPreview = await api(`/datasets/preview?path=${encodeURIComponent(path)}`);
                state.mergedDatasetFrame = 0;
                state.mergedDatasetPlaying = false;
                _stopDatasetPlayback("mergedDatasetPlaying");
                try {
                    state.mergedDatasetSeries = await api(`/datasets/series?path=${encodeURIComponent(path)}`);
                } catch (_) {
                    state.mergedDatasetSeries = null;
                }
            } catch (error) {
                state.mergedDatasetPreview = null;
                state.mergedDatasetSeries = null;
                showToast(error.message, true);
            }
            renderTraining();
        },
    }).catch((error) => showToast(error.message, true));
}

function openOutputBrowser() {
    openBrowser({
        title: "Select Training Output Directory",
        startPath: state.summary.storage.training,
        directoriesOnly: true,
        multiSelect: false,
        onConfirm: (paths) => {
            state.outputDir = paths[0] || state.outputDir;
            closeBrowser();
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
    document.addEventListener("click", () => {
        if (state.notifications.open) {
            toggleNotificationCenter(false);
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
            await api(endpoint, { method: "POST" });
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
            await api("/teleop/recording/start", {
                method: "POST",
                body: JSON.stringify({
                    task: "Dual-arm Flexiv teleoperation demonstration",
                    recording_entries: state.recordingEntries,
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
}

init().catch((error) => showToast(error.message, true));
