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
    combinedPreview: null,
    combinedPath: "",
    combineProgress: null,
    previewSeries: null,
    combinedSeries: null,
    previewFrame: 0,
    combinedFrame: 0,
    previewPlaying: false,
    combinedPlaying: false,
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
    },
    ui: {
        teleopRefreshBusy: false,
        teleopBootstrapBusy: false,
        teleopHomeBusy: false,
        recordingStartBusy: false,
        recordingSaveBusy: false,
        recordingSaveProgress: 0,
        serviceResetBusy: {
            teleop_service: false,
            robot_data_service: false,
            cameras: false,
        },
    },
    telemetryHistory: {
        left: [],
        right: [],
    },
    cameraFeeds: {},
    recordingEntries: [],
    notifications: {
        items: [],
        unreadCount: 0,
        open: false,
        nextId: 1,
        lastTeleopIssueSignature: "",
    },
};

const TELEMETRY_HISTORY_LIMIT = 90;
const TELEMETRY_FPS_OK_MIN = 0.55;
const TELEOP_BOOTSTRAP_TIMEOUT_MS = 4000;
const RECORDING_ENTRY_OPTIONS = [
    {
        id: "observation.images.ego",
        label: "observation.images.ego",
        bucket: "image",
        sourceField: "ego",
    },
    {
        id: "observation.images.left_wrist",
        label: "observation.images.left_wrist",
        bucket: "image",
        sourceField: "left_wrist",
    },
    {
        id: "observation.images.right_wrist",
        label: "observation.images.right_wrist",
        bucket: "image",
        sourceField: "right_wrist",
    },
    {
        id: "observation.state.tcp_pose",
        label: "observation.state.tcp_pose",
        bucket: "observation",
        payload: "cartesian_state",
        sourceField: "tcp_pose",
    },
    {
        id: "observation.state.tcp_twist",
        label: "observation.state.tcp_twist",
        bucket: "observation",
        payload: "cartesian_state",
        sourceField: "tcp_vel",
    },
    {
        id: "observation.state.tcp_wrench",
        label: "observation.state.tcp_wrench",
        bucket: "observation",
        payload: "cartesian_state",
        sourceField: "ext_wrench_in_world",
    },
    {
        id: "action.tcp_pose",
        label: "action.tcp_pose",
        bucket: "action",
        payload: "cartesian_command",
        sourceField: "tcp_pose_des",
    },
    {
        id: "action.tcp_twist",
        label: "action.tcp_twist",
        bucket: "action",
        payload: "cartesian_command",
        sourceField: "tcp_vel_des",
    },
    {
        id: "action.tcp_wrench",
        label: "action.tcp_wrench",
        bucket: "action",
        payload: "cartesian_command",
        sourceField: "wrench_des_in_ctrl_frame",
    },
];
const DEFAULT_RECORDING_ENTRY_IDS = RECORDING_ENTRY_OPTIONS.map((option) => option.id);
const SERVICE_RESET_TARGETS = {
    teleop_service: "teleop",
    robot_data_service: "ddk",
    cameras: "cameras",
};
const SERVICE_NAME_TO_KEY = {
    teleop: "teleop_service",
    ddk: "robot_data_service",
    cameras: "cameras",
};
const SERVICE_RESET_MESSAGES = {
    teleop_service: "Reconnect teleoperation service",
    robot_data_service: "Reconnect robot data service",
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
        colors: ["#8de0ff", "#86e4a8", "#ffbf7a"],
    },
};

const DATASET_PLOT_GROUPS = [
    {
        id: "tcp_pose",
        title: "TCP Pose",
        units: "",
        stateKey: "observation.state.tcp_pose",
        actionKey: "action.tcp_pose",
        labels: ["x", "y", "z", "rx", "ry", "rz"],
        stateColors: ["#8de0ff", "#86e4a8", "#ffbf7a", "#c78dff", "#ff8da8", "#a8d8ff"],
        actionColors: ["#4db8db", "#4dba72", "#db9a4d", "#9a4ddb", "#db4d6a", "#6aa8db"],
    },
    {
        id: "tcp_twist",
        title: "TCP Twist",
        units: "",
        stateKey: "observation.state.tcp_twist",
        actionKey: "action.tcp_twist",
        labels: ["vx", "vy", "vz", "wx", "wy", "wz"],
        stateColors: ["#8de0ff", "#86e4a8", "#ffbf7a", "#c78dff", "#ff8da8", "#a8d8ff"],
        actionColors: ["#4db8db", "#4dba72", "#db9a4d", "#9a4ddb", "#db4d6a", "#6aa8db"],
    },
    {
        id: "tcp_wrench",
        title: "TCP Wrench",
        units: "",
        stateKey: "observation.state.tcp_wrench",
        actionKey: "action.tcp_wrench",
        labels: ["fx", "fy", "fz", "mx", "my", "mz"],
        stateColors: ["#8de0ff", "#86e4a8", "#ffbf7a", "#c78dff", "#ff8da8", "#a8d8ff"],
        actionColors: ["#4db8db", "#4dba72", "#db9a4d", "#9a4ddb", "#db4d6a", "#6aa8db"],
    },
];

let _previewAnimFrame = null;
let _combinedAnimFrame = null;

function _buildDatasetPlotSvg(seriesData, group, numFrames, currentFrame) {
    const width = 960;
    const height = 540;
    const left = 44;
    const right = 18;
    const top = 18;
    const bottom = 24;
    const innerWidth = width - left - right;
    const innerHeight = height - top - bottom;

    // Collect all values for scale
    const allValues = [];
    for (const prefix of [group.stateKey, group.actionKey]) {
        for (let i = 0; i < group.labels.length; i++) {
            const key = `${prefix}.${i}`;
            const arr = seriesData[key];
            if (arr) {
                arr.forEach((v) => { if (v !== null && Number.isFinite(v)) allValues.push(v); });
            }
        }
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

    // Playhead
    if (numFrames > 1 && currentFrame >= 0) {
        const px = left + (currentFrame / (numFrames - 1)) * innerWidth;
        lines.push(`<line class="dataset-plot__playhead" x1="${px}" y1="${top}" x2="${px}" y2="${height - bottom}"></line>`);
    }

    // Paths
    const paths = [];
    const drawSeries = (prefix, colors, dash) => {
        for (let ci = 0; ci < group.labels.length; ci++) {
            const key = `${prefix}.${ci}`;
            const arr = seriesData[key];
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
    drawSeries(group.stateKey, group.stateColors, false);
    drawSeries(group.actionKey, group.actionColors, true);

    return `<svg class="trend-chart__svg" viewBox="0 0 ${width} ${height}" aria-hidden="true"><g>${lines.join("")}</g>${paths.join("")}</svg>`;
}

function _buildDatasetPlotLegend(seriesData, group) {
    const items = [];
    // State legends
    for (let i = 0; i < group.labels.length; i++) {
        const key = `${group.stateKey}.${i}`;
        const hasData = !!(seriesData[key] && seriesData[key].some((v) => v !== null));
        items.push(`
            <span class="trend-chart__legend-item${hasData ? "" : " trend-chart__legend-item--dim"}">
                <span class="trend-chart__swatch" style="--swatch:${group.stateColors[i]}"></span>
                <strong>state.${group.labels[i]}</strong>
            </span>
        `);
    }
    // Action legends (dashed indicator)
    for (let i = 0; i < group.labels.length; i++) {
        const key = `${group.actionKey}.${i}`;
        const hasData = !!(seriesData[key] && seriesData[key].some((v) => v !== null));
        items.push(`
            <span class="trend-chart__legend-item${hasData ? "" : " trend-chart__legend-item--dim"}">
                <span class="trend-chart__swatch trend-chart__swatch--dashed" style="--swatch:${group.actionColors[i]}"></span>
                <strong>action.${group.labels[i]}</strong>
            </span>
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

async function _pollCombineProgress() {
    while (true) {
        await new Promise((r) => setTimeout(r, 400));
        if (state.processingStep !== 3) return;
        try {
            const prog = await api("/datasets/combine-progress");
            state.combineProgress = prog;
            if (prog.status === "done") {
                const result = prog.result;
                state.combinedPath = result.root;
                // Hide progress block and show loading overlay
                const progressBlock = document.querySelector(".merge-progress-block");
                if (progressBlock) progressBlock.classList.add("hidden");
                const previewBlock = byId("combined-preview-block");
                if (previewBlock) {
                    previewBlock.classList.remove("hidden");
                    _showPreviewLoadingOverlay("combined-preview-block");
                }
                state.combinedPreview = await api(`/datasets/preview?path=${encodeURIComponent(result.root)}`);
                state.combinedFrame = 0;
                state.combinedPlaying = false;
                _stopDatasetPlayback("combinedPlaying");
                try {
                    state.combinedSeries = await api(`/datasets/series?path=${encodeURIComponent(result.root)}`);
                } catch (_) {
                    state.combinedSeries = null;
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

    // Camera feeds
    const feedsHtml = (preview.camera_keys || []).map((cameraKey) => `
        <div class="feed">
            <div class="feed__header"><span>${cameraKey}</span></div>
            <div class="feed__placeholder" data-render-mode="live">
                <img class="feed__image dataset-frame-img" data-camera-key="${cameraKey}" src="/datasets/frame-image?path=${encodeURIComponent(datasetPath)}&key=${encodeURIComponent(cameraKey)}&index=${currentFrame}" alt="${cameraKey} frame ${currentFrame}" />
            </div>
        </div>
    `).join("");

    // Playback controls
    const playbackHtml = `
        <div class="dataset-playback">
            <button class="dataset-playback__btn" data-action="${isPlaying ? "pause" : "play"}" type="button" aria-label="${isPlaying ? "Pause" : "Play"}">
                ${isPlaying ? "⏸" : "▶"}
            </button>
            <input type="range" class="dataset-playback__slider" min="0" max="${Math.max(numFrames - 1, 0)}" value="${currentFrame}" />
            <span class="dataset-playback__counter">${currentFrame + 1} / ${numFrames}</span>
        </div>
    `;

    // Plots
    const plotsHtml = DATASET_PLOT_GROUPS.map((group) => {
        // Check if any data exists for this group
        let hasAnyData = false;
        if (seriesData) {
            for (const prefix of [group.stateKey, group.actionKey]) {
                for (let i = 0; i < group.labels.length; i++) {
                    if (seriesData[`${prefix}.${i}`]) { hasAnyData = true; break; }
                }
                if (hasAnyData) break;
            }
        }
        const svg = seriesData ? _buildDatasetPlotSvg(seriesData, group, numFrames, currentFrame) : "";
        const legend = seriesData ? _buildDatasetPlotLegend(seriesData, group) : "";
        return `
            <div class="dataset-plot-card">
                <span class="eyebrow">${group.title}</span>
                <div class="trend-chart">
                    ${svg}
                    ${!hasAnyData ? `<div class="trend-chart__empty">No data available</div>` : ""}
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
            }
            // Update button
            playBtn.textContent = state[playingKey] ? "⏸" : "▶";
            playBtn.dataset.action = state[playingKey] ? "pause" : "play";
            playBtn.ariaLabel = state[playingKey] ? "Pause" : "Play";
        };
    }

    // Start playback if already playing
    if (isPlaying) {
        _startDatasetPlayback(container, preview, seriesData, frameKey, playingKey);
    }
}

function _updateDatasetFrames(container, preview, seriesData, frameKey, playingKey) {
    const currentFrame = state[frameKey];
    const numFrames = preview.num_frames || 0;
    const datasetPath = preview.path;

    // Update images
    container.querySelectorAll(".dataset-frame-img").forEach((img) => {
        const key = img.dataset.cameraKey;
        img.src = `/datasets/frame-image?path=${encodeURIComponent(datasetPath)}&key=${encodeURIComponent(key)}&index=${currentFrame}`;
    });

    // Update slider and counter
    const slider = container.querySelector(".dataset-playback__slider");
    if (slider) slider.value = currentFrame;
    const counter = container.querySelector(".dataset-playback__counter");
    if (counter) counter.textContent = `${currentFrame + 1} / ${numFrames}`;

    // Update playheads in SVGs
    container.querySelectorAll(".dataset-plot-card").forEach((card, gi) => {
        const group = DATASET_PLOT_GROUPS[gi];
        if (!group || !seriesData) return;
        const chartDiv = card.querySelector(".trend-chart");
        if (!chartDiv) return;
        chartDiv.innerHTML = _buildDatasetPlotSvg(seriesData, group, numFrames, currentFrame);
        // Re-add "no data" overlay if needed
        let hasAnyData = false;
        for (const prefix of [group.stateKey, group.actionKey]) {
            for (let i = 0; i < group.labels.length; i++) {
                if (seriesData[`${prefix}.${i}`]) { hasAnyData = true; break; }
            }
            if (hasAnyData) break;
        }
        if (!hasAnyData) {
            chartDiv.innerHTML += `<div class="trend-chart__empty">No data available</div>`;
        }
    });
}

function _startDatasetPlayback(container, preview, seriesData, frameKey, playingKey) {
    const animKey = playingKey === "previewPlaying" ? "_previewAnimFrame" : "_combinedAnimFrame";
    _stopDatasetPlayback(playingKey);
    const fps = preview.fps || 30;
    const numFrames = preview.num_frames || 0;
    const interval = 1000 / fps;
    let lastTime = 0;

    function tick(timestamp) {
        if (!state[playingKey]) return;
        if (timestamp - lastTime >= interval) {
            lastTime = timestamp;
            state[frameKey] = (state[frameKey] + 1) % numFrames;
            _updateDatasetFrames(container, preview, seriesData, frameKey, playingKey);
        }
        if (playingKey === "previewPlaying") {
            _previewAnimFrame = requestAnimationFrame(tick);
        } else {
            _combinedAnimFrame = requestAnimationFrame(tick);
        }
    }

    if (playingKey === "previewPlaying") {
        _previewAnimFrame = requestAnimationFrame(tick);
    } else {
        _combinedAnimFrame = requestAnimationFrame(tick);
    }
}

function _stopDatasetPlayback(playingKey) {
    if (playingKey === "previewPlaying" && _previewAnimFrame) {
        cancelAnimationFrame(_previewAnimFrame);
        _previewAnimFrame = null;
    } else if (playingKey === "combinedPlaying" && _combinedAnimFrame) {
        cancelAnimationFrame(_combinedAnimFrame);
        _combinedAnimFrame = null;
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

function setTeleopBootstrapBusy(busy) {
    state.ui.teleopBootstrapBusy = busy;
    if (state.activeView === "teleoperation") {
        renderTeleop();
    }
}

function setTeleopHomeBusy(busy) {
    state.ui.teleopHomeBusy = busy;
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
    }, 1500);
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

    const robots = Object.values(teleopStatus.ddk?.robots || {});
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

function createTeleopSystemCard(serviceKey, service = {}) {
    const tone = service.tone || "neutral";
    const resetBusy = !!state.ui.serviceResetBusy[serviceKey];
    const serviceState = formatValue(service.state);
    const reconnectMessage = SERVICE_RESET_MESSAGES[serviceKey] || `Reconnect ${service.label || serviceKey}`;
    const card = document.createElement("div");
    card.className = "status-card teleop-system-card";
    card.innerHTML = `
        <div class="teleop-system-card__header">
            <div class="teleop-system-card__title">
                <span class="teleop-system-card__dot teleop-system-card__dot--${tone}" role="img" aria-label="${serviceState}" title="${serviceState}"></span>
                <span class="eyebrow teleop-system-card__label">${service.label || serviceKey}</span>
            </div>
            <button class="secondary-button icon-button teleop-system-card__reset ${resetBusy ? "icon-button--spinning" : ""}" type="button" aria-label="${reconnectMessage}" title="${reconnectMessage}" ${resetBusy ? "disabled" : ""}>
                ${RESET_ICON_SVG}
            </button>
        </div>
    `;

    const resetButton = card.querySelector("button");
    resetButton.onclick = () => resetTeleopSystemService(serviceKey).catch((error) => showToast(error.message, true));
    return card;
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
            { label: "Connect", action: () => controlHomeService("teleop", "connect"), className: "start-button" },
            { label: "Disconnect", action: () => controlHomeService("teleop", "disconnect"), className: "stop-button" },
        ],
        robot_data_service: [
            { label: "Connect", action: () => controlHomeService("ddk", "connect"), className: "start-button" },
            { label: "Disconnect", action: () => controlHomeService("ddk", "disconnect"), className: "stop-button" },
        ],
        cameras: [
            { label: "Connect", action: () => controlHomeService("cameras", "connect"), className: "start-button" },
            { label: "Disconnect", action: () => controlHomeService("cameras", "disconnect"), className: "stop-button" },
        ],
    };
    (definitions[serviceKey] || []).forEach((definition) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = definition.label;
        if (definition.className) {
            button.classList.add(definition.className);
        }
        button.onclick = definition.action;
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

function renderHomeRobotConfigInputs() {
    if (!state.summary) {
        return;
    }
    const sideLabels = ["LEFT", "RIGHT"];
    const robotConfig = state.summary.robot_config || {
        local_robot_serials: ["", ""],
        remote_robot_serials: ["", ""],
    };
    const configs = [
        ["home-local-robots", "local_robot_serials", robotConfig.local_robot_serials || ["", ""]],
        ["home-remote-robots", "remote_robot_serials", robotConfig.remote_robot_serials || ["", ""]],
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
        ["Merged datasets", state.summary.storage.combined],
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
        force: extractCartesianVector(robot?.cartesian_state, "force", "cartesian_state")
            || extractCartesianVector(robot?.cartesian_command, "force", "cartesian_command"),
        moment: extractCartesianVector(robot?.cartesian_state, "moment", "cartesian_state")
            || extractCartesianVector(robot?.cartesian_command, "moment", "cartesian_command"),
    };
}

function getRobotTelemetryForSide(side, teleopStatus) {
    const robots = teleopStatus.ddk?.robots || {};
    const sideIndex = side === "left" ? 0 : 1;
    const preferredSerial = state.summary?.robot_config?.remote_robot_serials?.[sideIndex];
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

function buildVectorGeometry(side, vector) {
    const baseX = side === "left" ? 40 : 200;
    const baseY = 118;
    let dx = side === "left" ? 92 : -92;
    let dy = -48;
    let magnitude = null;

    if (vector) {
        const [fx, fy, fz] = vector;
        magnitude = Math.hypot(fx, fy, fz);
        const inwardX = side === "left" ? fx : -fx;
        const upwardY = -fy;
        const planarMagnitude = Math.hypot(inwardX, upwardY);
        if (planarMagnitude > 1e-6) {
            const length = clamp(42 + Math.min(planarMagnitude, 30) * 2.2, 42, 118);
            dx = (inwardX / planarMagnitude) * length;
            dy = (upwardY / planarMagnitude) * length;
        }
    }

    const tipX = clamp(baseX + dx, 18, 222);
    const tipY = clamp(baseY + dy, 18, 144);
    const angle = Math.atan2(tipY - baseY, tipX - baseX);
    const headLength = 20;
    const headWidth = 12;
    const leftX = tipX - Math.cos(angle) * headLength + Math.sin(angle) * headWidth;
    const leftY = tipY - Math.sin(angle) * headLength - Math.cos(angle) * headWidth;
    const rightX = tipX - Math.cos(angle) * headLength - Math.sin(angle) * headWidth;
    const rightY = tipY - Math.sin(angle) * headLength + Math.cos(angle) * headWidth;

    return {
        magnitude,
        baseX,
        baseY,
        tipX,
        tipY,
        points: `${tipX},${tipY} ${leftX},${leftY} ${rightX},${rightY}`,
    };
}

function formatComponentChips(vector, kind) {
    const meta = TELEMETRY_SERIES[kind];
    return meta.labels.map((label, index) => `
        <span class="telemetry-chip">
            <strong>${label}</strong>
            <span>${vector ? `${vector[index].toFixed(1)} ${meta.units}` : "--"}</span>
        </span>
    `).join("");
}

function resolveFpsTone(fps, okMin) {
    if (!Number.isFinite(fps) || fps <= 0) {
        return "offline";
    }
    return fps < okMin ? "warning" : "ok";
}

function setFpsBadge(element, fps, okMin) {
    if (!element) {
        return;
    }
    const safeFps = Number.isFinite(fps) && fps > 0 ? fps : 0;
    const tone = resolveFpsTone(safeFps, okMin);
    element.className = `feed__fps feed__fps--${tone}`;
    element.innerHTML = `
        <span class="feed__fps-dot" aria-hidden="true"></span>
        <span>${safeFps.toFixed(1)} FPS</span>
    `;
}

function buildFpsBadgeMarkup(fps, okMin) {
    const safeFps = Number.isFinite(fps) && fps > 0 ? fps : 0;
    const tone = resolveFpsTone(safeFps, okMin);
    return `
        <strong class="feed__fps feed__fps--${tone}">
            <span class="feed__fps-dot" aria-hidden="true"></span>
            <span>${safeFps.toFixed(1)} FPS</span>
        </strong>
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

function computeTelemetryStreamFps(history, kind, currentVector) {
    if (!Array.isArray(currentVector)) {
        return 0;
    }

    const validSamples = history.filter((sample) => Array.isArray(sample[kind]));
    if (validSamples.length < 2) {
        return 0;
    }

    const recentSamples = validSamples.slice(-6);
    const deltas = [];
    for (let index = 1; index < recentSamples.length; index += 1) {
        const delta = recentSamples[index].timestamp - recentSamples[index - 1].timestamp;
        if (delta > 0) {
            deltas.push(delta);
        }
    }

    if (!deltas.length) {
        return 0;
    }

    const averageDeltaMs = deltas.reduce((sum, value) => sum + value, 0) / deltas.length;
    return averageDeltaMs > 0 ? 1000 / averageDeltaMs : 0;
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
    container.innerHTML = "";
    const selected = new Set(state.recordingEntries);
    const allSelected = DEFAULT_RECORDING_ENTRY_IDS.every((entryId) => selected.has(entryId));

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

function getConfiguredRemoteSerials() {
    return (state.summary?.robot_config?.remote_robot_serials || [])
        .map((serial) => String(serial || "").trim())
        .filter(Boolean);
}

function areSelectedRecordingEntriesAvailable(teleopStatus) {
    const selectedOptions = RECORDING_ENTRY_OPTIONS.filter((option) => state.recordingEntries.includes(option.id));
    if (!selectedOptions.length) {
        return false;
    }

    const configuredRemoteSerials = getConfiguredRemoteSerials();
    return selectedOptions.every((option) => {
        if (option.bucket === "image") {
            const camera = teleopStatus?.cameras?.cameras?.[option.sourceField];
            return !!camera?.started;
        }

        if (!configuredRemoteSerials.length) {
            return false;
        }

        return configuredRemoteSerials.every((serial) => {
            const robot = teleopStatus?.ddk?.robots?.[serial];
            const payload = robot?.[option.payload];
            return hasRecordingPayload(payload?.[option.sourceField]);
        });
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
        return {
            kind: "recording",
            line1: formatElapsed(seconds),
            line2: `${frames} frames captured`,
            animated: false,
            canStart: false,
            canStop: true,
        };
    }

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

    if (state.ui.recordingStartBusy || state.ui.teleopBootstrapBusy || !state.teleopBootstrapped || !teleopStatus) {
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
    const textMarkup = model.line2
        ? `
            <span class="recording-status__text recording-status__text--stacked">
                <span class="recording-status__line">${model.line1}</span>
                <span class="recording-status__line recording-status__line--secondary">${model.line2}</span>
            </span>
        `
        : `<span class="recording-status__text">${model.line1}</span>`;
    setMarkupIfChanged(
        status,
        `${model.kind}:${model.line1}:${model.line2 || ""}:${model.animated ? "animated" : "static"}`,
        `
            ${recordingStatusIconMarkup(model.kind)}
            ${textMarkup}
        `,
    );

    byId("record-start").disabled = !model.canStart;
    byId("record-stop").disabled = !model.canStop;
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
    const canHome = teleopReady && !state.ui.teleopHomeBusy && !teleopResetBusy;

    byId("teleop-start").disabled = !canStart;
    byId("teleop-stop").disabled = !canStop;
    byId("teleop-home").disabled = !canHome;
}

function renderForcePanel(side, robotEntry, telemetry, history) {
    const panel = byId(`${side}-force-panel`);
    if (!panel) {
        return;
    }

    const geometry = buildVectorGeometry(side, telemetry.force);
    const title = `${side.toUpperCase()} FORCE VECTOR`;
    const fpsMarkup = buildFpsBadgeMarkup(
        computeTelemetryStreamFps(history, "force", telemetry.force),
        TELEMETRY_FPS_OK_MIN,
    );
    if (geometry.magnitude === null) {
        setMarkupIfChanged(
            panel,
            `${side}:force:awaiting`,
            `
                <div class="telemetry-card__header">
                    <div>
                        <span class="eyebrow">${title}</span>
                    </div>
                    ${fpsMarkup}
                </div>
                <div class="vector-panel vector-panel--empty">
                    ${buildAwaitingDataMarkup()}
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
            ${fpsMarkup}
        </div>
        <div class="vector-panel vector-panel--live">
            <svg class="vector-panel__svg" viewBox="0 0 240 160" aria-hidden="true">
                <line class="vector-panel__shaft" x1="${geometry.baseX}" y1="${geometry.baseY}" x2="${geometry.tipX}" y2="${geometry.tipY}"></line>
                <polygon class="vector-panel__head" points="${geometry.points}"></polygon>
            </svg>
            <div class="vector-panel__meta">
                <div class="telemetry-chip-row">${formatComponentChips(telemetry.force, "force")}</div>
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

function buildTrendGrid(scale) {
    const width = 960;
    const height = 540;
    const left = 34;
    const right = 18;
    const top = 18;
    const bottom = 24;
    const innerWidth = width - left - right;
    const innerHeight = height - top - bottom;
    const lines = [];

    for (let index = 0; index <= 5; index += 1) {
        const x = left + (innerWidth * index) / 5;
        lines.push(`<line class="trend-chart__grid-line" x1="${x}" y1="${top}" x2="${x}" y2="${height - bottom}"></line>`);
    }
    for (let index = 0; index <= 4; index += 1) {
        const y = top + (innerHeight * index) / 4;
        lines.push(`<line class="trend-chart__grid-line" x1="${left}" y1="${y}" x2="${width - right}" y2="${y}"></line>`);
    }
    if (scale.hasData && scale.min < 0 && scale.max > 0) {
        const zeroY = top + (1 - ((0 - scale.min) / (scale.max - scale.min))) * innerHeight;
        lines.push(`<line class="trend-chart__zero" x1="${left}" y1="${zeroY}" x2="${width - right}" y2="${zeroY}"></line>`);
    }
    return `<g>${lines.join("")}</g>`;
}

function buildTrendPath(history, kind, componentIndex, scale) {
    if (!scale.hasData || !history.length) {
        return "";
    }

    const width = 960;
    const height = 540;
    const left = 34;
    const right = 18;
    const top = 18;
    const bottom = 24;
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
    const title = `${side.toUpperCase()} ${kind === "force" ? "CARTESIAN FORCE" : "CARTESIAN MOMENT"}`;
    const hasLiveData = Array.isArray(currentVector);
    const scale = hasLiveData
        ? computeTelemetryScale(history, kind)
        : { min: -1, max: 1, hasData: false };
    const paths = meta.colors.map((color, index) => {
        const d = buildTrendPath(history, kind, index, scale);
        return d ? `<path class="trend-chart__line" style="--trend-color:${color}" d="${d}"></path>` : "";
    }).join("");
    const fpsMarkup = buildFpsBadgeMarkup(
        computeTelemetryStreamFps(history, kind, currentVector),
        TELEMETRY_FPS_OK_MIN,
    );

    if (!scale.hasData) {
        setMarkupIfChanged(
            panel,
            `${side}:${kind}:awaiting`,
            `
                <div class="telemetry-card__header">
                    <div>
                        <span class="eyebrow">${title}</span>
                    </div>
                    ${fpsMarkup}
                </div>
                <div class="trend-chart">
                    <svg class="trend-chart__svg" viewBox="0 0 960 540" aria-hidden="true">
                        ${buildTrendGrid(scale)}
                    </svg>
                    <div class="trend-chart__empty">${buildAwaitingDataMarkup()}</div>
                </div>
                <div class="trend-chart__legend">
                    ${meta.labels.map((label, index) => `
                        <span class="trend-chart__legend-item">
                            <span class="trend-chart__swatch" style="--swatch:${meta.colors[index]}"></span>
                            <strong>${label}</strong>
                            <span>--</span>
                        </span>
                    `).join("")}
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
            ${fpsMarkup}
        </div>
        <div class="trend-chart">
            <svg class="trend-chart__svg" viewBox="0 0 960 540" aria-hidden="true">
                ${buildTrendGrid(scale)}
                ${paths}
            </svg>
            ${scale.hasData ? "" : `<div class="trend-chart__empty">${buildAwaitingDataMarkup()}</div>`}
        </div>
        <div class="trend-chart__legend">
            ${meta.labels.map((label, index) => `
                <span class="trend-chart__legend-item">
                    <span class="trend-chart__swatch" style="--swatch:${meta.colors[index]}"></span>
                    <strong>${label}</strong>
                    <span>${currentVector ? `${currentVector[index].toFixed(1)} ${meta.units}` : "--"}</span>
                </span>
            `).join("")}
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
    const result = await api(`/system/services/${serviceName}/${action}`, { method: "POST" });
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
    const labels = {
        teleop: "Teleop service",
        ddk: "Robot data service",
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
        ddk: { robots: {}, errors: {} },
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
    grid.innerHTML = "";
    const services = teleopStatus.services || state.summary?.services || {};
    ["teleop_service", "robot_data_service", "cameras"].forEach((serviceKey) => {
        grid.appendChild(createTeleopSystemCard(serviceKey, services[serviceKey] || {}));
    });

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
    issues.push(...Object.values(teleopStatus.ddk.errors || {}));
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
                        <button id="training-combine" type="button" ${state.selectedEpisodes.length ? "" : "disabled"}>Merge Selected Episodes</button>
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
          <strong>${index + 1}</strong>
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
        byId("training-combine").onclick = async () => {
            try {
                state.processingStep = 3;
                state.combineProgress = null;
                state.combinedPreview = null;
                state.combinedSeries = null;
                renderProcessing();
                await api("/datasets/combine", {
                    method: "POST",
                    body: JSON.stringify({ episode_paths: state.selectedEpisodes, output_name: `merged-${Date.now()}` }),
                });
                // Poll for progress
                await _pollCombineProgress();
            } catch (error) {
                showToast(error.message, true);
                state.processingStep = 2;
                renderProcessing();
            }
        };
        return;
    }

    if (state.processingStep === 3) {
        const prog = state.combineProgress;
        const merging = !state.combinedPreview;
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
            <div class="${state.combinedPreview ? "" : "hidden"}" id="combined-preview-block"></div>
            <div class="control-bar"><button class="secondary-button" id="combine-prev" type="button">Previous Step</button></div>
        `;
        if (state.combinedPreview) {
            renderDatasetPreviewBlock("combined-preview-block", state.combinedPreview, state.combinedSeries?.series || null, "combinedFrame", "combinedPlaying");
        }
        byId("combine-prev").onclick = () => {
            state.processingStep = 2;
            renderProcessing();
        };
        return;
    }
}

function renderTraining() {
    const container = byId("training-content");

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
        startPath: state.summary.storage.combined,
        rootPath: state.summary.storage.combined,
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
    if (state.ui.teleopBootstrapBusy) {
        return;
    }
    if (state.teleopBootstrapped) {
        await refreshTeleopStatus();
        if (hasActiveTeleopServices(state.teleopStatus)) {
            startTeleopPolling();
        }
        return;
    }
    setTeleopBootstrapBusy(true);
    setComponentLoading("teleop-cameras-wrap", true, "Initializing modules");
    stopTeleopPolling();
    let bootstrapResult = null;
    try {
        bootstrapResult = await api(
            "/teleop/bootstrap",
            { method: "POST" },
            {
                timeoutMs: TELEOP_BOOTSTRAP_TIMEOUT_MS,
                timeoutMessage: "Automatic data-collection initialization timed out. Use the Connect buttons after the hardware is ready.",
            },
        );
        state.teleopBootstrapped = true;
    } catch (error) {
        // Even on timeout, the server may have partially started services.
        // Refresh status to discover what's running and start polling if needed.
        try {
            await refreshTeleopStatus();
        } catch (_) {
            // Status fetch also failed — nothing to recover
        }
        if (hasActiveTeleopServices(state.teleopStatus)) {
            state.teleopBootstrapped = true;
            startTeleopPolling();
            pushNotification(error.message, "warning");
        } else {
            state.teleopBootstrapped = false;
            throw error;
        }
        return;
    } finally {
        setComponentLoading("teleop-cameras-wrap", false);
        setTeleopBootstrapBusy(false);
    }
    // Bootstrap API call succeeded — refresh status and start polling
    try {
        await refreshTeleopStatus();
    } catch (_) {
        // Non-fatal: polling will retry
    }
    if (bootstrapResult.ready || hasActiveTeleopServices(state.teleopStatus)) {
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
    byId("teleop-start").onclick = async () => {
        try {
            await api("/teleop/start", { method: "POST" });
            await refreshTeleopStatus();
        } catch (error) {
            showToast(error.message, true);
        }
    };
    byId("teleop-stop").onclick = async () => {
        try {
            await api("/teleop/stop", { method: "POST" });
            await refreshTeleopStatus();
        } catch (error) {
            showToast(error.message, true);
        }
    };
    byId("teleop-home").onclick = async () => {
        try {
            setTeleopHomeBusy(true);
            const result = await api("/teleop/home", { method: "POST" });
            if (result.warnings?.length) {
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
    byId("record-start").onclick = async () => {
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
    byId("record-stop").onclick = async () => {
        try {
            await api("/teleop/recording/stop", { method: "POST" });
            await refreshTeleopStatus();
        } catch (error) {
            showToast(error.message, true);
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
