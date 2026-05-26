const state = {
    activeView: "home",
    summary: null,
    teleopStatus: null,
    trainingPolicies: null,
    trainingStatus: null,
    teleopBootstrapped: false,
    trainingBootstrapped: false,
    trainingStep: 1,
    episodes: [],
    selectedEpisodes: [],
    preview: null,
    combinedPreview: null,
    combinedPath: "",
    outputDir: "",
    selectedPolicy: "diffusion",
    pathBrowser: {
        title: "",
        currentPath: "/",
        directoriesOnly: true,
        multiSelect: false,
        selected: [],
        onConfirm: null,
    },
    intervals: {
        teleop: null,
        training: null,
    },
    timers: {
        robotConfigSave: null,
    },
    telemetryHistory: {
        left: [],
        right: [],
    },
};

const TELEMETRY_HISTORY_LIMIT = 90;
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

function byId(id) {
    return document.getElementById(id);
}

async function api(path, init) {
    const response = await fetch(path, {
        headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
        ...init,
    });
    if (!response.ok) {
        throw new Error(`Request failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
}

function showToast(message, isError = false) {
    const toast = byId("toast");
    toast.textContent = message;
    toast.classList.remove("hidden", "toast--error", "toast--success");
    toast.classList.add(isError ? "toast--error" : "toast--success");
    window.clearTimeout(showToast._timer);
    showToast._timer = window.setTimeout(() => toast.classList.add("hidden"), 4000);
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

function createRecordingCard(recording) {
    const active = !!recording.active;
    const awaitingSave = !!recording.awaiting_save;
    const frames = Number(recording.frames_captured || 0);
    const fps = Number(recording.fps || 0);
    const seconds = fps > 0 ? frames / fps : 0;
    const tone = active || awaitingSave ? "ok" : "neutral";

    const card = document.createElement("div");
    card.className = `status-card status-card--${tone}`;
    if (active) {
        card.innerHTML = `
            <span class="eyebrow">Recording</span>
            <h3 class="recording-live">
                <span class="recording-live__dot" aria-hidden="true"></span>
                <span>${formatElapsed(seconds)} · ${frames} frames</span>
            </h3>
        `;
        return card;
    }

    const value = awaitingSave ? `${frames} frames captured` : "Idle";
    card.innerHTML = `<span class="eyebrow">Recording</span><h3>${value}</h3>`;
    return card;
}

function setActiveView(view) {
    state.activeView = view;
    document.querySelectorAll(".view").forEach((element) => {
        element.classList.toggle("view--active", element.dataset.view === view);
    });
    document.querySelectorAll("[data-nav]").forEach((element) => {
        const isBrand = element.classList.contains("brand");
        const active = element.dataset.nav === view || (view === "home" && isBrand);
        element.classList.toggle("nav-link--active", active && !isBrand);
        element.classList.toggle("brand--active", active && isBrand);
    });
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
        ["Combined datasets", state.summary.storage.combined],
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

function renderCameraFps(elementId, camera) {
    const element = byId(elementId);
    if (!element) {
        return;
    }

    const fps = Number(camera?.fps || 0);
    const state = !camera?.started || fps <= 0
        ? "offline"
        : fps < 29
            ? "warning"
            : "ok";
    element.className = `feed__fps feed__fps--${state}`;
    element.innerHTML = `
        <span class="feed__fps-dot" aria-hidden="true"></span>
        <span>${fps.toFixed(1)} FPS</span>
    `;
}

function renderForcePanel(side, robotEntry, telemetry) {
    const panel = byId(`${side}-force-panel`);
    if (!panel) {
        return;
    }

    const geometry = buildVectorGeometry(side, telemetry.force);
    const title = `${side.toUpperCase()} FORCE VECTOR`;
    const valueClass = geometry.magnitude === null
        ? "telemetry-card__value telemetry-card__value--pending"
        : "telemetry-card__value";
    panel.innerHTML = `
        <div class="telemetry-card__header">
            <div>
                <span class="eyebrow">${title}</span>
            </div>
            <strong class="${valueClass}">${geometry.magnitude === null ? "Awaiting data" : `${geometry.magnitude.toFixed(1)} N`}</strong>
        </div>
        <div class="vector-panel ${geometry.magnitude !== null ? "vector-panel--live" : ""}">
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
    const scale = computeTelemetryScale(history, kind);
    const paths = meta.colors.map((color, index) => {
        const d = buildTrendPath(history, kind, index, scale);
        return d ? `<path class="trend-chart__line" style="--trend-color:${color}" d="${d}"></path>` : "";
    }).join("");
    const statusText = scale.hasData
        ? `Auto scale ${scale.min.toFixed(1)} to ${scale.max.toFixed(1)} ${meta.units}`
        : "Awaiting data";
    const valueClass = scale.hasData
        ? "telemetry-card__value"
        : "telemetry-card__value telemetry-card__value--pending";

    panel.innerHTML = `
        <div class="telemetry-card__header">
            <div>
                <span class="eyebrow">${title}</span>
            </div>
            <strong class="${valueClass}">${statusText}</strong>
        </div>
        <div class="trend-chart">
            <svg class="trend-chart__svg" viewBox="0 0 960 540" aria-hidden="true">
                ${buildTrendGrid(scale)}
                ${paths}
            </svg>
            ${scale.hasData ? "" : `<div class="trend-chart__empty">Awaiting data</div>`}
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

async function controlHomeService(serviceName, action) {
    const result = await api(`/system/services/${serviceName}/${action}`, { method: "POST" });
    state.summary.services = result.services;
    renderHomeStatus();
    if (state.activeView === "teleoperation") {
        await refreshTeleopStatus();
    }
    const labels = {
        teleop: "Teleop service",
        ddk: "Robot data service",
        cameras: "Cameras",
    };
    showToast(`${labels[serviceName] || serviceName} ${action === "connect" ? "connected" : "disconnected"}.`);
}

function renderTeleop() {
    const teleopStatus = state.teleopStatus || {
        teleop: { started: false, initialized: false, error: null },
        ddk: { robots: {}, errors: {} },
        cameras: { cameras: {}, errors: {} },
        recording: {
            active: false,
            awaiting_save: false,
            frames_captured: 0,
        },
    };

    const cameras = teleopStatus.cameras?.cameras || {};
    renderCameraFps("ego-fps", cameras.ego);
    renderCameraFps("left-wrist-fps", cameras.left_wrist);
    renderCameraFps("right-wrist-fps", cameras.right_wrist);

    const leftRobotEntry = getRobotTelemetryForSide("left", teleopStatus);
    const rightRobotEntry = getRobotTelemetryForSide("right", teleopStatus);
    const leftTelemetry = readRobotTelemetry(leftRobotEntry.robot);
    const rightTelemetry = readRobotTelemetry(rightRobotEntry.robot);
    if (state.teleopStatus) {
        appendTelemetrySample("left", leftTelemetry);
        appendTelemetrySample("right", rightTelemetry);
    }
    renderForcePanel("left", leftRobotEntry, leftTelemetry);
    renderForcePanel("right", rightRobotEntry, rightTelemetry);
    renderTrendGraph("left", "force", state.telemetryHistory.left, leftTelemetry.force);
    renderTrendGraph("left", "moment", state.telemetryHistory.left, leftTelemetry.moment);
    renderTrendGraph("right", "force", state.telemetryHistory.right, rightTelemetry.force);
    renderTrendGraph("right", "moment", state.telemetryHistory.right, rightTelemetry.moment);

    const grid = byId("teleop-status-grid");
    grid.innerHTML = "";
    const ddkRobotCount = Object.keys(teleopStatus.ddk.robots || {}).length;
    const ddkHasIssues = Object.keys(teleopStatus.ddk.errors || {}).length > 0;
    const runningCameraCount = Object.values(cameras).filter((entry) => entry.started).length;
    const cameraHasIssues = Object.keys(teleopStatus.cameras.errors || {}).length > 0;
    [
        [
            "Teleoperation",
            teleopStatus.teleop.started ? "Running" : teleopStatus.teleop.initialized ? "Ready" : "Initializing",
            teleopStatus.teleop.started || teleopStatus.teleop.initialized ? "ok" : "neutral",
        ],
        ["Robot Data Streams", ddkRobotCount, ddkRobotCount ? "ok" : ddkHasIssues ? "error" : "neutral"],
        ["Camera Feeds", runningCameraCount, runningCameraCount ? "ok" : cameraHasIssues ? "error" : "neutral"],
    ].forEach(([label, value, tone]) => {
        grid.appendChild(createStatusCard(label, value, tone));
    });
    grid.insertBefore(createRecordingCard(teleopStatus.recording || {}), grid.children[1] || null);

    byId("record-save").classList.toggle("hidden", !teleopStatus.recording.awaiting_save);
    byId("record-discard").classList.toggle("hidden", !teleopStatus.recording.awaiting_save);

    const issues = [];
    if (teleopStatus.teleop.error) {
        issues.push(teleopStatus.teleop.error);
    }
    issues.push(...Object.values(teleopStatus.ddk.errors || {}));
    issues.push(...Object.values(teleopStatus.cameras.errors || {}));
    const message = byId("teleop-message");
    if (issues.length) {
        message.textContent = issues.join(" | ");
        message.classList.remove("panel--ok");
        message.classList.add("panel--issue");
        message.classList.remove("hidden");
    } else {
        message.classList.remove("panel--issue", "panel--ok");
        message.classList.add("hidden");
    }
}

async function refreshTeleopStatus() {
    state.teleopStatus = await api("/teleop/status");
    renderTeleop();
}

function renderTraining() {
    const container = byId("training-content");

    if (state.trainingStep === 1) {
        container.innerHTML = `
      <div class="panel-header">
        <div>
          <h2>Load Episode Datasets</h2>
        </div>
      </div>
      <div class="episode-list" id="load-episode-list"></div>
      <div class="control-bar">
        <button class="secondary-button" id="training-add-episode" type="button">Add Episode</button>
        <button id="training-next-step" type="button" ${state.episodes.length ? "" : "disabled"}>Next</button>
      </div>
    `;
        const list = byId("load-episode-list");
        if (!state.episodes.length) {
            list.innerHTML = `<div class="episode-row"><span>No episode datasets selected yet.</span></div>`;
        } else {
            state.episodes.forEach((episode, index) => {
                const row = document.createElement("div");
                row.className = "episode-row";
                row.innerHTML = `
          <div class="episode-row__main"><strong>${index + 1}</strong><span>${episode.name}</span></div>
          <button class="secondary-button" data-remove-episode="${episode.path}" type="button">Remove</button>
        `;
                list.appendChild(row);
            });
        }
        byId("training-add-episode").onclick = () => openEpisodeBrowser();
        byId("training-next-step").onclick = () => {
            state.trainingStep = 2;
            renderTraining();
        };
        list.querySelectorAll("[data-remove-episode]").forEach((button) => {
            button.onclick = () => {
                const path = button.dataset.removeEpisode;
                state.episodes = state.episodes.filter((item) => item.path !== path);
                state.selectedEpisodes = state.selectedEpisodes.filter((item) => item !== path);
                renderTraining();
            };
        });
        return;
    }

    if (state.trainingStep === 2) {
        container.innerHTML = `
      <div class="training-layout">
        <aside class="panel">
          <div class="panel-header">
            <h2>Episodes</h2>
            <button class="secondary-button" id="training-select-all" type="button">Select All</button>
          </div>
          <div class="episode-list" id="training-episode-picker"></div>
        </aside>
        <div class="training-main">
          <div class="panel panel--soft"><div class="feed-row" id="training-preview-feeds"></div></div>
          <div class="panel panel--soft"><h3>Dataset Fields</h3><div id="training-preview-legend"></div></div>
          <div class="control-bar">
            <button class="secondary-button" id="training-prev-step" type="button">Previous Step</button>
            <button id="training-combine" type="button" ${state.selectedEpisodes.length ? "" : "disabled"}>Combine Selected Episodes</button>
          </div>
        </div>
      </div>
    `;
        const picker = byId("training-episode-picker");
        state.episodes.forEach((episode, index) => {
            const row = document.createElement("button");
            row.className = "episode-row";
            row.type = "button";
            row.innerHTML = `
        <div class="episode-row__main">
          <input data-toggle-episode="${episode.path}" type="checkbox" ${state.selectedEpisodes.includes(episode.path) ? "checked" : ""} />
          <strong>${index + 1}</strong>
          <span>${episode.name}</span>
        </div>
      `;
            row.onclick = async (event) => {
                if (event.target instanceof HTMLInputElement) {
                    return;
                }
                state.preview = await api(`/datasets/preview?path=${encodeURIComponent(episode.path)}`);
                renderTraining();
            };
            picker.appendChild(row);
        });
        picker.querySelectorAll("[data-toggle-episode]").forEach((input) => {
            input.onchange = () => {
                const path = input.dataset.toggleEpisode;
                if (!path) {
                    return;
                }
                if (state.selectedEpisodes.includes(path)) {
                    state.selectedEpisodes = state.selectedEpisodes.filter((item) => item !== path);
                } else {
                    state.selectedEpisodes = [...state.selectedEpisodes, path];
                }
                renderTraining();
            };
        });
        const feeds = byId("training-preview-feeds");
        const legend = byId("training-preview-legend");
        if (!state.preview) {
            feeds.innerHTML = `<div class="feed__placeholder">Select an episode dataset to preview metadata.</div>`;
            legend.innerHTML = "";
        } else {
            feeds.innerHTML = (state.preview.camera_keys || ["No camera keys available"]).map((cameraKey) => `
        <div class="feed"><div class="feed__header"><span>${cameraKey}</span><strong>${state.preview.fps} FPS</strong></div><div class="feed__placeholder">Recorded media placeholder</div></div>
      `).join("");
            legend.innerHTML = (state.preview.numeric_keys || []).slice(0, 24).map((key) => `<span class="legend-chip">${key}</span>`).join("");
        }
        byId("training-select-all").onclick = () => {
            state.selectedEpisodes = state.selectedEpisodes.length === state.episodes.length ? [] : state.episodes.map((episode) => episode.path);
            renderTraining();
        };
        byId("training-prev-step").onclick = () => {
            state.trainingStep = 1;
            renderTraining();
        };
        byId("training-combine").onclick = async () => {
            try {
                state.trainingStep = 3;
                renderTraining();
                const result = await api("/datasets/combine", {
                    method: "POST",
                    body: JSON.stringify({ episode_paths: state.selectedEpisodes, output_name: `combined-${Date.now()}` }),
                });
                state.combinedPath = result.root;
                state.combinedPreview = await api(`/datasets/preview?path=${encodeURIComponent(result.root)}`);
                renderTraining();
            } catch (error) {
                showToast(error.message, true);
                state.trainingStep = 2;
                renderTraining();
            }
        };
        return;
    }

    if (state.trainingStep === 3) {
        container.innerHTML = `
            <div class="panel-header"><div><h2>Combined Dataset</h2></div></div>
      <div class="progress-block ${state.combinedPreview ? "hidden" : ""}"><div><div class="progress-bar"><span style="width: 72%"></span></div><p>Combining selected episodes...</p></div></div>
      <div class="${state.combinedPreview ? "" : "hidden"}" id="combined-preview-block">
        <div class="feed-row" id="combined-feed-row"></div>
        <div class="panel panel--soft"><h3>Combined Dataset Fields</h3><div id="combined-legend"></div></div>
      </div>
      <div class="control-bar"><button class="secondary-button" id="combine-prev" type="button">Previous Step</button><button id="combine-next" type="button" ${state.combinedPreview ? "" : "disabled"}>Next</button></div>
    `;
        if (state.combinedPreview) {
            byId("combined-feed-row").innerHTML = (state.combinedPreview.camera_keys || []).map((cameraKey) => `
        <div class="feed"><div class="feed__header"><span>${cameraKey}</span><strong>${state.combinedPreview.fps} FPS</strong></div><div class="feed__placeholder">Combined media placeholder</div></div>
      `).join("");
            byId("combined-legend").innerHTML = (state.combinedPreview.numeric_keys || []).slice(0, 24).map((key) => `<span class="legend-chip">${key}</span>`).join("");
        }
        byId("combine-prev").onclick = () => {
            state.trainingStep = 2;
            renderTraining();
        };
        byId("combine-next").onclick = () => {
            state.trainingStep = 4;
            renderTraining();
        };
        return;
    }

    if (state.trainingStep === 4) {
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
            state.trainingStep = 3;
            renderTraining();
        };
        byId("policy-start").onclick = async () => {
            try {
                state.trainingStep = 5;
                renderTraining();
                state.trainingStatus = await api("/training/start", {
                    method: "POST",
                    body: JSON.stringify({ dataset_path: state.combinedPath, output_dir: state.outputDir, policy_type: state.selectedPolicy }),
                });
                renderTraining();
                window.clearInterval(state.intervals.training);
                state.intervals.training = window.setInterval(async () => {
                    if (state.activeView !== "training" || state.trainingStep !== 5) {
                        return;
                    }
                    state.trainingStatus = await api("/training/status");
                    renderTraining();
                }, 2000);
            } catch (error) {
                showToast(error.message, true);
                state.trainingStep = 4;
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
        state.trainingStep = 4;
        renderTraining();
    };
}

async function refreshBrowser(path) {
    const result = await api(`/datasets/browse?path=${encodeURIComponent(path)}&directories_only=${state.pathBrowser.directoriesOnly}`);
    state.pathBrowser.currentPath = result.path;
    state.pathBrowser.selected = [];
    byId("browser-title").textContent = state.pathBrowser.title;
    byId("browser-path").textContent = result.path;
    const list = byId("browser-list");
    list.innerHTML = "";
    result.items.forEach((item) => {
        const button = document.createElement("button");
        button.className = "browser-item";
        button.type = "button";
        button.innerHTML = `<strong>${item.name}</strong><span>${item.is_dir ? "Directory" : "File"}</span>`;
        button.onclick = () => {
            if (!item.is_dir) {
                return;
            }
            if (state.pathBrowser.multiSelect) {
                if (state.pathBrowser.selected.includes(item.path)) {
                    state.pathBrowser.selected = state.pathBrowser.selected.filter((value) => value !== item.path);
                } else {
                    state.pathBrowser.selected = [...state.pathBrowser.selected, item.path];
                }
                button.classList.toggle("browser-item--selected", state.pathBrowser.selected.includes(item.path));
            } else {
                state.pathBrowser.selected = [item.path];
                document.querySelectorAll(".browser-item").forEach((element) => element.classList.remove("browser-item--selected"));
                button.classList.add("browser-item--selected");
            }
        };
        button.ondblclick = () => item.is_dir && refreshBrowser(item.path).catch((error) => showToast(error.message, true));
        list.appendChild(button);
    });
    byId("browser-modal").classList.remove("hidden");
}

async function openBrowser(options) {
    state.pathBrowser = { ...state.pathBrowser, ...options, selected: [] };
    await refreshBrowser(options.startPath);
}

function closeBrowser() {
    byId("browser-modal").classList.add("hidden");
}

function openEpisodeBrowser() {
    openBrowser({
        title: "Select Episode Datasets",
        startPath: state.summary.storage.episodes,
        directoriesOnly: true,
        multiSelect: true,
        onConfirm: (paths) => {
            paths.forEach((path) => {
                if (!state.episodes.some((item) => item.path === path)) {
                    state.episodes.push({ name: path.split("/").pop(), path });
                }
            });
            closeBrowser();
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
    if (state.teleopBootstrapped) {
        return;
    }
    setComponentLoading("teleop-cameras-wrap", true, "Initializing cameras");
    try {
        await api("/teleop/bootstrap", { method: "POST" });
        state.teleopBootstrapped = true;
    } finally {
        setComponentLoading("teleop-cameras-wrap", false);
    }
    await refreshTeleopStatus();
    window.clearInterval(state.intervals.teleop);
    state.intervals.teleop = window.setInterval(() => {
        if (state.activeView === "teleoperation") {
            refreshTeleopStatus().catch((error) => showToast(error.message, true));
        }
    }, 1500);
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
            if (target === "training") {
                bootstrapTraining().catch((error) => showToast(error.message, true));
            }
        });
    });

    byId("teleop-refresh").onclick = () => refreshTeleopStatus().catch((error) => showToast(error.message, true));
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
            const result = await api("/teleop/home", { method: "POST" });
            if (result.warnings?.length) {
                showToast(result.warnings.join(" | "), true);
            } else {
                showToast("Home reset command sent.");
            }
        } catch (error) {
            showToast(error.message, true);
        }
    };
    byId("record-start").onclick = async () => {
        try {
            await api("/teleop/recording/start", { method: "POST", body: JSON.stringify({ task: "Dual-arm Flexiv teleoperation demonstration" }) });
            await refreshTeleopStatus();
        } catch (error) {
            showToast(error.message, true);
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
        try {
            const result = await api("/teleop/recording/save", { method: "POST" });
            showToast(`Saved ${result.episode_name}`);
            await refreshTeleopStatus();
        } catch (error) {
            showToast(error.message, true);
        }
    };
    byId("record-discard").onclick = async () => {
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
    byId("browser-up").onclick = () => {
        const parts = state.pathBrowser.currentPath.split("/").filter(Boolean);
        const parent = parts.length ? `/${parts.slice(0, -1).join("/")}` || "/" : "/";
        refreshBrowser(parent).catch((error) => showToast(error.message, true));
    };
    byId("browser-confirm").onclick = () => {
        if (!state.pathBrowser.onConfirm) {
            return;
        }
        const selection = state.pathBrowser.selected.length
            ? state.pathBrowser.selected
            : state.pathBrowser.multiSelect
                ? []
                : [state.pathBrowser.currentPath];
        if (!selection.length) {
            return;
        }
        state.pathBrowser.onConfirm(selection);
    };
}

async function init() {
    bindGlobalEvents();
    await refreshSummary();
    renderTeleop();
    renderTraining();
    setActiveView("home");
}

init().catch((error) => showToast(error.message, true));
