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
    ui: {
        teleopRefreshBusy: false,
        teleopBootstrapBusy: false,
        teleopHomeBusy: false,
        recordingStartBusy: false,
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

function byId(id) {
    return document.getElementById(id);
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
        <article class="notification-item notification-item--${item.level}">
            <div class="notification-item__row">
                <span class="notification-item__pill">${item.level.toUpperCase()}</span>
                <time class="notification-item__time">${formatNotificationTimestamp(item.timestamp)}</time>
            </div>
            <p class="notification-item__message">${item.message}</p>
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

function createTeleopSystemCard(serviceKey, service = {}) {
    const ready = service.tone === "ok";
    const resetBusy = !!state.ui.serviceResetBusy[serviceKey];
    const serviceState = formatValue(service.state);
    const reconnectMessage = SERVICE_RESET_MESSAGES[serviceKey] || `Reconnect ${service.label || serviceKey}`;
    const card = document.createElement("div");
    card.className = "status-card teleop-system-card";
    card.innerHTML = `
        <div class="teleop-system-card__header">
            <div class="teleop-system-card__title">
                <span class="teleop-system-card__dot teleop-system-card__dot--${ready ? "ready" : "error"}" role="img" aria-label="${serviceState}" title="${serviceState}"></span>
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

function renderCameraFps(elementId, camera) {
    const element = byId(elementId);
    if (!element) {
        return;
    }

    const fps = Number(camera?.fps || 0);
    const hasData = !!camera?.started && fps > 0;
    setFpsBadge(element, fps, 29);

    const placeholder = element.closest(".feed")?.querySelector(".feed__placeholder");
    if (!placeholder) {
        return;
    }

    if (!placeholder.dataset.defaultContent) {
        placeholder.dataset.defaultContent = placeholder.innerHTML;
    }

    placeholder.classList.toggle("feed__placeholder--awaiting", !hasData);
    const renderMode = hasData ? "default" : "awaiting";
    if (placeholder.dataset.renderMode !== renderMode) {
        placeholder.innerHTML = hasData
            ? placeholder.dataset.defaultContent
            : buildAwaitingDataMarkup();
        placeholder.dataset.renderMode = renderMode;
    }
}

function renderRecordingOptions(recording = {}) {
    const container = byId("recording-options");
    if (!container) {
        return;
    }

    const locked = !!recording.active || !!recording.awaiting_save || state.ui.recordingStartBusy;
    container.innerHTML = "";
    RECORDING_ENTRY_OPTIONS.forEach((option) => {
        const checked = state.recordingEntries.includes(option.id);
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
            if (input.checked) {
                state.recordingEntries = [...state.recordingEntries, option.id];
            } else {
                state.recordingEntries = state.recordingEntries.filter((entry) => entry !== option.id);
            }
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
            return !!camera?.started && Number(camera?.fps || 0) > 0;
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
    if (kind === "ready" || kind === "awaiting-save") {
        return `<span class="recording-status__icon recording-status__icon--ready" aria-hidden="true">${CHECK_ICON_SVG}</span>`;
    }
    if (kind === "recording") {
        return `
            <span class="recording-status__icon recording-status__icon--recording" aria-hidden="true">
                <span class="recording-live__dot recording-status__dot"></span>
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
    const seconds = fps > 0 ? frames / fps : 0;

    if (active) {
        return {
            kind: "recording",
            text: `${formatElapsed(seconds)} · ${frames} frames captured`,
            animated: false,
            canStart: false,
            canStop: true,
        };
    }

    if (awaitingSave) {
        return {
            kind: "awaiting-save",
            text: `Awaiting save · ${formatElapsed(seconds)} · ${frames} frames captured`,
            animated: false,
            canStart: false,
            canStop: false,
        };
    }

    if (state.ui.recordingStartBusy || state.ui.teleopBootstrapBusy || !state.teleopBootstrapped || !teleopStatus) {
        return {
            kind: "initializing",
            text: "Initializing recorder",
            animated: true,
            canStart: false,
            canStop: false,
        };
    }

    if (state.recordingEntries.length > 0 && areSelectedRecordingEntriesAvailable(teleopStatus)) {
        return {
            kind: "ready",
            text: "Ready to record",
            animated: false,
            canStart: true,
            canStop: false,
        };
    }

    return {
        kind: "awaiting-data",
        text: "Awaiting data",
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
    setMarkupIfChanged(
        status,
        `${model.kind}:${model.text}:${model.animated ? "animated" : "static"}`,
        `
            ${recordingStatusIconMarkup(model.kind)}
            <span class="recording-status__text">${model.text}</span>
        `,
    );

    byId("record-start").disabled = !model.canStart;
    byId("record-stop").disabled = !model.canStop;
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
    renderTeleop();
}

async function controlHomeService(serviceName, action, options = {}) {
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
    if (!options.silentToast) {
        showToast(`${labels[serviceName] || serviceName} ${action === "connect" ? "connected" : "disconnected"}.`);
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
    updateTeleopControlButtons(teleopStatus);

    byId("record-save").classList.toggle("hidden", !teleopStatus.recording.awaiting_save);
    byId("record-discard").classList.toggle("hidden", !teleopStatus.recording.awaiting_save);

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
    setTeleopBootstrapBusy(true);
    setComponentLoading("teleop-cameras-wrap", true, "Initializing cameras");
    try {
        await api("/teleop/bootstrap", { method: "POST" });
        state.teleopBootstrapped = true;
        await refreshTeleopStatus();
    } finally {
        setComponentLoading("teleop-cameras-wrap", false);
        setTeleopBootstrapBusy(false);
    }
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
    renderNotificationCenter();
    await refreshSummary();
    renderTeleop();
    renderTraining();
    setActiveView("home");
}

init().catch((error) => showToast(error.message, true));
