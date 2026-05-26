import type {
    DatasetPreview,
    EpisodeItem,
    ModuleBootstrap,
    PolicyCatalog,
    SystemSummary,
    TrainingStatus,
} from "../types";

function resolveDefaultApiBase(): string {
    if (typeof window === "undefined") {
        return "http://127.0.0.1:8000";
    }

    return `${window.location.protocol}//${window.location.hostname}:8000`;
}

const API_BASE = (import.meta.env.VITE_API_BASE ?? resolveDefaultApiBase()).replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${API_BASE}${path}`, {
        headers: {
            "Content-Type": "application/json",
            ...(init?.headers ?? {}),
        },
        ...init,
    });
    if (!response.ok) {
        throw new Error(`Request failed: ${response.status} ${response.statusText}`);
    }
    return (await response.json()) as T;
}

export const api = {
    getSystemSummary: () => request<SystemSummary>("/system/summary"),
    bootstrapTeleop: () => request<ModuleBootstrap>("/teleop/bootstrap", { method: "POST" }),
    getTeleopStatus: () => request("/teleop/status"),
    startTeleop: () => request("/teleop/start", { method: "POST" }),
    stopTeleop: () => request("/teleop/stop", { method: "POST" }),
    resetHome: () => request("/teleop/home", { method: "POST" }),
    startRecording: (task: string, fps?: number) =>
        request("/teleop/recording/start", {
            method: "POST",
            body: JSON.stringify({ task, fps }),
        }),
    stopRecording: () => request("/teleop/recording/stop", { method: "POST" }),
    saveRecording: () => request("/teleop/recording/save", { method: "POST" }),
    discardRecording: () => request("/teleop/recording/discard", { method: "POST" }),
    bootstrapTraining: () => request<ModuleBootstrap>("/training/bootstrap", { method: "POST" }),
    listPolicies: () => request<PolicyCatalog>("/training/policies"),
    getTrainingStatus: () => request<TrainingStatus>("/training/status"),
    startTraining: (datasetPath: string, outputDir: string, policyType: string) =>
        request<TrainingStatus>("/training/start", {
            method: "POST",
            body: JSON.stringify({ dataset_path: datasetPath, output_dir: outputDir, policy_type: policyType }),
        }),
    listEpisodes: async () => {
        const result = await request<{ episodes: EpisodeItem[] }>("/datasets/episodes");
        return result.episodes;
    },
    previewDataset: (path: string) => request<DatasetPreview>(`/datasets/preview?path=${encodeURIComponent(path)}`),
    browseServerPath: (path?: string, directoriesOnly = false) => {
        const params = new URLSearchParams();
        if (path) {
            params.set("path", path);
        }
        params.set("directories_only", String(directoriesOnly));
        return request<{ path: string; items: Array<{ name: string; path: string; is_dir: boolean }> }>(
            `/datasets/browse?${params.toString()}`,
        );
    },
    combineEpisodes: (episodePaths: string[], outputName: string) =>
        request("/datasets/combine", {
            method: "POST",
            body: JSON.stringify({ episode_paths: episodePaths, output_name: outputName }),
        }),
};
