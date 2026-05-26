export interface ModuleStage {
    stage: string;
    progress: number;
    detail: unknown;
}

export interface ModuleBootstrap {
    ready: boolean;
    stages: ModuleStage[];
}

export interface EpisodeItem {
    name: string;
    path: string;
}

export interface DatasetPreview {
    name: string;
    path: string;
    repo_id: string;
    fps: number;
    num_frames: number;
    num_episodes: number;
    camera_keys: string[];
    numeric_keys: string[];
    sample_task: string | null;
}

export interface SystemSummary {
    backend: { reachable: boolean; host: string; port: number };
    calibration: { root: string; available_files: string[] };
    cameras: { available: boolean; devices: Array<{ name: string; serial: string }> };
    ddk: { available: boolean; configured: boolean; robots: Record<string, { connected: boolean; error?: string }> };
    storage: { root: string; episodes: string; combined: string; training: string };
    teleop: { configured: boolean; available: boolean; initialized: boolean; error?: string | null };
}

export interface PolicyCatalog {
    default: string;
    policies: Record<string, { label: string; description: string }>;
}

export interface TrainingStatus {
    job_id?: string;
    status: string;
    progress: number;
    logs: string[];
    error?: string | null;
    return_code?: number | null;
}
