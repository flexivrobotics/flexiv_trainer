import type { ModuleStage } from "../types";

interface ModuleLoadingOverlayProps {
    title: string;
    active: boolean;
    stages: ModuleStage[];
    error?: string | null;
}

export default function ModuleLoadingOverlay({ title, active, stages, error }: ModuleLoadingOverlayProps) {
    if (!active) {
        return null;
    }

    const progress = stages.length ? stages[stages.length - 1].progress : 12;
    return (
        <div className="module-overlay">
            <div className="module-overlay__panel">
                <p className="eyebrow">Loading Module</p>
                <h2>{title}</h2>
                <div className="progress-bar">
                    <span style={{ width: `${progress}%` }} />
                </div>
                <ul className="stage-list">
                    {stages.map((stage) => (
                        <li key={stage.stage}>
                            <strong>{stage.stage}</strong>
                            <span>{stage.progress}%</span>
                        </li>
                    ))}
                </ul>
                {error ? <p className="status-error">{error}</p> : null}
            </div>
        </div>
    );
}
