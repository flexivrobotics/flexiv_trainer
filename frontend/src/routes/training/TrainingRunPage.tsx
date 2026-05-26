import type { TrainingStatus } from "../../types";

interface TrainingRunPageProps {
    status: TrainingStatus | null;
}

export default function TrainingRunPage({ status }: TrainingRunPageProps) {
    const progress = status?.progress ?? 0;
    const isSuccess = status?.status === "completed";
    const isError = status?.status === "failed";

    return (
        <section className="panel panel--wide">
            <h2>Training Run</h2>
            <div className="progress-bar progress-bar--large">
                <span style={{ width: `${progress}%` }} />
            </div>
            <div className={`result-pill ${isSuccess ? "result-pill--success" : isError ? "result-pill--error" : ""}`}>
                {isSuccess ? "Training completed" : isError ? status?.error ?? "Training failed" : status?.status ?? "Waiting"}
            </div>
            <pre className="log-pane">{(status?.logs ?? []).join("\n") || "Backend logs will appear here."}</pre>
        </section>
    );
}
