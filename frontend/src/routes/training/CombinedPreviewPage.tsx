import type { DatasetPreview } from "../../types";

interface CombinedPreviewPageProps {
    loading: boolean;
    preview: DatasetPreview | null;
    onPrevious: () => void;
    onNext: () => void;
}

export default function CombinedPreviewPage({ loading, preview, onPrevious, onNext }: CombinedPreviewPageProps) {
    return (
        <section className="panel panel--wide">
            <h2>Combined Dataset</h2>
            {loading ? (
                <div className="progress-block">
                    <div className="progress-bar">
                        <span style={{ width: "65%" }} />
                    </div>
                    <p>Combining selected episodes...</p>
                </div>
            ) : (
                <>
                    <div className="feed-row">
                        {(preview?.camera_keys ?? []).map((camera) => (
                            <div className="feed" key={camera}>
                                <div className="feed__header">
                                    <span>{camera}</span>
                                    <strong>{preview?.fps ?? "--"} FPS</strong>
                                </div>
                                <div className="feed__placeholder">Combined dataset preview placeholder</div>
                            </div>
                        ))}
                    </div>
                    <div className="legend-grid">
                        {(preview?.numeric_keys ?? []).slice(0, 18).map((key) => (
                            <span className="legend-pill" key={key}>
                                {key}
                            </span>
                        ))}
                    </div>
                </>
            )}
            <div className="control-bar control-bar--end">
                <button onClick={onPrevious} type="button">
                    Previous Step
                </button>
                <button disabled={loading || !preview} onClick={onNext} type="button">
                    Next
                </button>
            </div>
        </section>
    );
}
