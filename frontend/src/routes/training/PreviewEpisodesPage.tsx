import type { DatasetPreview, EpisodeItem } from "../../types";

interface PreviewEpisodesPageProps {
    episodes: EpisodeItem[];
    checked: string[];
    preview: DatasetPreview | null;
    onSelect: (path: string) => void;
    onToggle: (path: string) => void;
    onSelectAll: () => void;
    onPrevious: () => void;
    onCombine: () => void;
}

export default function PreviewEpisodesPage({
    episodes,
    checked,
    preview,
    onSelect,
    onToggle,
    onSelectAll,
    onPrevious,
    onCombine,
}: PreviewEpisodesPageProps) {
    return (
        <section className="training-preview-grid">
            <aside className="panel training-sidebar">
                <div className="panel-header">
                    <h2>Episodes</h2>
                    <button onClick={onSelectAll} type="button">
                        Select All
                    </button>
                </div>
                {episodes.map((episode, index) => (
                    <button className="episode-picker" key={episode.path} onClick={() => onSelect(episode.path)} type="button">
                        <input checked={checked.includes(episode.path)} onChange={() => onToggle(episode.path)} type="checkbox" />
                        <span>{index + 1}</span>
                        <strong>{episode.name}</strong>
                    </button>
                ))}
            </aside>
            <div className="training-main">
                <div className="panel training-main__top">
                    <div className="feed-row">
                        {(preview?.camera_keys ?? ["ego", "left_wrist", "right_wrist"]).map((camera) => (
                            <div className="feed" key={camera}>
                                <div className="feed__header">
                                    <span>{camera}</span>
                                    <strong>{preview?.fps ?? "--"} FPS</strong>
                                </div>
                                <div className="feed__placeholder">Dataset camera preview placeholder</div>
                            </div>
                        ))}
                    </div>
                </div>
                <div className="panel training-main__bottom">
                    <h3>Preview Graph</h3>
                    <p>Graph-ready legends</p>
                    <div className="legend-grid">
                        {(preview?.numeric_keys ?? []).slice(0, 18).map((key) => (
                            <span className="legend-pill" key={key}>
                                {key}
                            </span>
                        ))}
                    </div>
                </div>
                <div className="control-bar control-bar--end">
                    <button onClick={onPrevious} type="button">
                        Previous Step
                    </button>
                    <button disabled={!checked.length} onClick={onCombine} type="button">
                        Combine Selected Episodes
                    </button>
                </div>
            </div>
        </section>
    );
}
