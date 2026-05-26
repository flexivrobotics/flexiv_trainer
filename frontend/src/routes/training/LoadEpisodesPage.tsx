import type { EpisodeItem } from "../../types";

interface LoadEpisodesPageProps {
    episodes: EpisodeItem[];
    onAdd: () => void;
    onRemove: (path: string) => void;
    onNext: () => void;
}

export default function LoadEpisodesPage({ episodes, onAdd, onRemove, onNext }: LoadEpisodesPageProps) {
    return (
        <section className="panel panel--wide">
            <h2>Load Episode Datasets</h2>
            <div className="episode-list">
                {episodes.map((episode, index) => (
                    <div className="episode-list__row" key={episode.path}>
                        <span>{index + 1}</span>
                        <strong>{episode.name}</strong>
                        <button onClick={() => onRemove(episode.path)} type="button">
                            Remove
                        </button>
                    </div>
                ))}
            </div>
            <div className="control-bar control-bar--end">
                <button onClick={onAdd} type="button">
                    Add Episode
                </button>
                <button disabled={!episodes.length} onClick={onNext} type="button">
                    Next
                </button>
            </div>
        </section>
    );
}
