import type { PolicyCatalog } from "../../types";

interface ChoosePolicyPageProps {
    catalog: PolicyCatalog | null;
    selectedPolicy: string;
    outputDir: string;
    onSelectPolicy: (policy: string) => void;
    onSelectOutputDir: () => void;
    onPrevious: () => void;
    onStart: () => void;
}

export default function ChoosePolicyPage({
    catalog,
    selectedPolicy,
    outputDir,
    onSelectPolicy,
    onSelectOutputDir,
    onPrevious,
    onStart,
}: ChoosePolicyPageProps) {
    return (
        <section className="panel panel--wide">
            <h2>Choose Training Policy</h2>
            <div className="policy-grid">
                {Object.entries(catalog?.policies ?? {}).map(([key, policy]) => (
                    <button
                        className={`policy-card ${selectedPolicy === key ? "policy-card--selected" : ""}`}
                        key={key}
                        onClick={() => onSelectPolicy(key)}
                        type="button"
                    >
                        <h3>{policy.label}</h3>
                        <p>{policy.description}</p>
                    </button>
                ))}
            </div>
            <div className="output-picker">
                <div>
                    <p className="eyebrow">Output Directory</p>
                    <strong>{outputDir || "No directory selected"}</strong>
                </div>
                <button onClick={onSelectOutputDir} type="button">
                    Choose Directory
                </button>
            </div>
            <div className="control-bar control-bar--end">
                <button onClick={onPrevious} type="button">
                    Previous Step
                </button>
                <button disabled={!outputDir} onClick={onStart} type="button">
                    Start Training
                </button>
            </div>
        </section>
    );
}
