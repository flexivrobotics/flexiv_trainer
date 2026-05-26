import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import ModuleLoadingOverlay from "../components/ModuleLoadingOverlay";
import ServerPathBrowserDialog from "../components/ServerPathBrowserDialog";
import { useModuleBootstrap } from "../hooks/useModuleBootstrap";
import type { DatasetPreview, EpisodeItem, PolicyCatalog, TrainingStatus } from "../types";
import ChoosePolicyPage from "./training/ChoosePolicyPage";
import CombinedPreviewPage from "./training/CombinedPreviewPage";
import LoadEpisodesPage from "./training/LoadEpisodesPage";
import PreviewEpisodesPage from "./training/PreviewEpisodesPage";
import TrainingRunPage from "./training/TrainingRunPage";

export default function TrainingPage() {
    const bootstrap = useModuleBootstrap(useCallback(() => api.bootstrapTraining(), []));
    const [step, setStep] = useState(0);
    const [episodes, setEpisodes] = useState<EpisodeItem[]>([]);
    const [selected, setSelected] = useState<string[]>([]);
    const [preview, setPreview] = useState<DatasetPreview | null>(null);
    const [combinedPreview, setCombinedPreview] = useState<DatasetPreview | null>(null);
    const [combineLoading, setCombineLoading] = useState(false);
    const [catalog, setCatalog] = useState<PolicyCatalog | null>(null);
    const [selectedPolicy, setSelectedPolicy] = useState("diffusion");
    const [outputDir, setOutputDir] = useState("");
    const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
    const [pickerMode, setPickerMode] = useState<"episodes" | "output" | null>(null);

    useEffect(() => {
        api.listPolicies().then((result) => {
            setCatalog(result);
            setSelectedPolicy(result.default);
        });
    }, []);

    useEffect(() => {
        if (step !== 4) {
            return;
        }
        const handle = window.setInterval(() => {
            api.getTrainingStatus().then(setTrainingStatus).catch(() => undefined);
        }, 2000);
        return () => window.clearInterval(handle);
    }, [step]);

    const selectedAll = useMemo(() => episodes.length > 0 && selected.length === episodes.length, [episodes, selected]);

    const addEpisodes = (paths: string[]) => {
        setEpisodes((current: EpisodeItem[]) => {
            const merged = [...current];
            for (const path of paths) {
                if (!merged.some((item) => item.path === path)) {
                    merged.push({ name: path.split("/").pop() || path, path });
                }
            }
            return merged;
        });
        setPickerMode(null);
    };

    const previewEpisode = (path: string) => {
        api.previewDataset(path).then(setPreview);
    };

    const combineSelected = async () => {
        setCombineLoading(true);
        const outputName = `combined-${Date.now()}`;
        const result = await api.combineEpisodes(selected, outputName);
        const dataset = await api.previewDataset((result as { root: string }).root);
        setCombinedPreview(dataset);
        setCombineLoading(false);
        setStep(2);
    };

    const startTraining = async () => {
        if (!combinedPreview) {
            return;
        }
        const status = await api.startTraining(combinedPreview.path, outputDir, selectedPolicy);
        setTrainingStatus(status);
        setStep(4);
    };

    return (
        <div className="page-shell module-shell">
            {step === 0 ? (
                <LoadEpisodesPage
                    episodes={episodes}
                    onAdd={() => setPickerMode("episodes")}
                    onNext={() => setStep(1)}
                    onRemove={(path) => setEpisodes((current: EpisodeItem[]) => current.filter((item: EpisodeItem) => item.path !== path))}
                />
            ) : null}

            {step === 1 ? (
                <PreviewEpisodesPage
                    checked={selected}
                    episodes={episodes}
                    preview={preview}
                    onCombine={combineSelected}
                    onPrevious={() => setStep(0)}
                    onSelect={previewEpisode}
                    onSelectAll={() => setSelected(selectedAll ? [] : episodes.map((episode: EpisodeItem) => episode.path))}
                    onToggle={(path) =>
                        setSelected((current: string[]) =>
                            current.includes(path) ? current.filter((item: string) => item !== path) : [...current, path],
                        )
                    }
                />
            ) : null}

            {step === 2 ? (
                <CombinedPreviewPage
                    loading={combineLoading}
                    onNext={() => setStep(3)}
                    onPrevious={() => setStep(1)}
                    preview={combinedPreview}
                />
            ) : null}

            {step === 3 ? (
                <ChoosePolicyPage
                    catalog={catalog}
                    onPrevious={() => setStep(2)}
                    onSelectOutputDir={() => setPickerMode("output")}
                    onSelectPolicy={setSelectedPolicy}
                    onStart={startTraining}
                    outputDir={outputDir}
                    selectedPolicy={selectedPolicy}
                />
            ) : null}

            {step === 4 ? <TrainingRunPage status={trainingStatus} /> : null}

            <ServerPathBrowserDialog
                directoriesOnly
                multiSelect={pickerMode === "episodes"}
                onClose={() => setPickerMode(null)}
                onConfirm={(paths) => {
                    if (pickerMode === "episodes") {
                        addEpisodes(paths);
                    } else {
                        setOutputDir(paths[0] ?? "");
                        setPickerMode(null);
                    }
                }}
                open={pickerMode !== null}
                title={pickerMode === "episodes" ? "Select Episode Datasets" : "Select Training Output Directory"}
            />

            <ModuleLoadingOverlay
                active={bootstrap.loading}
                error={bootstrap.error}
                stages={bootstrap.data?.stages ?? []}
                title="Preparing Training"
            />
        </div>
    );
}
