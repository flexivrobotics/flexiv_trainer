import { useCallback, useEffect, useState } from "react";

import { api } from "../api/client";
import ModuleLoadingOverlay from "../components/ModuleLoadingOverlay";
import { useModuleBootstrap } from "../hooks/useModuleBootstrap";

type TeleopStatus = Awaited<ReturnType<typeof api.getTeleopStatus>>;

export default function TeleoperationPage() {
    const bootstrap = useModuleBootstrap(useCallback(() => api.bootstrapTeleop(), []));
    const [status, setStatus] = useState<TeleopStatus | null>(null);

    useEffect(() => {
        if (bootstrap.loading || !bootstrap.data) {
            return;
        }
        api.getTeleopStatus().then(setStatus);
        const handle = window.setInterval(() => {
            api.getTeleopStatus().then(setStatus).catch(() => undefined);
        }, 1500);
        return () => window.clearInterval(handle);
    }, [bootstrap.loading, bootstrap.data]);

    const recordStatus = status?.recording;
    const cameraEntries = Object.entries(status?.cameras.cameras ?? {});

    return (
        <div className="page-shell module-shell">
            <div className="panel panel--wide teleop-layout teleop-layout--blurred">
                <div className="feed feed--hero">
                    <div className="feed__header">
                        <span>Ego Camera</span>
                        <strong>{status?.cameras.cameras?.ego?.fps?.toFixed?.(1) ?? "--"} FPS</strong>
                    </div>
                    <div className="feed__placeholder">Force overlay canvas and egocentric stream will render here.</div>
                </div>
                <div className="feed-row">
                    {cameraEntries
                        .filter(([name]) => name !== "ego")
                        .map(([name, camera]) => (
                            <div className="feed" key={name}>
                                <div className="feed__header">
                                    <span>{name}</span>
                                    <strong>{camera.fps?.toFixed?.(1) ?? "--"} FPS</strong>
                                </div>
                                <div className="feed__placeholder">In-hand camera stream placeholder</div>
                            </div>
                        ))}
                </div>
                <div className="control-bar">
                    <button onClick={() => api.startTeleop().then(() => api.getTeleopStatus().then(setStatus))} type="button">
                        Start Teleoperation
                    </button>
                    <button onClick={() => api.stopTeleop().then(() => api.getTeleopStatus().then(setStatus))} type="button">
                        Stop Teleoperation
                    </button>
                    <button onClick={() => api.resetHome().then(() => api.getTeleopStatus().then(setStatus))} type="button">
                        Reset All Robots to Home
                    </button>
                    <button
                        onClick={() => api.startRecording("Dual-arm Flexiv teleoperation demonstration").then(() => api.getTeleopStatus().then(setStatus))}
                        type="button"
                    >
                        Start Recording
                    </button>
                    <button onClick={() => api.stopRecording().then(() => api.getTeleopStatus().then(setStatus))} type="button">
                        Stop Recording
                    </button>
                    {recordStatus?.awaiting_save ? (
                        <>
                            <button onClick={() => api.saveRecording().then(() => api.getTeleopStatus().then(setStatus))} type="button">
                                Save Episode
                            </button>
                            <button onClick={() => api.discardRecording().then(() => api.getTeleopStatus().then(setStatus))} type="button">
                                Discard Episode
                            </button>
                        </>
                    ) : null}
                </div>
            </div>
            <ModuleLoadingOverlay
                active={bootstrap.loading}
                error={bootstrap.error}
                stages={bootstrap.data?.stages ?? []}
                title="Preparing Teleoperation"
            />
        </div>
    );
}
