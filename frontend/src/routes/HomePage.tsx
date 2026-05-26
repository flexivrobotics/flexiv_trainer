import { useEffect, useState } from "react";

import { api } from "../api/client";
import ExampleWorkflowCard from "../components/ExampleWorkflowCard";
import type { SystemSummary } from "../types";

export default function HomePage() {
    const [summary, setSummary] = useState<SystemSummary | null>(null);

    useEffect(() => {
        api.getSystemSummary().then(setSummary).catch(() => setSummary(null));
    }, []);

    return (
        <div className="page-shell home-grid">
            <section className="hero-panel">
                <p className="eyebrow">Home</p>
                <h1>Flexiv Trainer</h1>
                <p>
                    Start from the navigation bar when you are ready to enter Teleoperation or Training. This home page stays light on
                    startup and only checks the backend summary, storage roots, and basic hardware visibility.
                </p>
            </section>

            <section className="panel">
                <h2>Quick Start</h2>
                <ol className="number-list">
                    <li>Check robot, network, and camera readiness.</li>
                    <li>Open Teleoperation and verify live status before recording.</li>
                    <li>Record one or more episodes, then save them to local storage.</li>
                    <li>Open Training, load saved episodes, combine them, and start a policy run.</li>
                </ol>
            </section>

            <section className="panel">
                <h2>Safety &amp; Readiness Checklist</h2>
                <ul className="plain-list">
                    <li>All four robots powered and reachable on the expected network interfaces.</li>
                    <li>Two D405 wrist cameras and one D435 egocentric camera connected and discoverable.</li>
                    <li>Calibration files available for the egocentric overlay path.</li>
                    <li>Home postures configured before issuing reset commands.</li>
                </ul>
            </section>

            <section className="panel">
                <h2>Data Storage &amp; Outputs</h2>
                <dl className="kv-list">
                    <div>
                        <dt>Episodes</dt>
                        <dd>{summary?.storage.episodes ?? "Loading..."}</dd>
                    </div>
                    <div>
                        <dt>Combined datasets</dt>
                        <dd>{summary?.storage.combined ?? "Loading..."}</dd>
                    </div>
                    <div>
                        <dt>Training outputs</dt>
                        <dd>{summary?.storage.training ?? "Loading..."}</dd>
                    </div>
                </dl>
            </section>

            <section className="panel panel--wide">
                <div className="panel-header">
                    <h2>Example Workflows</h2>
                    <p>Open a module only when you intend to use it. Each module initializes lazily.</p>
                </div>
                <div className="workflow-grid">
                    <ExampleWorkflowCard
                        title="Record a New Episode"
                        description="Open Teleoperation, start teleop, then record and save a single episode dataset."
                        to="/teleoperation"
                    />
                    <ExampleWorkflowCard
                        title="Review or Combine Saved Episodes"
                        description="Open Training and load one or more saved episode datasets from backend storage."
                        to="/training"
                    />
                    <ExampleWorkflowCard
                        title="Train a Policy"
                        description="Combine selected episodes, pick a LeRobot policy, and launch training to a chosen output directory."
                        to="/training"
                    />
                </div>
            </section>

            <section className="panel panel--wide">
                <h2>System Status</h2>
                <div className="status-grid">
                    <div className="status-chip">
                        <span>Backend</span>
                        <strong>{summary?.backend.reachable ? "Reachable" : "Unknown"}</strong>
                    </div>
                    <div className="status-chip">
                        <span>Teleop Config</span>
                        <strong>{summary?.teleop.configured ? "Configured" : "Missing"}</strong>
                    </div>
                    <div className="status-chip">
                        <span>DDK</span>
                        <strong>{summary?.ddk.available ? "Available" : "Missing"}</strong>
                    </div>
                    <div className="status-chip">
                        <span>Cameras</span>
                        <strong>{summary?.cameras.devices.length ?? 0} discovered</strong>
                    </div>
                    <div className="status-chip">
                        <span>Calibration</span>
                        <strong>{summary?.calibration.available_files.length ? "Present" : "Missing"}</strong>
                    </div>
                </div>
            </section>
        </div>
    );
}
