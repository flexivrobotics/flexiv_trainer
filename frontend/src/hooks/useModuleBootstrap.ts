import { startTransition, useEffect, useState } from "react";

import type { ModuleBootstrap } from "../types";

export function useModuleBootstrap(loader: () => Promise<ModuleBootstrap>) {
    const [data, setData] = useState<ModuleBootstrap | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        let active = true;
        startTransition(() => {
            setLoading(true);
            loader()
                .then((result) => {
                    if (!active) {
                        return;
                    }
                    setData(result);
                    setError(null);
                    setLoading(false);
                })
                .catch((reason: unknown) => {
                    if (!active) {
                        return;
                    }
                    setError(reason instanceof Error ? reason.message : "Failed to bootstrap module");
                    setLoading(false);
                });
        });
        return () => {
            active = false;
        };
    }, [loader]);

    return { data, loading, error };
}
