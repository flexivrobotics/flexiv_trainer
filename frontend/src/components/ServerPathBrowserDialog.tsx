import { useEffect, useState } from "react";

import { api } from "../api/client";

interface ServerPathBrowserDialogProps {
    open: boolean;
    title: string;
    initialPath?: string;
    directoriesOnly?: boolean;
    multiSelect?: boolean;
    onClose: () => void;
    onConfirm: (paths: string[]) => void;
}

interface BrowserItem {
    name: string;
    path: string;
    is_dir: boolean;
}

export default function ServerPathBrowserDialog({
    open,
    title,
    initialPath,
    directoriesOnly = true,
    multiSelect = false,
    onClose,
    onConfirm,
}: ServerPathBrowserDialogProps) {
    const [currentPath, setCurrentPath] = useState(initialPath ?? "");
    const [items, setItems] = useState<BrowserItem[]>([]);
    const [selected, setSelected] = useState<string[]>([]);

    useEffect(() => {
        if (!open) {
            return;
        }
        api.browseServerPath(initialPath, directoriesOnly).then((result) => {
            setCurrentPath(result.path);
            setItems(result.items);
            setSelected([]);
        });
    }, [open, initialPath, directoriesOnly]);

    if (!open) {
        return null;
    }

    const navigate = (path?: string) => {
        api.browseServerPath(path, directoriesOnly).then((result) => {
            setCurrentPath(result.path);
            setItems(result.items);
            setSelected([]);
        });
    };

    const toggle = (path: string) => {
        setSelected((previous: string[]) => {
            if (!multiSelect) {
                return [path];
            }
            return previous.includes(path) ? previous.filter((item: string) => item !== path) : [...previous, path];
        });
    };

    const parentPath = currentPath.includes("/") ? currentPath.split("/").slice(0, -1).join("/") || "/" : undefined;

    return (
        <div className="dialog-backdrop">
            <div className="dialog-panel">
                <div className="dialog-header">
                    <div>
                        <p className="eyebrow">Server Path Browser</p>
                        <h3>{title}</h3>
                    </div>
                    <button onClick={onClose} type="button">
                        Close
                    </button>
                </div>
                <div className="dialog-toolbar">
                    <button onClick={() => navigate(parentPath)} type="button">
                        Up
                    </button>
                    <code>{currentPath}</code>
                </div>
                <div className="dialog-list">
                    {items.map((item: BrowserItem) => (
                        <button
                            className={`dialog-item ${selected.includes(item.path) ? "dialog-item--selected" : ""}`}
                            key={item.path}
                            onClick={() => (item.is_dir ? toggle(item.path) : undefined)}
                            onDoubleClick={() => (item.is_dir ? navigate(item.path) : undefined)}
                            type="button"
                        >
                            <span>{item.is_dir ? "DIR" : "FILE"}</span>
                            <strong>{item.name}</strong>
                        </button>
                    ))}
                </div>
                <div className="dialog-footer">
                    <button onClick={onClose} type="button">
                        Cancel
                    </button>
                    <button disabled={!selected.length} onClick={() => onConfirm(selected)} type="button">
                        Select
                    </button>
                </div>
            </div>
        </div>
    );
}
