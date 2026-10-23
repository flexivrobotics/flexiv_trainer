# Copyright 2026 Flexiv Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Train a learned check from the web UI: one job at a time, in its own process.

The job runs `python -m flexivtrainer.evaluation.learned_check train`, the same
command the README gives, so a check trained from the page and one trained from
a terminal are the same check. Its own process keeps the server responsive and
frees the GPU memory when it ends. The command logs its progress on stderr and
prints its result, the held-out report, as JSON on stdout.

A trained check is a folder under `<storage>/checks/`, which session plans,
chains and practice name as `checks/<name>`.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = ["CHECKS_DIR", "CheckTraining", "list_checks"]

CHECKS_DIR = "checks"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CAMERA = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_LOG_LINES = 200


def list_checks(storage_root: Path) -> list[dict[str, Any]]:
    """Every trained check under `<storage>/checks/`, with what it looks at and
    its held-out report, newest first."""
    directory = Path(storage_root) / CHECKS_DIR
    if not directory.is_dir():
        return []
    checks = []
    for path in sorted(directory.glob("*/check.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            checks.append({"name": path.parent.name, "error": str(exc)})
            continue
        trained = data.get("trained") or {}
        inputs = data.get("inputs") or {}
        checks.append(
            {
                "name": path.parent.name,
                "model": f"{CHECKS_DIR}/{path.parent.name}",
                "check": data.get("check"),
                "at": trained.get("at"),
                "cameras": inputs.get("cameras") or [],
                "state": inputs.get("state") or [],
                "threshold": data.get("threshold"),
                "held_out": trained.get("held_out"),
            }
        )
    checks.sort(key=lambda entry: entry.get("at") or "", reverse=True)
    return checks


class CheckTraining:
    """The one check being trained, or the last one."""

    def __init__(self, storage_root: Path, *, python: str = sys.executable) -> None:
        self._root = Path(storage_root)
        self._python = python
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._state = "idle"
        self._job: dict[str, Any] = {}
        self._log: deque[str] = deque(maxlen=_LOG_LINES)
        self._result: dict[str, Any] | None = None
        self._error: str | None = None
        self._started = 0.0
        self._ended: float | None = None

    def command(self, request: dict[str, Any], data: Sequence[Path]) -> list[str]:
        """The training command for `request`, its datasets already resolved."""
        name = request["name"]
        if not _NAME.match(name):
            raise ValueError("a check's name is 1-64 letters, digits, '.', '_' or '-'")
        if not _NAME.match(request["check"]):
            raise ValueError("a label name is 1-64 letters, digits, '.', '_' or '-'")
        if not data:
            raise ValueError("name at least one labelled dataset")
        cameras = list(request.get("cameras") or [])
        state = [s for s in request.get("state") or [] if s]
        if not cameras and not state:
            raise ValueError("a check looks at a camera, a state column, or both")
        command = [
            self._python, "-m", "flexivtrainer.evaluation.learned_check", "train",
            "--check", request["check"],
            "--out", str(self._root / CHECKS_DIR / name),
        ]  # fmt: skip
        for path in data:
            command += ["--data", str(path)]
        for camera in cameras:
            if not _CAMERA.match(camera):
                raise ValueError(f"{camera!r} is not a camera name")
            command += ["--camera", camera]
        for camera, box in (request.get("crops") or {}).items():
            if camera not in cameras:
                raise ValueError(
                    f"a crop for {camera!r}, which the check does not look at"
                )
            values = [float(v) for v in box]
            if len(values) != 4 or not (
                0 <= values[0] < values[2] <= 1 and 0 <= values[1] < values[3] <= 1
            ):
                raise ValueError(
                    f"{camera}'s crop is x0,y0,x1,y1, fractions of the frame, each "
                    "low below its high"
                )
            command += ["--crop", f"{camera}=" + ",".join(f"{v:g}" for v in values)]
        for column in state:
            command += ["--state", column]
        encoder = request.get("encoder")
        if encoder:
            command += ["--encoder", encoder]
        if request.get("epochs"):
            command += ["--epochs", str(int(request["epochs"]))]
        return command

    def start(self, request: dict[str, Any], data: Sequence[Path]) -> dict[str, Any]:
        with self._lock:
            if self._state == "running":
                raise RuntimeError("A check is already training")
            out = self._root / CHECKS_DIR / request.get("name", "")
            command = self.command(request, data)
            if out.exists():
                raise FileExistsError(f"a check named {request['name']!r} exists")
            out.parent.mkdir(parents=True, exist_ok=True)
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            self._state = "running"
            self._job = {
                "name": request["name"],
                "check": request["check"],
                "model": f"{CHECKS_DIR}/{request['name']}",
                "data": [str(path) for path in data],
                "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            self._log.clear()
            self._result = None
            self._error = None
            self._started = time.monotonic()
            self._ended = None
            process = self._process
        threading.Thread(target=self._read_log, args=(process,), daemon=True).start()
        threading.Thread(target=self._wait, args=(process,), daemon=True).start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if self._state != "running" or process is None:
                raise RuntimeError("No check is training")
            self._state = "stopping"
        process.terminate()
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            end = self._ended if self._ended is not None else time.monotonic()
            return {
                "state": self._state,
                **self._job,
                "seconds": round(end - self._started, 1) if self._job else 0.0,
                "log": list(self._log),
                "result": self._result,
                "error": self._error,
            }

    def _read_log(self, process: subprocess.Popen) -> None:
        assert process.stderr is not None
        for line in process.stderr:
            line = line.rstrip()
            if line:
                with self._lock:
                    self._log.append(line)

    def _wait(self, process: subprocess.Popen) -> None:
        assert process.stdout is not None
        output = process.stdout.read()
        code = process.wait()
        with self._lock:
            self._ended = time.monotonic()
            if self._state == "stopping":
                self._state = "stopped"
                return
            try:
                self._result = json.loads(output) if code == 0 else None
            except ValueError:
                self._result = None
            if code == 0 and self._result is not None:
                self._state = "done"
                return
            self._state = "failed"
            tail = [line for line in self._log if line.startswith("error:")]
            self._error = (
                tail[-1][len("error:") :].strip()
                if tail
                else (self._log[-1] if self._log else f"exited with code {code}")
            )
