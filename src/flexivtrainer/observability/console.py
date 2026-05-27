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

from __future__ import annotations

import logging
import os
import shlex
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, TextIO

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
except ImportError:  # pragma: no cover - optional dependency
    Console = None
    Panel = None
    Rule = None
    Text = None


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

LEVEL_STYLES = {
    "INFO": "\033[38;5;81m",
    "WARN": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "OK": "\033[38;5;114m",
}

RICH_LEVEL_STYLES = {
    "INFO": "bold black on bright_cyan",
    "WARN": "bold black on bright_yellow",
    "ERROR": "bold white on red",
    "OK": "bold black on bright_green",
}

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
OUTPUT_LOCK = threading.RLock()
DEPENDENCY_LOG_BRIDGE_LOCK = threading.RLock()
DEPENDENCY_LOG_HANDLER: logging.Handler | None = None

DetailFactory = Callable[[], str | None]


def _supports_color(stream: TextIO) -> bool:
    return (
        hasattr(stream, "isatty")
        and stream.isatty()
        and os.getenv("TERM") not in {None, "dumb"}
    )


def _console(stream: TextIO) -> Console | None:
    if Console is None:
        return None
    color = _supports_color(stream)
    return Console(
        file=stream,
        force_terminal=color,
        no_color=not color,
        soft_wrap=True,
        highlight=False,
    )


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _plain_style(text: str, color: str, stream: TextIO) -> str:
    if not _supports_color(stream):
        return text
    return f"{color}{text}{RESET}"


def _plain_line(
    level: str, message: str, detail: str | None = None, *, stream: TextIO
) -> str:
    prefix = f"[{_stamp()}] [{level}]"
    prefix = _plain_style(prefix, LEVEL_STYLES[level], stream)
    if _supports_color(stream):
        message = f"{BOLD}{message}{RESET}"
        if detail:
            detail = f"{DIM}{detail}{RESET}"
    line = f"{prefix} {message}"
    if detail:
        line = f"{line} {detail}"
    return line


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, remaining = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining:02d}"
    return f"{minutes:02d}:{remaining:02d}"


def describe_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    exception_name = type(exc).__name__
    if not message:
        return exception_name
    return f"{exception_name}: {message}"


def _record_detail(record: logging.LogRecord) -> str:
    parts = [f"logger={record.name}"]
    if record.pathname:
        parts.append(f"source={os.path.basename(record.pathname)}:{record.lineno}")
    if record.exc_info and record.exc_info[1] is not None:
        parts.append(f"cause={describe_exception(record.exc_info[1])}")
    return " ".join(parts)


class _DependencyLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage().strip()
            if not message:
                return

            if record.levelno >= logging.ERROR:
                level = "ERROR"
            elif record.levelno >= logging.WARNING:
                level = "WARN"
            else:
                level = "INFO"

            emit(level, message, _record_detail(record))
        except Exception:
            self.handleError(record)


def install_dependency_log_bridge(level: int = logging.WARNING) -> None:
    global DEPENDENCY_LOG_HANDLER

    with DEPENDENCY_LOG_BRIDGE_LOCK:
        if DEPENDENCY_LOG_HANDLER is not None:
            return

        handler = _DependencyLogHandler()
        handler.setLevel(level)
        logging.getLogger().addHandler(handler)
        logging.captureWarnings(True)
        DEPENDENCY_LOG_HANDLER = handler


def emit(
    level: str,
    message: str,
    detail: str | None = None,
    *,
    stream: TextIO | None = None,
) -> None:
    target = stream or (sys.stderr if level in {"WARN", "ERROR"} else sys.stdout)
    with OUTPUT_LOCK:
        console = _console(target)
        if console is None or Text is None:
            print(
                _plain_line(level, message, detail, stream=target),
                file=target,
                flush=True,
            )
            return

        line = Text(f"[{_stamp()}]", style="dim")
        line.append(" ")
        line.append(f" {level} ", style=RICH_LEVEL_STYLES[level])
        line.append(" ")
        line.append(message, style="bold")
        if detail:
            line.append(" ")
            line.append(detail, style="dim")
        console.print(line)


def info(message: str, detail: str | None = None) -> None:
    emit("INFO", message, detail)


def warn(message: str, detail: str | None = None) -> None:
    emit("WARN", message, detail)


def error(message: str, detail: str | None = None) -> None:
    emit("ERROR", message, detail)


def ok(message: str, detail: str | None = None) -> None:
    emit("OK", message, detail)


def banner(title: str, *lines: str) -> None:
    with OUTPUT_LOCK:
        console = _console(sys.stdout)
        if console is None or Panel is None:
            width = max(len(title), *(len(line) for line in lines), 24)
            border = "+" + "-" * (width + 2) + "+"
            print(border, flush=True)
            print(f"| {title.ljust(width)} |", flush=True)
            print(border, flush=True)
            for line in lines:
                print(f"| {line.ljust(width)} |", flush=True)
            print(border, flush=True)
            return

        body = "\n".join(lines) if lines else title
        console.print(
            Panel.fit(
                body,
                title=f"[bold]{title}[/bold]",
                border_style="bright_cyan",
                padding=(1, 2),
            )
        )


def section(
    title: str, subtitle: str | None = None, *, style: str = "bright_cyan"
) -> None:
    with OUTPUT_LOCK:
        console = _console(sys.stdout)
        if console is None or Rule is None or Text is None:
            print(f"\n== {title} ==", flush=True)
            if subtitle:
                print(subtitle, flush=True)
            return

        console.print(Rule(Text(title, style=f"bold {style}"), style=style))
        if subtitle:
            console.print(Text(subtitle, style="dim"))


def print_command(title: str, argv: list[str]) -> None:
    command_line = shlex.join(argv)
    with OUTPUT_LOCK:
        console = _console(sys.stdout)
        if console is None or Panel is None:
            print(f"{title}: {command_line}", flush=True)
            return

        console.print(
            Panel.fit(
                command_line,
                title=f"[bold]{title}[/bold]",
                border_style="bright_blue",
                padding=(0, 1),
            )
        )


def stream(source: str, line: str, detail: str | None = None) -> None:
    text = line.strip()
    if not text:
        return

    lowered = text.lower()
    if any(
        token in lowered
        for token in ("traceback", "exception", " fatal", "failed", "error")
    ):
        emit("ERROR", f"{source} {text}", detail)
        return
    if any(token in lowered for token in ("warning", "warn", "deprecated")):
        emit("WARN", f"{source} {text}", detail)
        return
    emit("INFO", f"{source} {text}", detail)


@dataclass
class Pulse:
    label: str
    detail_factory: DetailFactory | None = None
    interval_seconds: float = 4.0
    stream_target: TextIO | None = None
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _frame_index: int = field(default=0, init=False, repr=False)

    def start(self) -> Pulse:
        if self._thread is not None:
            return self
        info(f"{SPINNER_FRAMES[self._frame_index]} {self.label}", self._detail())
        self._thread = threading.Thread(
            target=self._run, name=f"pulse-{self.label}", daemon=True
        )
        self._thread.start()
        return self

    def stop(
        self, level: str = "OK", message: str | None = None, detail: str | None = None
    ) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds + 0.25)
            self._thread = None
        emit(
            level,
            message or self.label,
            detail or self._detail(),
            stream=self.stream_target,
        )

    def _detail(self) -> str | None:
        if self.detail_factory is None:
            return None
        return self.detail_factory()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self._frame_index = (self._frame_index + 1) % len(SPINNER_FRAMES)
            info(f"{SPINNER_FRAMES[self._frame_index]} {self.label}", self._detail())
