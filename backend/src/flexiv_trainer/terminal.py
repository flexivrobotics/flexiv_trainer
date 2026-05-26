from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import TextIO

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"

LEVEL_STYLES = {
    "INFO": "\033[38;5;81m",
    "WARN": "\033[38;5;214m",
    "ERROR": "\033[38;5;203m",
    "OK": "\033[38;5;114m",
}


def _supports_color(stream: TextIO) -> bool:
    return (
        hasattr(stream, "isatty")
        and stream.isatty()
        and os.getenv("TERM")
        not in {
            None,
            "dumb",
        }
    )


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _style(text: str, color: str, stream: TextIO) -> str:
    if not _supports_color(stream):
        return text
    return f"{color}{text}{RESET}"


def _line(
    level: str, message: str, detail: str | None = None, *, stream: TextIO
) -> str:
    prefix = f"[{_stamp()}] [{level}]"
    prefix = _style(prefix, LEVEL_STYLES[level], stream)
    if _supports_color(stream):
        message = f"{BOLD}{message}{RESET}"
        if detail:
            detail = f"{DIM}{detail}{RESET}"
    base = f"{prefix} {message}"
    if detail:
        base = f"{base} {detail}"
    return base


def emit(
    level: str, message: str, detail: str | None = None, *, stream: TextIO | None = None
) -> None:
    target = stream or (sys.stderr if level in {"WARN", "ERROR"} else sys.stdout)
    print(_line(level, message, detail, stream=target), file=target, flush=True)


def info(message: str, detail: str | None = None) -> None:
    emit("INFO", message, detail)


def warn(message: str, detail: str | None = None) -> None:
    emit("WARN", message, detail)


def error(message: str, detail: str | None = None) -> None:
    emit("ERROR", message, detail)


def ok(message: str, detail: str | None = None) -> None:
    emit("OK", message, detail)


def banner(title: str, *lines: str) -> None:
    width = max(len(title), *(len(line) for line in lines), 24)
    border = "+" + "-" * (width + 2) + "+"
    print(border, flush=True)
    print(f"| {title.ljust(width)} |", flush=True)
    print(border, flush=True)
    for line in lines:
        print(f"| {line.ljust(width)} |", flush=True)
    print(border, flush=True)
