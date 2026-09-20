"""Run the preview JavaScript regressions as part of the Python test suite."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")
def test_dataset_preview_javascript() -> None:
    script = Path(__file__).with_name("dataset_preview.test.cjs")
    result = subprocess.run(
        [shutil.which("node"), "--test", str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
