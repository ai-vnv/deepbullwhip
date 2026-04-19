"""V&V specification: CLI validation and scripted assessment."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = _REPO_ROOT / "vnvspec.yaml"
_VNV_ASSESS = _REPO_ROOT / "scripts" / "vnv_assess.py"


pytest.importorskip("vnvspec", reason="vnvspec required for V&V tests")


def test_vnvspec_yaml_validates() -> None:
    """``vnvspec validate`` must succeed (same gate as CI ``vnv`` job)."""
    exe = shutil.which("vnvspec")
    if exe is None:
        pytest.skip("vnvspec CLI not on PATH (install vnv extras: pip install -e '.[vnv]')")
    proc = subprocess.run(
        [exe, "validate", str(_SPEC), "-v"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


def test_vnv_assess_script_passes() -> None:
    """``scripts/vnv_assess.py`` must satisfy every requirement with evidence."""
    proc = subprocess.run(
        [sys.executable, str(_VNV_ASSESS)],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
