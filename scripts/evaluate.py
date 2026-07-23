"""Wrapper script for geometric evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geo_pose_eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))

