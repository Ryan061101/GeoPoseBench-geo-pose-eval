"""Wrapper script for prediction pose encoding."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geo_pose_eval.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["encode-prediction", *sys.argv[1:]]))

