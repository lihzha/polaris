#!/usr/bin/env python3
"""Fully probe and decode official pi0.5 joint-position rollout videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polaris.pi05_droid_jointpos_video import (
    build_video_report,
    publish_video_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--container-image-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_video_report(
        args.task_dir,
        expected_rollouts=args.expected_rollouts,
        container_image_sha256=args.container_image_sha256,
    )
    artifact = publish_video_report(args.output, report)
    print(json.dumps(artifact, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
