#!/usr/bin/env python3
"""Prepare a completed-episode prefix for a fresh pi0.5 retry."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_resume_prefix(
    source_task_dir: Path,
    destination_task_dir: Path,
    expected_rollouts: int,
) -> dict:
    source_task_dir = source_task_dir.resolve()
    destination_task_dir = destination_task_dir.resolve()
    if source_task_dir == destination_task_dir:
        raise ValueError("Resume source and destination must differ")
    if expected_rollouts <= 1:
        raise ValueError("expected_rollouts must be greater than one")

    source_csv = source_task_dir / "eval_results.csv"
    source_trace = source_task_dir / "policy_traces.jsonl"
    if not source_csv.is_file() or not source_trace.is_file():
        raise ValueError("Resume source is missing metrics or policy trace")

    with source_csv.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    completed_episodes = len(rows)
    if not 0 < completed_episodes < expected_rollouts:
        raise ValueError(
            "Resume source must contain a nonempty strict prefix of the rollout set"
        )
    for expected_episode, row in enumerate(rows):
        if int(float(row["episode"])) != expected_episode:
            raise ValueError(
                f"Resume metrics are not contiguous at episode {expected_episode}"
            )

    expected_videos = [
        source_task_dir / f"episode_{episode}.mp4"
        for episode in range(completed_episodes)
    ]
    if any(not video.is_file() or video.stat().st_size == 0 for video in expected_videos):
        raise ValueError("Resume source is missing a completed-episode video")
    source_video_names = {
        path.name
        for path in source_task_dir.glob("episode_*.mp4")
        if path.is_file() and path.stat().st_size > 0
    }
    expected_video_names = {path.name for path in expected_videos}
    if source_video_names != expected_video_names:
        raise ValueError(
            "Resume source video set does not exactly match completed metrics"
        )

    retained_lines = []
    discarded_records = 0
    retained_records = 0
    with source_trace.open("rb") as trace_file:
        for line_number, raw_line in enumerate(trace_file, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid source trace JSON at line {line_number}: {error}"
                ) from error
            reset_index = record.get("reset_index")
            if not isinstance(reset_index, int) or reset_index < 0:
                raise ValueError(
                    f"Invalid source trace reset_index at line {line_number}"
                )
            if reset_index < completed_episodes:
                retained_lines.append(raw_line)
                retained_records += 1
            else:
                discarded_records += 1
    if not retained_lines:
        raise ValueError("Resume source has no completed-episode trace records")

    destination_task_dir.mkdir(parents=True, exist_ok=True)
    destination_csv = destination_task_dir / source_csv.name
    destination_trace = destination_task_dir / source_trace.name
    if destination_csv.exists() or destination_trace.exists():
        raise ValueError("Resume destination already contains metrics or trace")
    if any(destination_task_dir.glob("episode_*.mp4")):
        raise ValueError("Resume destination already contains episode videos")

    shutil.copy2(source_csv, destination_csv)
    with destination_trace.open("wb") as trace_file:
        trace_file.writelines(retained_lines)
    for source_video in expected_videos:
        shutil.copy2(source_video, destination_task_dir / source_video.name)

    return {
        "schema_version": 1,
        "source_task_dir": str(source_task_dir),
        "destination_task_dir": str(destination_task_dir),
        "expected_rollouts": expected_rollouts,
        "completed_episodes": completed_episodes,
        "retained_trace_records": retained_records,
        "discarded_partial_trace_records": discarded_records,
        "source_metrics_sha256": _sha256(source_csv),
        "destination_metrics_sha256": _sha256(destination_csv),
        "source_trace_sha256": _sha256(source_trace),
        "destination_trace_sha256": _sha256(destination_trace),
        "video_count": len(expected_videos),
        "status": "prepared",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_task_dir", type=Path)
    parser.add_argument("destination_task_dir", type=Path)
    parser.add_argument("--expected-rollouts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = prepare_resume_prefix(
        args.source_task_dir,
        args.destination_task_dir,
        args.expected_rollouts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
