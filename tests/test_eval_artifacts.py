import json
from pathlib import Path

import pandas as pd
import pytest

from polaris.eval_artifacts import atomic_write_episode_video
from polaris.eval_artifacts import atomic_write_results
from polaris.eval_artifacts import load_resume_results


def _write_trace(path: Path, *, episode: int, length: int) -> None:
    records = [
        {"event": "reset", "episode": episode},
        *(
            {"event": "action", "episode": episode, "step": step}
            for step in range(length)
        ),
        {
            "event": "episode_complete",
            "episode": episode,
            "episode_length": length,
            "status": "completed",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _frame(episode: int = 0, length: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "episode": episode,
                "episode_length": length,
                "success": False,
                "progress": 0.25,
                "numerical_failure": False,
                "numerical_failure_reason": "",
            }
        ]
    )


def _video_probe(_path: Path, *, frames: int = 2):
    return {"frame_count": frames, "height": 224, "width": 448}


def test_resume_requires_contiguous_rows_video_and_finalized_trace(tmp_path: Path):
    csv_path = tmp_path / "eval_results.csv"
    _frame().to_csv(csv_path, index=False)
    (tmp_path / "episode_0.mp4").write_bytes(b"video")
    trace_dir = tmp_path / "policy_traces"
    _write_trace(trace_dir / "episode_000000.jsonl", episode=0, length=2)

    actual = load_resume_results(
        csv_path,
        run_folder=tmp_path,
        expected_rollouts=50,
        expected_horizon=450,
        require_episode_artifacts=True,
        trace_dir=trace_dir,
        video_probe_fn=_video_probe,
    )

    assert actual["episode"].tolist() == [0]


def test_resume_rejects_noncontiguous_episode_ids(tmp_path: Path):
    csv_path = tmp_path / "eval_results.csv"
    _frame(episode=1).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="contiguous prefix"):
        load_resume_results(
            csv_path,
            run_folder=tmp_path,
            expected_rollouts=50,
            expected_horizon=450,
            require_episode_artifacts=False,
        )


def test_resume_rejects_missing_video_and_incomplete_trace(tmp_path: Path):
    csv_path = tmp_path / "eval_results.csv"
    _frame().to_csv(csv_path, index=False)
    trace_dir = tmp_path / "policy_traces"
    _write_trace(trace_dir / "episode_000000.jsonl", episode=0, length=2)
    with pytest.raises(ValueError, match="Missing nonempty completed rollout video"):
        load_resume_results(
            csv_path,
            run_folder=tmp_path,
            expected_rollouts=50,
            expected_horizon=450,
            require_episode_artifacts=True,
            trace_dir=trace_dir,
            video_probe_fn=_video_probe,
        )

    (tmp_path / "episode_0.mp4").write_bytes(b"video")
    trace_path = trace_dir / "episode_000000.jsonl"
    records = [json.loads(line) for line in trace_path.read_text().splitlines()]
    trace_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records[:-2] + records[-1:])
    )
    with pytest.raises(ValueError, match="action count"):
        load_resume_results(
            csv_path,
            run_folder=tmp_path,
            expected_rollouts=50,
            expected_horizon=450,
            require_episode_artifacts=True,
            trace_dir=trace_dir,
            video_probe_fn=_video_probe,
        )


def test_episode_video_and_csv_publish_atomically(tmp_path: Path):
    video_path = tmp_path / "episode_0.mp4"
    calls = []

    def writer(path, frames, *, fps):
        calls.append((path, len(frames), fps))
        path.write_bytes(b"complete-video")

    atomic_write_episode_video(
        video_path,
        [object(), object()],
        fps=15,
        writer=writer,
        probe_fn=_video_probe,
    )
    assert video_path.read_bytes() == b"complete-video"
    assert calls[0][1:] == (2, 15)
    assert not list(tmp_path.glob(".*.tmp.mp4"))

    csv_path = tmp_path / "eval_results.csv"
    atomic_write_results(_frame(), csv_path)
    assert pd.read_csv(csv_path)["episode"].tolist() == [0]
    assert not list(tmp_path.glob(".*.tmp.csv"))


def test_failed_video_publish_preserves_existing_final(tmp_path: Path):
    video_path = tmp_path / "episode_0.mp4"
    video_path.write_bytes(b"old-complete")

    def failing_writer(path, frames, *, fps):
        path.write_bytes(b"partial")
        raise RuntimeError("encoder failed")

    with pytest.raises(RuntimeError, match="encoder failed"):
        atomic_write_episode_video(
            video_path,
            [object()],
            fps=15,
            writer=failing_writer,
            probe_fn=lambda _path: {
                "frame_count": 1,
                "height": 224,
                "width": 448,
            },
        )
    assert video_path.read_bytes() == b"old-complete"
    assert not list(tmp_path.glob(".*.tmp.mp4"))
