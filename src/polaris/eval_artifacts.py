"""Transactional PolaRiS rollout artifacts and fail-closed resume checks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVAL_RESULT_COLUMNS = (
    "episode",
    "episode_length",
    "success",
    "progress",
    "numerical_failure",
    "numerical_failure_reason",
)


def empty_eval_results() -> pd.DataFrame:
    """Return the canonical empty episode table."""

    return pd.DataFrame(
        {
            "episode": pd.Series(dtype="int64"),
            "episode_length": pd.Series(dtype="int64"),
            "success": pd.Series(dtype="bool"),
            "progress": pd.Series(dtype="float64"),
            "numerical_failure": pd.Series(dtype="bool"),
            "numerical_failure_reason": pd.Series(dtype="str"),
        }
    )


def probe_episode_video(path: Path) -> dict[str, int]:
    """Decode an entire rollout video and return its structural identity."""

    import mediapy  # noqa: PLC0415 - simulator runtime dependency

    try:
        frames = np.asarray(mediapy.read_video(path))
    except Exception as error:
        raise ValueError(f"Rollout video is not decodable: {path}: {error}") from error
    if frames.ndim != 4 or frames.shape[0] < 1 or frames.shape[-1] < 3:
        raise ValueError(
            f"Rollout video must decode as T x H x W x C>=3; got {frames.shape}: {path}"
        )
    return {
        "frame_count": int(frames.shape[0]),
        "height": int(frames.shape[1]),
        "width": int(frames.shape[2]),
    }


def validate_episode_video(
    path: Path,
    *,
    expected_frames: int,
    probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> dict[str, int]:
    """Require one nonempty, decodable 448x224 video matching its CSV row."""

    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing nonempty completed rollout video: {path}")
    probe = probe_fn(path)
    expected = {"frame_count": expected_frames, "height": 224, "width": 448}
    for key, expected_value in expected.items():
        if probe.get(key) != expected_value:
            raise ValueError(
                f"Completed rollout video {key} mismatch for {path}: "
                f"expected={expected_value!r}, actual={probe.get(key)!r}"
            )
    return expected


def sha256_file(path: Path) -> str:
    """Return the SHA-256 identity of one durable rollout artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_episode_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and strictly validate the row shared by CSV/trace/sidecar."""

    required = set(EVAL_RESULT_COLUMNS)
    if set(result) != required:
        raise ValueError(
            "Episode result must have exactly the canonical columns: "
            f"expected={sorted(required)!r}, actual={sorted(result)!r}"
        )
    episode_value = result["episode"]
    length_value = result["episode_length"]
    if (
        not isinstance(episode_value, numbers.Integral)
        or isinstance(episode_value, (bool, np.bool_))
        or int(episode_value) < 0
    ):
        raise ValueError(f"Episode result has invalid episode: {episode_value!r}")
    if (
        not isinstance(length_value, numbers.Integral)
        or isinstance(length_value, (bool, np.bool_))
        or int(length_value) < 1
    ):
        raise ValueError(f"Episode result has invalid length: {length_value!r}")
    success_value = result["success"]
    failure_value = result["numerical_failure"]
    if not isinstance(success_value, (bool, np.bool_)) or not isinstance(
        failure_value, (bool, np.bool_)
    ):
        raise ValueError("Episode success/failure fields must be booleans")
    progress_value = result["progress"]
    if not isinstance(progress_value, numbers.Real) or isinstance(
        progress_value, (bool, np.bool_)
    ):
        raise ValueError(f"Episode progress must be numeric: {progress_value!r}")
    progress = float(progress_value)
    if not math.isfinite(progress):
        raise ValueError(f"Episode progress must be finite: {progress!r}")
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"Episode progress must be in [0, 1]: {progress!r}")
    reason = result["numerical_failure_reason"]
    if not isinstance(reason, str):
        raise ValueError(
            f"Episode numerical failure reason must be a string: {reason!r}"
        )
    numerical_failure = bool(failure_value)
    if numerical_failure != bool(reason):
        raise ValueError(
            "Episode numerical failure flag and reason disagree: "
            f"flag={numerical_failure!r}, reason={reason!r}"
        )
    if numerical_failure and bool(success_value):
        raise ValueError("A numerical-failure episode cannot be successful")
    if numerical_failure and progress != 0.0:
        raise ValueError("A numerical-failure episode must have progress=0.0")
    return {
        "episode": int(episode_value),
        "episode_length": int(length_value),
        "success": bool(success_value),
        "progress": progress,
        "numerical_failure": numerical_failure,
        "numerical_failure_reason": reason,
    }


def _read_trace_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing nonempty completed policy trace: {path}")
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            record = json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {value!r}")
                ),
            )
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not an object")
            records.append(record)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Policy trace is not valid JSONL: {path}: {error}") from error
    if not records:
        raise ValueError(f"Policy trace contains no records: {path}")
    return records


def validate_episode_trace(
    path: Path,
    *,
    episode: int,
    expected_length: int,
    expected_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require one finalized, internally consistent per-episode policy trace."""

    records = _read_trace_records(path)
    if (
        records[0].get("event") != "reset"
        or records[-1].get("event") != "episode_complete"
    ):
        raise ValueError(
            f"Policy trace must start with reset and end with episode_complete: {path}"
        )
    wrong_episode = [
        record.get("episode") for record in records if record.get("episode") != episode
    ]
    if wrong_episode:
        raise ValueError(
            f"Policy trace contains records for the wrong episode {episode}: "
            f"{wrong_episode[:5]} in {path}"
        )
    if records[-1].get("episode_length") != expected_length:
        raise ValueError(
            f"Policy trace length mismatch for episode {episode}: "
            f"expected={expected_length}, actual={records[-1].get('episode_length')!r}"
        )
    action_count = sum(record.get("event") == "action" for record in records)
    if action_count != expected_length:
        raise ValueError(
            f"Policy trace action count mismatch for episode {episode}: "
            f"expected={expected_length}, actual={action_count}"
        )
    terminal = records[-1]
    terminal_result = canonical_episode_result(
        {field: terminal.get(field) for field in EVAL_RESULT_COLUMNS}
    )
    expected_status = (
        "numerical_failure" if terminal_result["numerical_failure"] else "completed"
    )
    if terminal.get("status") != expected_status:
        raise ValueError(
            f"Policy trace terminal status mismatch for episode {episode}: "
            f"expected={expected_status!r}, actual={terminal.get('status')!r}"
        )
    if expected_result is not None:
        canonical_expected = canonical_episode_result(expected_result)
        if terminal_result != canonical_expected:
            raise ValueError(
                f"Policy trace terminal result mismatch for episode {episode}: "
                f"expected={canonical_expected!r}, actual={terminal_result!r}"
            )
    return terminal_result


def build_episode_artifact_identity(
    *,
    run_folder: Path,
    trace_path: Path,
    episode_result: Mapping[str, Any],
    video_probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> dict[str, Any]:
    """Capture immutable video and finalized terminal-trace identities."""

    result = canonical_episode_result(episode_result)
    episode = result["episode"]
    video_path = run_folder / f"episode_{episode}.mp4"
    expected_trace_path = trace_path.parent / f"episode_{episode:06d}.jsonl"
    if trace_path != expected_trace_path:
        raise ValueError(
            "Transactional Ego-LAP evidence requires one per-episode trace: "
            f"expected={expected_trace_path}, actual={trace_path}"
        )
    video_shape = validate_episode_video(
        video_path,
        expected_frames=result["episode_length"],
        probe_fn=video_probe_fn,
    )
    terminal_result = validate_episode_trace(
        trace_path,
        episode=episode,
        expected_length=result["episode_length"],
        expected_result=result,
    )
    return {
        "video": {
            "filename": video_path.name,
            "size_bytes": video_path.stat().st_size,
            "sha256": sha256_file(video_path),
            **video_shape,
        },
        "terminal_trace": {
            "filename": trace_path.name,
            "size_bytes": trace_path.stat().st_size,
            "sha256": sha256_file(trace_path),
            "episode_result": terminal_result,
        },
    }


def validate_episode_artifact_identity(
    identity: Mapping[str, Any],
    *,
    run_folder: Path,
    trace_dir: Path,
    episode_result: Mapping[str, Any],
    video_probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> dict[str, Any]:
    """Recompute and exactly match one sidecar's artifact identities."""

    result = canonical_episode_result(episode_result)
    actual = build_episode_artifact_identity(
        run_folder=run_folder,
        trace_path=trace_dir / f"episode_{result['episode']:06d}.jsonl",
        episode_result=result,
        video_probe_fn=video_probe_fn,
    )
    if dict(identity) != actual:
        raise ValueError(
            "Episode sidecar artifact identity drift: "
            f"recorded={dict(identity)!r}, actual={actual!r}"
        )
    return actual


def _validate_legacy_trace(
    path: Path,
    *,
    completed_rows: Sequence[tuple[int, int]],
) -> None:
    records = _read_trace_records(path)
    for episode, expected_length in completed_rows:
        episode_records = [
            record for record in records if record.get("episode") == episode
        ]
        if not episode_records:
            raise ValueError(
                f"Legacy policy trace has no records for episode {episode}: {path}"
            )
        if (
            episode_records[0].get("event") != "reset"
            or episode_records[-1].get("event") != "episode_complete"
        ):
            raise ValueError(
                f"Legacy policy trace episode {episode} is incomplete: {path}"
            )
        action_count = sum(
            record.get("event") == "action" for record in episode_records
        )
        if action_count != expected_length:
            raise ValueError(
                f"Legacy policy trace action count mismatch for episode {episode}: "
                f"expected={expected_length}, actual={action_count}"
            )


def load_resume_results(
    csv_path: Path,
    *,
    run_folder: Path,
    expected_rollouts: int,
    expected_horizon: int,
    require_episode_artifacts: bool,
    trace_dir: Path | None = None,
    trace_path: Path | None = None,
    video_probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> pd.DataFrame:
    """Load only a contiguous, artifact-complete prefix of rollout results."""

    if trace_dir is not None and trace_path is not None:
        raise ValueError("Configure either trace_dir or trace_path, not both")
    if not csv_path.exists():
        return empty_eval_results()
    try:
        frame = pd.read_csv(csv_path)
    except Exception as error:
        raise ValueError(f"Could not read resume CSV {csv_path}: {error}") from error
    required = {"episode", "episode_length", "success", "progress"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Resume CSV is missing required columns {missing}: {csv_path}"
        )
    if len(frame) > expected_rollouts:
        raise ValueError(
            f"Resume CSV has {len(frame)} rows but only {expected_rollouts} rollouts were requested"
        )

    episodes: list[int] = []
    lengths: list[int] = []
    for row_index, (episode_value, length_value) in enumerate(
        zip(frame["episode"], frame["episode_length"], strict=True)
    ):
        try:
            episode = int(episode_value)
            length = int(length_value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Resume CSV row {row_index} has a non-integer identity"
            ) from error
        if float(episode_value) != episode or float(length_value) != length:
            raise ValueError(f"Resume CSV row {row_index} has a non-integral identity")
        if not 1 <= length <= expected_horizon:
            raise ValueError(
                f"Resume CSV episode {episode} length must be in [1, {expected_horizon}], got {length}"
            )
        episodes.append(episode)
        lengths.append(length)
    expected_episodes = list(range(len(frame)))
    if episodes != expected_episodes:
        raise ValueError(
            f"Resume CSV episode IDs must be the contiguous prefix {expected_episodes}; got {episodes}"
        )

    if "numerical_failure" not in frame:
        frame["numerical_failure"] = False
    if "numerical_failure_reason" not in frame:
        frame["numerical_failure_reason"] = ""
    frame["numerical_failure_reason"] = frame[
        "numerical_failure_reason"
    ].fillna("")
    frame = frame.loc[:, list(EVAL_RESULT_COLUMNS)]
    normalized_rows = [
        canonical_episode_result(row)
        for row in frame.to_dict(orient="records")
    ]
    frame = (
        pd.DataFrame(normalized_rows, columns=EVAL_RESULT_COLUMNS)
        if normalized_rows
        else empty_eval_results()
    )

    if require_episode_artifacts:
        if trace_dir is None and trace_path is None:
            raise ValueError("Ego-LAP resume requires per-episode policy traces")
        for episode, length in zip(episodes, lengths, strict=True):
            result = canonical_episode_result(
                frame.loc[frame["episode"] == episode].iloc[0].to_dict()
            )
            validate_episode_video(
                run_folder / f"episode_{episode}.mp4",
                expected_frames=length,
                probe_fn=video_probe_fn,
            )
            if trace_dir is not None:
                validate_episode_trace(
                    trace_dir / f"episode_{episode:06d}.jsonl",
                    episode=episode,
                    expected_length=length,
                    expected_result=result,
                )
        if trace_path is not None and episodes:
            _validate_legacy_trace(
                trace_path,
                completed_rows=list(zip(episodes, lengths, strict=True)),
            )
    return frame


def atomic_write_episode_video(
    path: Path,
    frames: Sequence[np.ndarray],
    *,
    fps: int,
    writer: Callable[..., Any] | None = None,
    probe_fn: Callable[[Path], Mapping[str, Any]] = probe_episode_video,
) -> None:
    """Write, decode-check, and atomically publish one episode video."""

    if not frames:
        raise ValueError("Cannot finalize an empty rollout video")
    if writer is None:
        import mediapy  # noqa: PLC0415 - simulator runtime dependency

        writer = mediapy.write_video
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        writer(temporary, frames, fps=fps)
        validate_episode_video(
            temporary,
            expected_frames=len(frames),
            probe_fn=probe_fn,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_results(frame: pd.DataFrame, path: Path) -> None:
    """Atomically replace the episode CSV after all row artifacts are durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as output:
            frame.to_csv(output, index=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
