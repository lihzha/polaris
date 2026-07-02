import json
from pathlib import Path

import numpy as np

from polaris.config import PolicyArgs
from polaris.policy.replay_eef_pose_client import ReplayEefPoseClient


def _write_source(path: Path) -> None:
    records = [
        {"event": "action", "episode": 2, "polaris_action": [1, 2, 3, 1, 0, 0, 0, 0]},
        {"event": "action", "episode": 3, "polaris_action": [9, 9, 9, 1, 0, 0, 0, 1]},
        {"event": "action", "episode": 2, "polaris_action": [4, 5, 6, 1, 0, 0, 0, 1]},
    ]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _observation() -> dict:
    return {
        "policy": {
            "eef_pos": np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            "eef_quat": np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        },
        "splat": {
            "external_cam": np.zeros((8, 12, 3), dtype=np.uint8),
            "wrist_cam": np.full((8, 12, 3), 255, dtype=np.uint8),
        },
    }


def test_replays_selected_episode_then_holds_last_action(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "replay.jsonl"
    _write_source(source)
    client = ReplayEefPoseClient(
        PolicyArgs(
            client="ReplayEefPose",
            replay_trace_path=str(source),
            replay_episode=2,
            trace_path=str(output),
        )
    )
    client.reset()

    first, visualization = client.infer(_observation(), "ignored", return_viz=True)
    second, _ = client.infer(_observation(), "ignored")
    held, _ = client.infer(_observation(), "ignored")

    np.testing.assert_array_equal(first, [1, 2, 3, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(second, [4, 5, 6, 1, 0, 0, 0, 1])
    np.testing.assert_array_equal(held, second)
    assert visualization is not None and visualization.shape == (224, 448, 3)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["event"] == "reset"
    assert [record["exact_prefix"] for record in records[1:]] == [True, True, False]


def test_rejects_missing_episode(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_source(source)
    try:
        ReplayEefPoseClient(
            PolicyArgs(
                client="ReplayEefPose",
                replay_trace_path=str(source),
                replay_episode=99,
            )
        )
    except ValueError as error:
        assert "contains no actions" in str(error)
    else:
        raise AssertionError("missing replay episode should fail")
