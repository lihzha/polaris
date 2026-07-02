"""Diagnostic client that replays absolute PolaRiS EEF actions from a trace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from polaris.config import LAP_EEF_FRAME, PolicyArgs
from polaris.policy.abstract_client import InferenceClient
from polaris.policy.lap_eef_pose_client import _finite_vector
from polaris.policy.lap_eef_pose_client import _rgb_uint8
from polaris.policy.lap_eef_pose_client import resize_lap_image


@InferenceClient.register(client_name="ReplayEefPose")
class ReplayEefPoseClient(InferenceClient):
    """Replay an exact recorded action prefix, then hold its final target.

    This client is intentionally diagnostic-only. It consumes the absolute
    eight-dimensional ``polaris_action`` values recorded by
    :class:`EgoLAPEefPoseClient`; it performs no policy inference or frame math.
    If the source rollout ended in a numerical failure, the last valid target
    is held after the recorded prefix so a non-reproducing replay can finish.
    """

    def __init__(self, args: PolicyArgs) -> None:
        if not args.replay_trace_path:
            raise ValueError("ReplayEefPose requires replay_trace_path")
        if args.replay_episode is None or args.replay_episode < 0:
            raise ValueError("ReplayEefPose requires a non-negative replay_episode")
        if args.eef_frame != LAP_EEF_FRAME:
            raise ValueError(
                f"ReplayEefPose requires eef_frame={LAP_EEF_FRAME}; got {args.eef_frame!r}"
            )

        self.args = args
        self.source_path = Path(args.replay_trace_path)
        self.source_episode = int(args.replay_episode)
        self.actions = self._load_actions(self.source_path, self.source_episode)
        self.output_trace_path = Path(args.trace_path) if args.trace_path else None
        if self.output_trace_path is not None:
            self.output_trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.step_index = 0
        print(
            "POLARIS_REPLAY_SOURCE="
            f"{self.source_path};episode={self.source_episode};"
            f"exact_actions={len(self.actions)};eef_frame={LAP_EEF_FRAME}",
            flush=True,
        )

    @staticmethod
    def _load_actions(path: Path, episode: int) -> np.ndarray:
        if not path.is_file():
            raise FileNotFoundError(f"Replay trace does not exist: {path}")
        actions: list[np.ndarray] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") != "action" or record.get("episode") != episode:
                    continue
                action = np.asarray(record.get("polaris_action"), dtype=np.float64)
                if action.shape != (8,) or not np.isfinite(action).all():
                    raise ValueError(
                        f"Invalid replay action at {path}:{line_number}: shape={action.shape}"
                    )
                actions.append(action)
        if not actions:
            raise ValueError(f"Trace {path} contains no actions for episode {episode}")
        return np.stack(actions).astype(np.float32)

    @property
    def rerender(self) -> bool:
        return True

    def _write_trace(self, record: dict[str, Any]) -> None:
        if self.output_trace_path is None:
            return
        with self.output_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        del instruction
        exact_prefix = self.step_index < len(self.actions)
        source_index = min(self.step_index, len(self.actions) - 1)
        action = self.actions[source_index].copy()

        policy_obs = obs["policy"]
        position = _finite_vector(policy_obs["eef_pos"], name="end-effector position", size=3)
        quaternion = _finite_vector(
            policy_obs["eef_quat"], name="end-effector quaternion", size=4
        )
        self._write_trace(
            {
                "event": "replay_action",
                "step": self.step_index,
                "source_episode": self.source_episode,
                "source_action_index": source_index,
                "exact_prefix": exact_prefix,
                "observed_eef_position": position.tolist(),
                "observed_eef_quaternion_wxyz": quaternion.tolist(),
                "polaris_action": action.tolist(),
            }
        )

        visualization = None
        if return_viz:
            splat = obs["splat"]
            external = resize_lap_image(_rgb_uint8(splat["external_cam"], name="external image"))
            wrist = resize_lap_image(_rgb_uint8(splat["wrist_cam"], name="wrist image"))
            visualization = np.concatenate([external, wrist], axis=1)

        self.step_index += 1
        return action, visualization

    def reset(self) -> None:
        self.step_index = 0
        if self.output_trace_path is not None:
            self.output_trace_path.unlink(missing_ok=True)
        self._write_trace(
            {
                "event": "reset",
                "source_trace": str(self.source_path),
                "source_episode": self.source_episode,
                "exact_action_count": len(self.actions),
            }
        )
