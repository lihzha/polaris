"""Ego-LAP client for absolute end-effector pose control in PolaRiS."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import image_tools, websocket_client_policy
from scipy.spatial.transform import Rotation

from polaris.config import PolicyArgs
from polaris.policy.abstract_client import InferenceClient


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _finite_vector(value: Any, *, name: str, size: int) -> np.ndarray:
    """Convert a tensor/array-like value to one finite single-environment vector."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.asarray(value, dtype=np.float64)
    if array.shape == (1, size):
        array = array[0]
    if array.shape != (size,):
        raise ValueError(
            f"{name} must have shape ({size},) or (1, {size}); got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _unit_quaternion_wxyz(value: Any, *, name: str) -> np.ndarray:
    quaternion = _finite_vector(value, name=name, size=4)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-8:
        raise ValueError(f"{name} has near-zero norm")
    return quaternion / norm


def rotate_image_180(image: np.ndarray) -> np.ndarray:
    """Match DROID's training-time 180-degree wrist-image rotation."""

    return np.asarray(image)[::-1, ::-1].copy()


def quaternion_wxyz_to_rot6d(quaternion_wxyz: Any) -> np.ndarray:
    """Encode an Isaac ``wxyz`` quaternion as Ego-LAP's two-column R6 state."""

    quaternion_wxyz = _unit_quaternion_wxyz(
        quaternion_wxyz, name="end-effector quaternion"
    )
    quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
    rotation_matrix = Rotation.from_quat(quaternion_xyzw).as_matrix()
    return np.concatenate([rotation_matrix[:, 0], rotation_matrix[:, 1]])


def build_lap_state(
    eef_position: Any,
    eef_quaternion_wxyz: Any,
    closed_gripper: Any,
) -> np.ndarray:
    """Build ``[xyz, rot6d, open_gripper]`` for Ego-LAP."""

    position = _finite_vector(eef_position, name="end-effector position", size=3)
    rot6d = quaternion_wxyz_to_rot6d(eef_quaternion_wxyz)
    closed = _finite_vector(closed_gripper, name="closed gripper", size=1)[0]
    # Match the official DROID runner: invert the normalized observation and
    # binarize at 0.5 before building the policy state.
    open_gripper = float((1.0 - np.clip(closed, 0.0, 1.0)) > 0.5)
    state = np.concatenate([position, rot6d, np.array([open_gripper])])
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid Ego-LAP state with shape {state.shape}")
    return state.astype(np.float32)


def validate_action_chunk(response: Any) -> np.ndarray:
    """Return a non-empty, finite ``T x 7`` action chunk or fail loudly."""

    if not isinstance(response, dict):
        raise TypeError(
            f"Policy response must be a dict; got {type(response).__name__}"
        )
    if "actions" not in response:
        raise KeyError("Policy response is missing the 'actions' field")
    actions = np.asarray(response["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 7:
        raise ValueError(
            f"Ego-LAP actions must have shape (T, 7), T >= 1; got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("Ego-LAP action chunk contains non-finite values")
    return actions


def resolve_action_frame(action_frame: str) -> str:
    """Normalize the explicit numeric frame used by LAP's flow actions."""

    normalized = " ".join(action_frame.lower().replace("_", " ").split())
    if normalized in {"robot base", "robot base frame", "base", "base frame"}:
        return "robot_base"
    if normalized in {"egocentric", "egocentric frame", "eef", "eef frame"}:
        return "egocentric"
    raise ValueError(
        "Unsupported action_frame. Expected robot_base or egocentric; "
        f"got {action_frame!r}"
    )


def egocentric_action_chunk_to_base(
    delta_actions: Any,
    anchor_quaternion_wxyz: Any,
    *,
    dataset_name: str,
    rotation_applied: bool,
) -> np.ndarray:
    """Invert Ego-LAP's single-arm DROID semantic EEF-frame transform."""

    actions = np.asarray(delta_actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 7:
        raise ValueError(
            f"Ego-LAP delta actions must have shape (T, 7), T >= 1; got {actions.shape}"
        )
    if dataset_name not in {"droid", "droid_100"}:
        raise ValueError(
            "Egocentric action decoding currently supports the DROID convention; "
            f"got dataset_name={dataset_name!r}"
        )

    anchor_wxyz = _unit_quaternion_wxyz(
        anchor_quaternion_wxyz, name="anchor quaternion"
    )
    eef_to_base = Rotation.from_quat(anchor_wxyz[[1, 2, 3, 0]]).as_matrix()

    base_actions = actions.copy()
    geometric_eef_positions = actions[:, :3] * np.array([1.0, -1.0, -1.0])
    base_actions[:, :3] = geometric_eef_positions @ eef_to_base.T

    geometric_eef_euler = actions[:, 3:6].copy()
    if not rotation_applied:
        geometric_eef_euler *= np.array([1.0, -1.0, -1.0])
    eef_delta_matrices = Rotation.from_euler("xyz", geometric_eef_euler).as_matrix()
    base_delta_matrices = (
        eef_to_base[None, :, :] @ eef_delta_matrices @ eef_to_base.T[None, :, :]
    )
    base_actions[:, 3:6] = Rotation.from_matrix(base_delta_matrices).as_euler("xyz")
    return base_actions


def anchor_action_chunk(
    delta_actions: Any,
    anchor_position: Any,
    anchor_quaternion_wxyz: Any,
    *,
    action_frame: str = "robot_base",
    dataset_name: str = "droid",
    rotation_applied: bool = True,
) -> np.ndarray:
    """Anchor one full Ego-LAP delta chunk to one query-time EEF pose.

    Ego-LAP emits ``[dx, dy, dz, droll, dpitch, dyaw, gripper_open]``.
    Every delta in the chunk is relative to the same query-time anchor. The
    returned PolaRiS actions are
    ``[x, y, z, qw, qx, qy, qz, gripper_closed]``.
    """

    delta_actions = np.asarray(delta_actions, dtype=np.float64)
    if (
        delta_actions.ndim != 2
        or delta_actions.shape[0] < 1
        or delta_actions.shape[1] != 7
    ):
        raise ValueError(
            f"Ego-LAP delta actions must have shape (T, 7), T >= 1; got {delta_actions.shape}"
        )
    if not np.isfinite(delta_actions).all():
        raise ValueError("Ego-LAP delta action chunk contains non-finite values")

    anchor_position = _finite_vector(anchor_position, name="anchor position", size=3)
    anchor_wxyz = _unit_quaternion_wxyz(
        anchor_quaternion_wxyz, name="anchor quaternion"
    )
    anchor_xyzw = anchor_wxyz[[1, 2, 3, 0]]

    action_frame = resolve_action_frame(action_frame)
    if action_frame == "egocentric":
        delta_actions = egocentric_action_chunk_to_base(
            delta_actions,
            anchor_wxyz,
            dataset_name=dataset_name,
            rotation_applied=rotation_applied,
        )

    target_positions = anchor_position[None, :] + delta_actions[:, :3]
    anchor_rotation = Rotation.from_quat(anchor_xyzw)
    delta_rotations = Rotation.from_euler("xyz", delta_actions[:, 3:6])
    target_xyzw = (anchor_rotation * delta_rotations).as_quat()
    target_wxyz = target_xyzw[:, [3, 0, 1, 2]]

    # Ego-LAP and DROID use open-positive gripper values; PolaRiS is
    # closed-positive. Clipping keeps the binary action term well-defined.
    closed_gripper = 1.0 - np.clip(delta_actions[:, 6:7], 0.0, 1.0)
    actions = np.concatenate([target_positions, target_wxyz, closed_gripper], axis=1)
    if actions.shape != (len(delta_actions), 8) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid anchored PolaRiS actions with shape {actions.shape}")
    return actions.astype(np.float32)


def _rgb_uint8(image: Any, *, name: str) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"{name} must have shape (H, W, C>=3); got {image.shape}")
    image = image[..., :3]
    if not np.isfinite(image).all():
        raise ValueError(f"{name} contains non-finite values")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    image = image.astype(np.float32)
    if image.size and image.min() >= 0.0 and image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


@InferenceClient.register(client_name="EgoLAPEefPose")
class EgoLAPEefPoseClient(InferenceClient):
    """Websocket Ego-LAP client that emits absolute PolaRiS EEF targets."""

    def __init__(self, args: PolicyArgs) -> None:
        if args.open_loop_horizon is not None and args.open_loop_horizon < 1:
            raise ValueError("open_loop_horizon must be positive or None")
        if not args.frame_description.strip():
            raise ValueError("frame_description must not be empty")
        resolve_action_frame(args.action_frame)

        self.args = args
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        self.open_loop_horizon = args.open_loop_horizon
        self.trace_path = Path(args.trace_path) if args.trace_path else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        self.pred_action_chunk: np.ndarray | None = None
        self.raw_delta_chunk: np.ndarray | None = None
        self.actions_from_chunk_completed = 0
        self.current_execution_horizon = 0
        self.episode_index = -1
        self.query_index = 0
        self.step_index = 0

    @property
    def rerender(self) -> bool:
        if self.args.render_every_step:
            # Full-step videos must use the same splat-composited cameras as
            # query frames, rather than raw simulator cameras between queries.
            return True
        return (
            self.pred_action_chunk is None
            or self.actions_from_chunk_completed >= self.current_execution_horizon
        )

    def reset(self):
        self.pred_action_chunk = None
        self.raw_delta_chunk = None
        self.actions_from_chunk_completed = 0
        self.current_execution_horizon = 0
        self.episode_index += 1
        self.query_index = 0
        self.step_index = 0
        self._write_trace({"event": "reset", "episode": self.episode_index})

    def visualize(self, observation: dict) -> np.ndarray:
        """Return the external and rotated wrist images exactly as LAP sees them."""

        current = self._extract_observation(observation)
        exterior_image, wrist_image = self._model_images(current)
        return np.concatenate([exterior_image, wrist_image], axis=1)

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Infer or consume a chunk, anchoring the whole chunk exactly once."""

        visualization = None
        needs_query = (
            self.pred_action_chunk is None
            or self.actions_from_chunk_completed >= self.current_execution_horizon
        )

        if needs_query:
            current = self._extract_observation(obs)
            request, exterior_image, wrist_image = self._build_request(
                current, instruction
            )
            response = self.client.infer(request)
            raw_delta_chunk = validate_action_chunk(response)
            execution_horizon = (
                len(raw_delta_chunk)
                if self.open_loop_horizon is None
                else self.open_loop_horizon
            )
            if execution_horizon > len(raw_delta_chunk):
                raise ValueError(
                    "open_loop_horizon exceeds returned action chunk: "
                    f"{execution_horizon} > {len(raw_delta_chunk)}"
                )

            anchored_chunk = anchor_action_chunk(
                raw_delta_chunk,
                current["eef_position"],
                current["eef_quaternion_wxyz"],
                action_frame=self.args.action_frame,
                dataset_name=self.args.dataset_name,
                rotation_applied=self.args.rotate_wrist_180,
            )
            action_frame = resolve_action_frame(self.args.action_frame)
            base_delta_chunk = (
                raw_delta_chunk
                if action_frame == "robot_base"
                else egocentric_action_chunk_to_base(
                    raw_delta_chunk,
                    current["eef_quaternion_wxyz"],
                    dataset_name=self.args.dataset_name,
                    rotation_applied=self.args.rotate_wrist_180,
                )
            )
            self.raw_delta_chunk = raw_delta_chunk
            self.pred_action_chunk = anchored_chunk
            self.actions_from_chunk_completed = 0
            self.current_execution_horizon = execution_horizon

            self._write_trace(
                {
                    "event": "query",
                    "episode": self.episode_index,
                    "query": self.query_index,
                    "step": self.step_index,
                    "instruction": instruction,
                    "frame_description": self.args.frame_description,
                    "numeric_action_frame": action_frame,
                    "anchor_position": current["eef_position"].tolist(),
                    "anchor_quaternion_wxyz": current["eef_quaternion_wxyz"].tolist(),
                    "state": request["observation"]["state"].tolist(),
                    "raw_delta_chunk": raw_delta_chunk.tolist(),
                    "base_delta_chunk": base_delta_chunk.tolist(),
                    "anchored_action_chunk": anchored_chunk.tolist(),
                    "execution_horizon": execution_horizon,
                    "reasoning": response.get("reasoning"),
                }
            )
            self.query_index += 1
            if return_viz:
                visualization = np.concatenate([exterior_image, wrist_image], axis=1)
        elif return_viz:
            # Record every simulator step, not just policy-query frames.
            visualization = self.visualize(obs)

        if self.pred_action_chunk is None or self.raw_delta_chunk is None:
            raise RuntimeError("No Ego-LAP action chunk is available")

        chunk_index = self.actions_from_chunk_completed
        action = self.pred_action_chunk[chunk_index].copy()
        self._write_trace(
            {
                "event": "action",
                "episode": self.episode_index,
                "query": self.query_index - 1,
                "step": self.step_index,
                "chunk_index": chunk_index,
                "raw_delta": self.raw_delta_chunk[chunk_index].tolist(),
                "polaris_action": action.tolist(),
            }
        )
        self.actions_from_chunk_completed += 1
        self.step_index += 1
        return action, visualization

    def _extract_observation(self, obs_dict: dict) -> dict[str, np.ndarray]:
        try:
            splat_observation = obs_dict["splat"]
            policy_observation = obs_dict["policy"]
            external_image = splat_observation["external_cam"]
            wrist_image = splat_observation["wrist_cam"]
            eef_position = policy_observation["eef_pos"]
            eef_quaternion = policy_observation["eef_quat"]
            closed_gripper = policy_observation["gripper_pos"]
        except KeyError as error:
            raise KeyError(f"Missing PolaRiS observation field: {error}") from error

        return {
            "external_image": _rgb_uint8(external_image, name="external camera image"),
            "wrist_image": _rgb_uint8(wrist_image, name="wrist camera image"),
            "eef_position": _finite_vector(
                eef_position, name="end-effector position", size=3
            ),
            "eef_quaternion_wxyz": _unit_quaternion_wxyz(
                eef_quaternion, name="end-effector quaternion"
            ),
            "closed_gripper": _finite_vector(
                closed_gripper, name="closed gripper", size=1
            ),
        }

    def _model_images(
        self, current: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        exterior_image = image_tools.resize_with_pad(
            current["external_image"], 224, 224
        )
        wrist_image = current["wrist_image"]
        if self.args.rotate_wrist_180:
            wrist_image = rotate_image_180(wrist_image)
        wrist_image = image_tools.resize_with_pad(wrist_image, 224, 224)
        return exterior_image, wrist_image

    def _build_request(
        self, current: dict[str, np.ndarray], instruction: str
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        exterior_image, wrist_image = self._model_images(current)
        state = build_lap_state(
            current["eef_position"],
            current["eef_quaternion_wxyz"],
            current["closed_gripper"],
        )
        request = {
            "observation": {
                "base_0_rgb": exterior_image,
                "left_wrist_0_rgb": wrist_image,
                "cartesian_position": state[:9],
                "gripper_position": state[9:],
                "state": state,
            },
            "prompt": instruction,
            "frame_description": self.args.frame_description,
            "dataset_name": self.args.dataset_name,
            "state_type": self.args.state_type,
            "has_wrist_image": True,
            "is_bimanual": False,
            "rotation_applied": self.args.rotate_wrist_180,
        }
        return request, exterior_image, wrist_image

    def _write_trace(self, record: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        payload = {"timestamp": time.time(), **record}
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(
                json.dumps(payload, separators=(",", ":"), default=_json_default) + "\n"
            )
