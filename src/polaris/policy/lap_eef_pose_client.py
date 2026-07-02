"""Ego-LAP client for absolute end-effector pose control in PolaRiS."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation

from polaris.config import LAP_EEF_FRAME, PolicyArgs
from polaris.policy.abstract_client import InferenceClient
from polaris.policy.ego_lap_contract import R6_COLUMNS_STATE_LAYOUT
from polaris.policy.ego_lap_contract import R6_ROWS_STATE_LAYOUT
from polaris.policy.ego_lap_contract import persist_ego_lap_contract
from polaris.policy.ego_lap_contract import validate_ego_lap_server_metadata


LAP_IMAGE_SIZE = 224
LAP_IMAGE_PREPROCESSOR_MARKER = (
    "POLARIS_LAP_IMAGE_PREPROCESSOR="
    "tf_bilinear_half_pixel_antialias_false_uint8_round_"
    "symmetric_zero_pad_224x224_numpy_float32_exact_v2"
)
LAP_EEF_FRAME_MARKER = f"POLARIS_LAP_POLICY_EEF_FRAME={LAP_EEF_FRAME}"


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


def quaternion_wxyz_to_rot6d(
    quaternion_wxyz: Any,
    *,
    state_layout: str,
) -> np.ndarray:
    """Encode an Isaac ``wxyz`` quaternion using the contracted R6 ordering."""

    quaternion_wxyz = _unit_quaternion_wxyz(
        quaternion_wxyz, name="end-effector quaternion"
    )
    quaternion_xyzw = quaternion_wxyz[[1, 2, 3, 0]]
    rotation_matrix = Rotation.from_quat(quaternion_xyzw).as_matrix()
    if state_layout == R6_ROWS_STATE_LAYOUT:
        return np.concatenate([rotation_matrix[0, :], rotation_matrix[1, :]])
    if state_layout == R6_COLUMNS_STATE_LAYOUT:
        return np.concatenate([rotation_matrix[:, 0], rotation_matrix[:, 1]])
    raise ValueError(f"Unsupported Ego-LAP R6 state layout: {state_layout!r}")


def build_lap_state(
    eef_position: Any,
    eef_quaternion_wxyz: Any,
    closed_gripper: Any,
    *,
    state_layout: str,
) -> np.ndarray:
    """Build ``[xyz, rot6d, open_gripper]`` for Ego-LAP."""

    position = _finite_vector(eef_position, name="end-effector position", size=3)
    rot6d = quaternion_wxyz_to_rot6d(
        eef_quaternion_wxyz,
        state_layout=state_layout,
    )
    closed = _finite_vector(closed_gripper, name="closed gripper", size=1)[0]
    # Match the official DROID runner: invert the normalized observation and
    # binarize at 0.5 before building the policy state.
    open_gripper = float((1.0 - np.clip(closed, 0.0, 1.0)) > 0.5)
    state = np.concatenate([position, rot6d, np.array([open_gripper])])
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError(f"Invalid Ego-LAP state with shape {state.shape}")
    return state.astype(np.float32)


def validate_action_chunk(
    response: Any, *, expected_horizon: int | None = None
) -> np.ndarray:
    """Return a finite ``T x 7`` action chunk with the contracted horizon."""

    if not isinstance(response, dict):
        raise TypeError(
            f"Policy response must be a dict; got {type(response).__name__}"
        )
    if "actions" not in response:
        raise KeyError("Policy response is missing the 'actions' field")
    actions = np.asarray(response["actions"], dtype=np.float64)
    if expected_horizon == 1 and actions.shape == (7,):
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] != 7:
        raise ValueError(
            f"Ego-LAP actions must have shape (T, 7), T >= 1; got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("Ego-LAP action chunk contains non-finite values")
    if expected_horizon is not None and actions.shape[0] != expected_horizon:
        raise ValueError(
            "Ego-LAP response horizon does not match serving metadata: "
            f"got {actions.shape[0]}, expected {expected_horizon}"
        )
    return actions


def interpolate_ar_endpoint(endpoint_chunk: Any, *, steps: int = 16) -> np.ndarray:
    """Expand one AR total-delta endpoint into cumulative delta targets.

    Translation and Euler deltas advance linearly from ``1 / steps`` to the
    endpoint. The endpoint gripper command is a target state and is therefore
    held for every interpolated action rather than fractionally scaled.
    """

    endpoint = np.asarray(endpoint_chunk, dtype=np.float64)
    if endpoint.shape != (1, 7):
        raise ValueError(f"AR endpoint must have shape (1, 7); got {endpoint.shape}")
    if steps < 1:
        raise ValueError(f"AR interpolation steps must be positive; got {steps}")
    if not np.isfinite(endpoint).all():
        raise ValueError("AR endpoint contains non-finite values")

    fractions = np.arange(1, steps + 1, dtype=np.float64)[:, None] / steps
    interpolated = np.repeat(endpoint, steps, axis=0)
    interpolated[:, :6] *= fractions
    return interpolated


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


def resize_lap_image(
    image: np.ndarray, target_h: int = LAP_IMAGE_SIZE, target_w: int = LAP_IMAGE_SIZE
) -> np.ndarray:
    """Match Ego-LAP's training-time uint8 ``_tf_resize_with_pad`` path.

    TensorFlow's default bilinear resize uses half-pixel centers and no
    antialiasing. Reproduce its float32 coordinate and interpolation operation
    order directly with NumPy so rounding-edge outputs are bit-exact without
    adding TensorFlow to the PolaRiS runtime. Keep the float32 dimension
    calculation, uint8 rounding, and asymmetric remainder placement aligned
    with training.
    """

    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 1:
        raise ValueError(f"image must have shape (H, W, C>=1); got {image.shape}")
    if image.dtype != np.uint8:
        raise TypeError(f"image must have dtype uint8; got {image.dtype}")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"image dimensions must be nonzero; got {image.shape[:2]}")
    if target_h < 1 or target_w < 1:
        raise ValueError(
            f"target dimensions must be positive; got {(target_h, target_w)}"
        )

    in_h, in_w = image.shape[:2]
    # Use float32 explicitly: this reproduces the TensorFlow graph's dimension
    # arithmetic, including floor behavior at nearly integral boundaries.
    h_f = np.float32(in_h)
    w_f = np.float32(in_w)
    ratio = np.maximum(w_f / np.float32(target_w), h_f / np.float32(target_h))
    resized_h = int(np.floor(h_f / ratio))
    resized_w = int(np.floor(w_f / ratio))
    if resized_h < 1 or resized_w < 1:
        raise ValueError(
            "aspect-preserving resize produced an empty dimension: "
            f"input={(in_h, in_w)}, target={(target_h, target_w)}, "
            f"resized={(resized_h, resized_w)}"
        )

    def half_pixel_indices_and_lerp(
        input_size: int, output_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        output_indices = np.arange(output_size, dtype=np.float32)
        input_positions = (output_indices + np.float32(0.5)) * np.float32(
            input_size / output_size
        ) - np.float32(0.5)
        lower_unclipped = np.floor(input_positions).astype(np.int64)
        lerp = input_positions - lower_unclipped.astype(np.float32)
        lower = np.clip(lower_unclipped, 0, input_size - 1)
        upper = np.clip(lower_unclipped + 1, 0, input_size - 1)
        return lower, upper, lerp

    y0, y1, y_lerp = half_pixel_indices_and_lerp(in_h, resized_h)
    x0, x1, x_lerp = half_pixel_indices_and_lerp(in_w, resized_w)
    top_left = image[y0[:, None], x0[None, :]].astype(np.float32)
    top_right = image[y0[:, None], x1[None, :]].astype(np.float32)
    bottom_left = image[y1[:, None], x0[None, :]].astype(np.float32)
    bottom_right = image[y1[:, None], x1[None, :]].astype(np.float32)
    x_lerp = x_lerp[None, :, None]
    y_lerp = y_lerp[:, None, None]
    top = top_left + (top_right - top_left) * x_lerp
    bottom = bottom_left + (bottom_right - bottom_left) * x_lerp
    resized_f32 = top + (bottom - top) * y_lerp
    resized = np.clip(np.rint(resized_f32), 0.0, 255.0).astype(np.uint8)

    pad_h_total = target_h - resized_h
    pad_w_total = target_w - resized_w
    pad_h0 = pad_h_total // 2
    pad_w0 = pad_w_total // 2
    padded = np.zeros((target_h, target_w, image.shape[2]), dtype=np.uint8)
    padded[
        pad_h0 : pad_h0 + resized_h,
        pad_w0 : pad_w0 + resized_w,
    ] = resized
    return padded


@InferenceClient.register(client_name="EgoLAPEefPose")
class EgoLAPEefPoseClient(InferenceClient):
    """Websocket Ego-LAP client that emits absolute PolaRiS EEF targets."""

    def __init__(self, args: PolicyArgs) -> None:
        if args.open_loop_horizon is not None and args.open_loop_horizon < 1:
            raise ValueError("open_loop_horizon must be positive or None")
        if args.frame_description is not None and not args.frame_description.strip():
            raise ValueError("frame_description must not be empty")
        if args.eef_frame != LAP_EEF_FRAME:
            raise ValueError(
                "EgoLAPEefPose requires the DROID/LAP Cartesian frame "
                f"{LAP_EEF_FRAME!r}; got {args.eef_frame!r}"
            )
        if args.action_frame is not None:
            resolve_action_frame(args.action_frame)
        if args.contract_output is None:
            raise ValueError("EgoLAPEefPose requires --policy.contract-output")

        self.args = args
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        self.contract = validate_ego_lap_server_metadata(
            self.client.get_server_metadata(),
            expected_checkpoint_profile=args.checkpoint_profile,
            expected_checkpoint_path=args.checkpoint_path,
            expected_policy_type=args.policy_type,
            expected_normalization_scope=args.normalization_scope,
            expected_normalization_stats_sha256=args.normalization_stats_sha256,
            expected_normalization_profile=args.normalization_profile,
            expected_normalization_input_formula=args.normalization_input_formula,
            expected_normalization_output_formula=args.normalization_output_formula,
            expected_frame_description=args.frame_description,
            expected_action_frame=args.action_frame,
            expected_dataset_name=args.dataset_name,
            expected_state_type=args.state_type,
            expected_open_loop_horizon=args.open_loop_horizon,
            ar_interpolation_steps=args.ar_interpolation_steps,
        )
        if args.rotate_wrist_180 is not None and args.rotate_wrist_180 is not True:
            raise ValueError(
                "Ego-LAP serving metadata requires a 180-degree wrist rotation"
            )
        persist_ego_lap_contract(self.contract.document, args.contract_output)

        self.policy_type = self.contract.policy_type
        self.frame_description = self.contract.frame_description
        self.action_frame = self.contract.action_frame
        self.dataset_name = self.contract.dataset_name
        self.state_type = self.contract.state_type
        self.state_layout = self.contract.state_layout
        self.state_layout_mode = self.contract.state_layout_mode
        self.rotate_wrist_180 = self.contract.rotate_wrist_180
        self.open_loop_horizon = self.contract.execution_horizon
        print(LAP_IMAGE_PREPROCESSOR_MARKER, flush=True)
        print(LAP_EEF_FRAME_MARKER, flush=True)
        print(
            "POLARIS_LAP_SERVING_CONTRACT="
            f"sha256={self.contract.contract_sha256};"
            f"profile={self.contract.checkpoint_profile};"
            f"checkpoint={self.contract.checkpoint_path};"
            f"policy_type={self.policy_type};"
            f"response={self.contract.response_horizon}x7/{self.contract.response_semantics};"
            f"execute={self.open_loop_horizon};"
            f"normalization={self.contract.normalization_scope}/"
            f"{self.contract.normalization_stats_sha256};"
            f"normalization_profile={self.contract.normalization_profile};"
            f"input_formula={self.contract.normalization_input_formula};"
            f"output_formula={self.contract.normalization_output_formula};"
            f"formula_probe={self.contract.normalization_formula_probe_sha256};"
            f"state_layout={self.state_layout};"
            f"state_layout_mode={self.state_layout_mode};"
            f"polaris_profile={self.contract.polaris_profile}",
            flush=True,
        )
        if args.trace_dir is not None and args.trace_path is not None:
            raise ValueError("Configure either trace_dir or trace_path, not both")
        self.trace_dir = Path(args.trace_dir) if args.trace_dir else None
        self.trace_path = Path(args.trace_path) if args.trace_path else None
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._active_trace_path: Path | None = None
        self._active_trace_final_path: Path | None = None
        self._legacy_trace_reconciled = False

        self.pred_action_chunk: np.ndarray | None = None
        self.server_delta_chunk: np.ndarray | None = None
        self.raw_delta_chunk: np.ndarray | None = None
        self.actions_from_chunk_completed = 0
        self.current_execution_horizon = 0
        self.episode_index = -1
        self.query_index = 0
        self.step_index = 0
        self._image_io_logged = False

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

    def reset(self, episode_index: int | None = None):
        self.pred_action_chunk = None
        self.server_delta_chunk = None
        self.raw_delta_chunk = None
        self.actions_from_chunk_completed = 0
        self.current_execution_horizon = 0
        if episode_index is None:
            episode_index = self.episode_index + 1
        if not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError(
                f"episode_index must be a nonnegative integer; got {episode_index!r}"
            )
        self.episode_index = episode_index
        self.query_index = 0
        self.step_index = 0
        self._begin_episode_trace()
        self._write_trace({"event": "reset", "episode": self.episode_index})

    def finalize_episode(
        self,
        *,
        episode_length: int,
        success: bool,
        progress: float,
        numerical_failure_reason: str = "",
    ) -> Path | None:
        """Atomically publish the active episode trace after rollout artifacts exist."""

        if self.episode_index < 0:
            raise RuntimeError("Cannot finalize an Ego-LAP trace before reset")
        if episode_length != self.step_index:
            raise ValueError(
                "Episode trace length does not match emitted action records: "
                f"episode_length={episode_length}, actions={self.step_index}"
            )
        numerical_failure = bool(numerical_failure_reason)
        self._write_trace(
            {
                "event": "episode_complete",
                "episode": self.episode_index,
                "episode_length": episode_length,
                "status": "numerical_failure" if numerical_failure else "completed",
                "success": bool(success),
                "progress": float(progress),
                "numerical_failure": numerical_failure,
                "numerical_failure_reason": numerical_failure_reason,
            }
        )
        if self.trace_dir is None:
            return self.trace_path
        if self._active_trace_path is None or self._active_trace_final_path is None:
            raise RuntimeError("No active per-episode trace is available to finalize")
        with self._active_trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.flush()
            os.fsync(trace_file.fileno())
        os.replace(self._active_trace_path, self._active_trace_final_path)
        finalized = self._active_trace_final_path
        self._active_trace_path = None
        self._active_trace_final_path = None
        return finalized

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
            server_delta_chunk = validate_action_chunk(
                response,
                expected_horizon=self.contract.response_horizon,
            )
            raw_delta_chunk = (
                interpolate_ar_endpoint(
                    server_delta_chunk,
                    steps=self.contract.interpolation_steps,
                )
                if self.policy_type == "ar"
                else server_delta_chunk
            )
            execution_horizon = self.open_loop_horizon
            if execution_horizon > len(raw_delta_chunk):
                raise ValueError(
                    "open_loop_horizon exceeds returned action chunk: "
                    f"{execution_horizon} > {len(raw_delta_chunk)}"
                )

            anchored_chunk = anchor_action_chunk(
                raw_delta_chunk,
                current["eef_position"],
                current["eef_quaternion_wxyz"],
                action_frame=self.action_frame,
                dataset_name=self.dataset_name,
                rotation_applied=self.rotate_wrist_180,
            )
            action_frame = resolve_action_frame(self.action_frame)
            base_delta_chunk = (
                raw_delta_chunk
                if action_frame == "robot_base"
                else egocentric_action_chunk_to_base(
                    raw_delta_chunk,
                    current["eef_quaternion_wxyz"],
                    dataset_name=self.dataset_name,
                    rotation_applied=self.rotate_wrist_180,
                )
            )
            self.server_delta_chunk = server_delta_chunk
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
                    "checkpoint_profile": self.contract.checkpoint_profile,
                    "checkpoint_path": self.contract.checkpoint_path,
                    "contract_sha256": self.contract.contract_sha256,
                    "policy_type": self.policy_type,
                    "response_semantics": self.contract.response_semantics,
                    "frame_description": self.frame_description,
                    "eef_frame": self.args.eef_frame,
                    "numeric_action_frame": action_frame,
                    "normalization_scope": self.contract.normalization_scope,
                    "normalization_stats_sha256": self.contract.normalization_stats_sha256,
                    "normalization_profile": self.contract.normalization_profile,
                    "normalization_input_formula": self.contract.normalization_input_formula,
                    "normalization_output_formula": self.contract.normalization_output_formula,
                    "normalization_formula_probe_sha256": self.contract.normalization_formula_probe_sha256,
                    "state_layout": self.state_layout,
                    "state_layout_mode": self.state_layout_mode,
                    "polaris_profile": self.contract.polaris_profile,
                    "anchor_position": current["eef_position"].tolist(),
                    "anchor_quaternion_wxyz": current["eef_quaternion_wxyz"].tolist(),
                    "state": request["observation"]["state"].tolist(),
                    "server_delta_chunk": server_delta_chunk.tolist(),
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
        exterior_image = resize_lap_image(current["external_image"])
        wrist_image = current["wrist_image"]
        if self.rotate_wrist_180:
            wrist_image = rotate_image_180(wrist_image)
        wrist_image = resize_lap_image(wrist_image)
        if not self._image_io_logged:
            image_io = {
                "external_input": {
                    "shape": list(current["external_image"].shape),
                    "dtype": str(current["external_image"].dtype),
                },
                "wrist_input": {
                    "shape": list(current["wrist_image"].shape),
                    "dtype": str(current["wrist_image"].dtype),
                },
                "external_output": {
                    "shape": list(exterior_image.shape),
                    "dtype": str(exterior_image.dtype),
                },
                "wrist_output": {
                    "shape": list(wrist_image.shape),
                    "dtype": str(wrist_image.dtype),
                },
            }
            print(
                "POLARIS_LAP_IMAGE_IO=" + json.dumps(image_io, separators=(",", ":")),
                flush=True,
            )
            self._image_io_logged = True
        return exterior_image, wrist_image

    def _build_request(
        self, current: dict[str, np.ndarray], instruction: str
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        exterior_image, wrist_image = self._model_images(current)
        state = build_lap_state(
            current["eef_position"],
            current["eef_quaternion_wxyz"],
            current["closed_gripper"],
            state_layout=self.state_layout,
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
            "frame_description": self.frame_description,
            "eef_frame": self.args.eef_frame,
            "dataset_name": self.dataset_name,
            "state_type": self.state_type,
            "has_wrist_image": True,
            "is_bimanual": False,
            "rotation_applied": self.rotate_wrist_180,
        }
        return request, exterior_image, wrist_image

    def _write_trace(self, record: dict[str, Any]) -> None:
        path = (
            self._active_trace_path if self.trace_dir is not None else self.trace_path
        )
        if path is None:
            return
        payload = {"timestamp": time.time(), **record}
        with path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(
                json.dumps(payload, separators=(",", ":"), default=_json_default) + "\n"
            )

    def _begin_episode_trace(self) -> None:
        if self.trace_dir is not None:
            filename = f"episode_{self.episode_index:06d}.jsonl"
            self._active_trace_final_path = self.trace_dir / filename
            self._active_trace_path = self.trace_dir / f".{filename}.tmp"
            # Requeue reconciliation: an unfinished episode never owns a final
            # marker, and its stable hidden temporary is replaced from scratch.
            self._active_trace_path.write_text("", encoding="utf-8")
            return
        if self.trace_path is not None and not self._legacy_trace_reconciled:
            self._reconcile_legacy_trace(self.episode_index)
            self._legacy_trace_reconciled = True

    def _reconcile_legacy_trace(self, resume_episode: int) -> None:
        """Drop a partial legacy JSONL suffix before resuming a global episode."""

        if self.trace_path is None or not self.trace_path.exists():
            return
        retained: list[str] = []
        for line in self.trace_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                record_episode = (
                    record.get("episode") if isinstance(record, dict) else None
                )
            except json.JSONDecodeError:
                continue
            if isinstance(record_episode, int) and record_episode < resume_episode:
                retained.append(line)
        temporary = self.trace_path.with_name(
            f".{self.trace_path.name}.{os.getpid()}.tmp"
        )
        try:
            temporary.write_text(
                "" if not retained else "\n".join(retained) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, self.trace_path)
        finally:
            temporary.unlink(missing_ok=True)
