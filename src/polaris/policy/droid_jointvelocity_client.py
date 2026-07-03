"""Official OpenPI :math:`pi0.5` DROID native joint-velocity client."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from polaris.pi05_droid_jointvelocity_contract import (
    PANDA_ARM_JOINT_NAMES,
    PI05_DROID_JOINTVELOCITY_PROFILE,
    validate_pi05_droid_server_metadata,
)
from polaris.policy.abstract_client import InferenceClient, PolicyArgs
from polaris.policy.droid_jointpos_client import (
    _latest_trace_reset_index,
    validate_joint_action_chunk,
)


PI05_DROID_JOINTVELOCITY_CONTRACT_MARKER = "POLARIS_PI05_DROID_JOINTVELOCITY_CONTRACT="


class JointVelocityObservationNumericalError(FloatingPointError):
    """Raised when simulator joint-velocity proprioception is invalid."""


def _image_contract(image: np.ndarray) -> dict[str, Any]:
    image = np.ascontiguousarray(np.asarray(image))
    if image.shape != (224, 224, 3) or image.dtype != np.uint8:
        raise ValueError(
            "pi0.5-DROID images must be 224x224 uint8 RGB; "
            f"got shape={image.shape}, dtype={image.dtype}"
        )
    return {
        "shape": [224, 224, 3],
        "dtype": "uint8",
        "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def process_native_jointvelocity_action(
    raw_action: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply upstream DROID gripper binarization, then clip all eight values."""

    raw_action = np.asarray(raw_action)
    if raw_action.shape != (8,):
        raise ValueError(f"Expected one 8-D DROID action, got {raw_action.shape}")
    if not np.isfinite(raw_action).all():
        raise ValueError("DROID joint-velocity action contains non-finite values")
    # This intentionally matches examples/droid/main.py.  np.ones/np.zeros
    # preserve the upstream float64 promotion before the all-dimension clip.
    if raw_action[-1].item() > 0.5:
        binary_action = np.concatenate([raw_action[:-1], np.ones((1,))])
    else:
        binary_action = np.concatenate([raw_action[:-1], np.zeros((1,))])
    clipped_action = np.clip(binary_action, -1.0, 1.0)
    return binary_action, clipped_action


def _tensor_numpy(value: Any, *, field: str) -> np.ndarray:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must be numeric, got {array.dtype}")
    return array


@InferenceClient.register(client_name="DroidJointVelocity")
class DroidJointVelocityClient(InferenceClient):
    """Serve the immutable official ``pi05_droid`` velocity contract."""

    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self._validate_args()
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        self.serving_contract = validate_pi05_droid_server_metadata(
            self.client.get_server_metadata()
        )
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk: np.ndarray | None = None
        self.open_loop_horizon = args.open_loop_horizon
        self.query_index = 0
        self.active_query_index: int | None = None
        self.trace_path = Path(args.trace_path) if args.trace_path else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self.reset_index = _latest_trace_reset_index(self.trace_path)
        else:
            self.reset_index = -1
        self._pending_execution: dict[str, Any] | None = None

        marker = {
            "client": "DroidJointVelocity",
            "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
            "serving_contract_sha256": self.serving_contract["contract_sha256"],
            "open_loop_horizon": self.open_loop_horizon,
            "expected_action_horizon": args.expected_action_horizon,
            "expected_action_dim": args.expected_action_dim,
            "image_resolution": [224, 224],
            "wrist_rotation_degrees": 0,
            "initial_reset_index": self.reset_index,
        }
        print(
            PI05_DROID_JOINTVELOCITY_CONTRACT_MARKER
            + json.dumps(marker, sort_keys=True),
            flush=True,
        )

    def _validate_args(self) -> None:
        expected = {
            "policy_profile": PI05_DROID_JOINTVELOCITY_PROFILE,
            "open_loop_horizon": 8,
            "expected_action_horizon": 15,
            "expected_action_dim": 8,
            "state_type": "joint_position",
            "frame_description": "robot base frame",
            "action_frame": "robot_base",
            "dataset_name": "droid",
            "rotate_wrist_180": False,
            "render_every_step": True,
        }
        for field, expected_value in expected.items():
            actual = getattr(self.args, field, None)
            if actual != expected_value:
                raise ValueError(
                    f"DroidJointVelocity requires {field}={expected_value!r}; "
                    f"got {actual!r}"
                )

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def visualize(self, request: dict) -> np.ndarray:
        current = self._extract_observation(request)
        external, wrist = self._resize_images(current)
        return np.concatenate([external, wrist], axis=1)

    def reset(self) -> None:
        if self._pending_execution is not None:
            raise RuntimeError(
                "Cannot reset DroidJointVelocity before recording the prior execution"
            )
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.reset_index += 1
        self.query_index = 0
        self.active_query_index = None

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self._pending_execution is not None:
            raise RuntimeError(
                "Previous DroidJointVelocity action has no measured execution record"
            )
        current = self._extract_observation(obs)
        visualization = None
        if self.rerender:
            self.actions_from_chunk_completed = 0
            external, wrist = self._resize_images(current)
            request_data = {
                "observation/exterior_image_1_left": external,
                "observation/wrist_image_left": wrist,
                "observation/joint_position": current["joint_position"],
                "observation/gripper_position": current["gripper_position"],
                "prompt": instruction,
            }
            response = self.client.infer(request_data)
            self.pred_action_chunk = validate_joint_action_chunk(
                response,
                open_loop_horizon=self.open_loop_horizon,
                expected_action_horizon=15,
                expected_action_dim=8,
            )
            if self.pred_action_chunk.dtype != np.float64:
                raise ValueError(
                    "Official pi05_droid response must be float64 after checkpoint "
                    f"unnormalization; got {self.pred_action_chunk.dtype}"
                )
            self.active_query_index = self.query_index
            self._trace_query(request_data, self.pred_action_chunk)
            self.query_index += 1
            visualization = np.concatenate([external, wrist], axis=1)
        elif return_viz:
            external, wrist = self._resize_images(current)
            visualization = np.concatenate([external, wrist], axis=1)

        if self.pred_action_chunk is None or self.active_query_index is None:
            raise ValueError("No pi0.5-DROID action chunk predicted")

        action_index = self.actions_from_chunk_completed
        raw_action = self.pred_action_chunk[action_index].copy()
        binary_action, clipped_action = process_native_jointvelocity_action(raw_action)
        self.actions_from_chunk_completed += 1
        self._pending_execution = {
            "query_index": self.active_query_index,
            "chunk_action_index": action_index,
            "raw_action": raw_action,
            "binary_action": binary_action,
            "clipped_action": clipped_action,
            "pre_joint_position": current["joint_position"],
            "pre_joint_velocity": current["joint_velocity"],
        }
        self._trace_emitted_action()
        return clipped_action, visualization

    def record_execution(self, obs: dict, env: Any) -> None:
        """Validate live velocity targets and trace measured post-step q/dq."""

        if self._pending_execution is None:
            raise RuntimeError("No pending DroidJointVelocity action to record")
        current = self._extract_observation(obs)
        root_env = getattr(env, "unwrapped", env)
        arm_term = root_env.action_manager._terms["arm"]
        robot = root_env.scene["robot"]
        joint_ids, joint_names = robot.find_joints(
            list(PANDA_ARM_JOINT_NAMES), preserve_order=True
        )
        if tuple(joint_names) != PANDA_ARM_JOINT_NAMES:
            raise ValueError(f"Live Panda joint order mismatch: {joint_names}")

        processed = _tensor_numpy(
            arm_term.processed_actions, field="arm processed actions"
        )
        targets = _tensor_numpy(
            robot.data.joint_vel_target[:, joint_ids], field="joint velocity targets"
        )
        if processed.dtype != np.float32 or targets.dtype != np.float32:
            raise ValueError(
                "Live processed and target velocities must both be float32; "
                f"got {processed.dtype} and {targets.dtype}"
            )
        if processed.shape != (1, 7) or targets.shape != (1, 7):
            raise ValueError(
                "Live velocity target shape mismatch: "
                f"processed={processed.shape}, target={targets.shape}"
            )
        expected = np.asarray(
            self._pending_execution["clipped_action"][:7], dtype=targets.dtype
        )[None, :]
        if not np.array_equal(processed.astype(targets.dtype, copy=False), expected):
            raise ValueError("Action manager changed emitted joint velocity")
        if not np.array_equal(targets, expected):
            raise ValueError("Articulation joint-velocity target differs from emission")

        if self.trace_path is not None:
            pending = self._pending_execution
            self._append_trace(
                {
                    "schema_version": 1,
                    "record_type": "openpi_joint_velocity_execution",
                    "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                    "serving_contract_sha256": self.serving_contract["contract_sha256"],
                    "reset_index": self.reset_index,
                    "query_index": pending["query_index"],
                    "chunk_action_index": pending["chunk_action_index"],
                    "processed_joint_velocity": processed[0].tolist(),
                    "articulation_joint_velocity_target": targets[0].tolist(),
                    "measured_joint_position_after": current["joint_position"].tolist(),
                    "measured_joint_velocity_after": current["joint_velocity"].tolist(),
                }
            )
        self._pending_execution = None

    def _resize_images(
        self, current: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        external = image_tools.resize_with_pad(current["right_image"], 224, 224)
        wrist = image_tools.resize_with_pad(current["wrist_image"], 224, 224)
        _image_contract(external)
        _image_contract(wrist)
        return external, wrist

    def _trace_query(self, request: dict, action_chunk: np.ndarray) -> None:
        if self.trace_path is None:
            return
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        planned_binary = []
        planned_clipped = []
        for raw_action in action_chunk[: self.open_loop_horizon]:
            binary, clipped = process_native_jointvelocity_action(raw_action)
            planned_binary.append(binary.tolist())
            planned_clipped.append(clipped.tolist())
        self._append_trace(
            {
                "schema_version": 1,
                "record_type": "openpi_joint_velocity_query",
                "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                "serving_contract_sha256": self.serving_contract["contract_sha256"],
                "reset_index": self.reset_index,
                "query_index": self.query_index,
                "prompt": request["prompt"],
                "state": {
                    "joint_position": np.asarray(
                        request["observation/joint_position"]
                    ).tolist(),
                    "gripper_position": np.asarray(
                        request["observation/gripper_position"]
                    ).tolist(),
                },
                "images": {
                    "external": _image_contract(
                        request["observation/exterior_image_1_left"]
                    ),
                    "wrist": _image_contract(request["observation/wrist_image_left"]),
                    "blank_masked_right_wrist": _image_contract(blank),
                    "model_order": [
                        "base_0_rgb",
                        "left_wrist_0_rgb",
                        "right_wrist_0_rgb_masked",
                    ],
                    "wrist_rotation_degrees": 0,
                },
                "response_action_shape": list(action_chunk.shape),
                "response_action_chunk_raw": action_chunk.tolist(),
                "execution_horizon": self.open_loop_horizon,
                "planned_action_chunk_binary_gripper": planned_binary,
                "planned_action_chunk_clipped": planned_clipped,
            }
        )

    def _trace_emitted_action(self) -> None:
        if self.trace_path is None:
            return
        if self._pending_execution is None:
            raise RuntimeError("Cannot trace an action without pending execution")
        pending = self._pending_execution
        self._append_trace(
            {
                "schema_version": 1,
                "record_type": "openpi_joint_velocity_action",
                "profile": PI05_DROID_JOINTVELOCITY_PROFILE,
                "serving_contract_sha256": self.serving_contract["contract_sha256"],
                "reset_index": self.reset_index,
                "query_index": pending["query_index"],
                "chunk_action_index": pending["chunk_action_index"],
                "raw_action": np.asarray(pending["raw_action"]).tolist(),
                "binary_gripper_action": np.asarray(pending["binary_action"]).tolist(),
                "clipped_action": np.asarray(pending["clipped_action"]).tolist(),
                "emitted_joint_velocity": np.asarray(
                    pending["clipped_action"][:7]
                ).tolist(),
                "emitted_gripper_closed": float(pending["clipped_action"][7]),
                "measured_joint_position_before": np.asarray(
                    pending["pre_joint_position"]
                ).tolist(),
                "measured_joint_velocity_before": np.asarray(
                    pending["pre_joint_velocity"]
                ).tolist(),
            }
        )

    def _append_trace(self, record: dict[str, Any]) -> None:
        if self.trace_path is None:
            return
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(
                json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
            )

    def _extract_observation(self, obs_dict: dict) -> dict[str, np.ndarray]:
        right_image = np.asarray(obs_dict["splat"]["external_cam"])
        wrist_image = np.asarray(obs_dict["splat"]["wrist_cam"])
        state = obs_dict["policy"]
        joint_position = _tensor_numpy(
            state["arm_joint_pos"], field="arm joint position"
        )[0]
        joint_velocity = _tensor_numpy(
            state["arm_joint_vel"], field="arm joint velocity"
        )[0]
        gripper_position = _tensor_numpy(
            state["gripper_pos"], field="gripper position"
        )[0]
        expected_shapes = {
            "joint position": (joint_position, (7,)),
            "joint velocity": (joint_velocity, (7,)),
            "gripper position": (gripper_position, (1,)),
        }
        for field, (array, shape) in expected_shapes.items():
            if array.shape != shape:
                raise ValueError(f"Expected {field} shape {shape}, got {array.shape}")
            if array.dtype != np.float32:
                raise ValueError(f"DROID {field} must be float32, got {array.dtype}")
            if not np.isfinite(array).all():
                raise JointVelocityObservationNumericalError(
                    f"DROID {field} contains non-finite values"
                )
        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "gripper_position": gripper_position,
        }
