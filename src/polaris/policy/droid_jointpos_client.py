import hashlib
import json
from pathlib import Path

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from polaris.policy.abstract_client import InferenceClient, PolicyArgs


PI05_DROID_CONTRACT_MARKER = "POLARIS_PI05_DROID_CONTRACT="


class JointPositionObservationNumericalError(FloatingPointError):
    """Raised when simulator corruption makes joint proprioception non-finite."""


def _image_contract(image: np.ndarray) -> dict:
    image = np.ascontiguousarray(np.asarray(image))
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    }


def validate_joint_action_chunk(
    response: dict,
    *,
    open_loop_horizon: int,
    expected_action_horizon: int | None = None,
    expected_action_dim: int | None = None,
) -> np.ndarray:
    """Validate the absolute joint-position chunk returned by OpenPI."""
    if "actions" not in response:
        raise KeyError("OpenPI response is missing 'actions'")
    actions = np.asarray(response["actions"])
    if actions.ndim != 2:
        raise ValueError(
            f"Expected OpenPI actions with shape (T, D), got {actions.shape}"
        )
    if (
        expected_action_horizon is not None
        and actions.shape[0] != expected_action_horizon
    ):
        raise ValueError(
            "OpenPI action horizon mismatch: "
            f"expected {expected_action_horizon}, got {actions.shape[0]}"
        )
    if expected_action_dim is not None and actions.shape[1] != expected_action_dim:
        raise ValueError(
            f"OpenPI action width mismatch: expected {expected_action_dim}, got {actions.shape[1]}"
        )
    if actions.shape[0] < open_loop_horizon:
        raise ValueError(
            "OpenPI action chunk is shorter than the requested execution horizon: "
            f"{actions.shape[0]} < {open_loop_horizon}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("OpenPI action chunk contains non-finite values")
    return actions


def _binarize_gripper(action_chunk: np.ndarray) -> np.ndarray:
    action_chunk = np.asarray(action_chunk)
    # Preserve the frozen client's float64 promotion from concatenating the
    # server action with np.ones/np.zeros while applying the same >0.5 rule.
    gripper = np.where(action_chunk[..., -1:] > 0.5, 1.0, 0.0)
    return np.concatenate([action_chunk[..., :-1], gripper], axis=-1)


# Joint Position Client for DROID
@InferenceClient.register(client_name="DroidJointPos")
class DroidJointPosClient(InferenceClient):
    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        if args.open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for DroidJointPosClient")

        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.open_loop_horizon = args.open_loop_horizon
        self.reset_index = -1
        self.query_index = 0
        self.active_query_index = None
        self.trace_path = Path(args.trace_path) if args.trace_path else None
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        contract = {
            "client": "DroidJointPos",
            "state": "7_panda_joint_positions_radians_plus_closed_positive_gripper",
            "action": "7_absolute_panda_joint_targets_radians_plus_closed_positive_gripper",
            "image_slots": [
                "base_0_rgb",
                "left_wrist_0_rgb",
                "right_wrist_0_rgb_masked",
            ],
            "image_resolution": [224, 224],
            "wrist_rotation_degrees": 0,
            "open_loop_horizon": self.open_loop_horizon,
            "expected_action_horizon": args.expected_action_horizon,
            "expected_action_dim": args.expected_action_dim,
            "server_metadata": self.client.get_server_metadata(),
        }
        print(
            PI05_DROID_CONTRACT_MARKER + json.dumps(contract, sort_keys=True),
            flush=True,
        )

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        base_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        combined = np.concatenate([base_img, wrist_img], axis=1)
        return combined

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.reset_index += 1
        self.query_index = 0
        self.active_query_index = None

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Infer the next action from the policy in a server-client setup
        """
        both = None
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            curr_obs = self._extract_observation(obs)

            self.actions_from_chunk_completed = 0
            exterior_image = image_tools.resize_with_pad(
                curr_obs["right_image"], 224, 224
            )
            wrist_image = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
            request_data = {
                "observation/exterior_image_1_left": exterior_image,
                "observation/wrist_image_left": wrist_image,
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            server_response = self.client.infer(request_data)
            self.pred_action_chunk = validate_joint_action_chunk(
                server_response,
                open_loop_horizon=self.open_loop_horizon,
                expected_action_horizon=self.args.expected_action_horizon,
                expected_action_dim=self.args.expected_action_dim,
            )
            self.active_query_index = self.query_index
            self._trace_query(request_data, self.pred_action_chunk)
            self.query_index += 1
            both = np.concatenate([exterior_image, wrist_image], axis=1)

        if return_viz and both is None:
            curr_obs = self._extract_observation(obs)
            both = np.concatenate(
                [
                    image_tools.resize_with_pad(curr_obs["right_image"], 224, 224),
                    image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
                ],
                axis=1,
            )

        if self.pred_action_chunk is None:
            raise ValueError("No action chunk predicted")

        action_index = self.actions_from_chunk_completed
        raw_action = self.pred_action_chunk[action_index].copy()
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        action = _binarize_gripper(raw_action)
        self._trace_emitted_action(raw_action, action, action_index)

        return action, both

    def _trace_query(self, request: dict, action_chunk: np.ndarray) -> None:
        if self.trace_path is None:
            return
        planned_action_chunk = _binarize_gripper(action_chunk[: self.open_loop_horizon])
        record = {
            "schema_version": 1,
            "record_type": "openpi_joint_position_query",
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
                "wrist_rotation_degrees": 0,
            },
            "response_action_shape": list(action_chunk.shape),
            "response_action_chunk": action_chunk.tolist(),
            "execution_horizon": self.open_loop_horizon,
            "planned_action_chunk": planned_action_chunk.tolist(),
        }
        self._append_trace(record)

    def _trace_emitted_action(
        self, raw_action: np.ndarray, emitted_action: np.ndarray, action_index: int
    ) -> None:
        if self.trace_path is None:
            return
        if self.active_query_index is None:
            raise RuntimeError("Cannot trace an emitted action without an active query")
        record = {
            "schema_version": 1,
            "record_type": "openpi_joint_position_action",
            "reset_index": self.reset_index,
            "query_index": self.active_query_index,
            "chunk_action_index": action_index,
            "raw_action": np.asarray(raw_action).tolist(),
            "emitted_action": np.asarray(emitted_action).tolist(),
        }
        self._append_trace(record)

    def _append_trace(self, record: dict) -> None:
        if self.trace_path is None:
            return
        with self.trace_path.open("a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _extract_observation(self, obs_dict):
        # Assign images
        right_image = obs_dict["splat"]["external_cam"]
        wrist_image = obs_dict["splat"]["wrist_cam"]

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]
        if joint_position.shape != (7,):
            raise ValueError(
                f"Expected seven Panda joint positions, got {joint_position.shape}"
            )
        if gripper_position.shape != (1,):
            raise ValueError(
                f"Expected one gripper position, got {gripper_position.shape}"
            )
        if (
            not np.isfinite(joint_position).all()
            or not np.isfinite(gripper_position).all()
        ):
            raise JointPositionObservationNumericalError(
                "Joint-position observation contains non-finite values"
            )

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
