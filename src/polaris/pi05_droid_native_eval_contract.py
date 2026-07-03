"""Pure contracts for the official pi0.5-DROID canary.

This module intentionally has no Isaac Lab, JAX, or Torch dependency.  It is
shared by the simulator child, host finalizer, and host-runnable tests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

from polaris.pi05_droid_jointvelocity_contract import (
    NATIVE_GRIPPER_DRIVE_PROFILE,
    PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
    PI05_DROID_CHECKPOINT_URI,
    PI05_DROID_NORM_STATS_SHA256,
)


PI05_DROID_NATIVE_CANARY_PROFILE = (
    "openpi_pi05_droid_native_jointvelocity_polaris_canary_v1"
)
PI05_DROID_NATIVE_TASK = "DROID-FoodBussing"
PI05_DROID_NATIVE_ROLLOUTS = 1
PI05_DROID_NATIVE_EPISODE_STEPS = 450
PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS = 451
PI05_DROID_NATIVE_POLICY_HZ = 15
PI05_DROID_NATIVE_PHYSICS_HZ = 120
PI05_DROID_NATIVE_DECIMATION = 8
PI05_DROID_NATIVE_RESPONSE_HORIZON = 15
PI05_DROID_NATIVE_EXECUTION_HORIZON = 8
PI05_DROID_NATIVE_ACTION_WIDTH = 8
PI05_DROID_NATIVE_TRACE_SCHEMA_VERSION = 2
PI05_DROID_NATIVE_VIDEO_WIDTH = 448
PI05_DROID_NATIVE_VIDEO_HEIGHT = 224
PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS = (
    PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS / PI05_DROID_NATIVE_POLICY_HZ
)
PI05_DROID_NATIVE_ENVIRONMENT_RUNTIME_PROFILE = (
    "pi05_droid_outer450_internal_timeout451_no_autoreset_v1"
)
PI05_DROID_NATIVE_SENSOR_LIVENESS_PROFILE = (
    "isaaclab_camera_frame_counter_exact_policy_step_increment_v1"
)
PI05_DROID_NATIVE_SENSOR_NAMES = ("external_cam", "wrist_cam")
PI05_DROID_NATIVE_NORM_REFERENCE_PROBES = {
    "actions_q01_first8": [
        -0.45799999999999996,
        -0.8076,
        -0.44719999999999993,
        -0.9268,
        -0.6456,
        -0.6459999999999999,
        -0.7616,
        0.0,
    ],
    "actions_q99_first8": [
        0.4476,
        0.7652000000000001,
        0.4480000000000002,
        0.7944,
        0.6484000000000001,
        0.6628000000000001,
        0.7344000000000002,
        0.9998,
    ],
    "state_q01_first8": [
        -0.8279732212066653,
        -0.8398311847686768,
        -0.8425482082366944,
        -2.773015278291702,
        -1.8426181347846986,
        1.1716566389799117,
        -2.047264838027954,
        0.0,
    ],
    "state_q99_first8": [
        0.8996522880554196,
        1.385467470359802,
        0.6920277433395388,
        -0.4542043057203293,
        1.732314240932464,
        3.4672964780330657,
        2.1984972072601314,
        0.991,
    ],
}
PI05_DROID_NATIVE_TRANSFORM_RUNTIME_CONTRACT = {
    "asset_id": "droid",
    "use_quantile_norm": True,
    "repack_inputs": [],
    "repack_outputs": [],
    "data_inputs": ["openpi.policies.droid_policy.DroidInputs"],
    "data_outputs": ["openpi.policies.droid_policy.DroidOutputs"],
    "model_inputs": [
        "openpi.transforms.InjectDefaultPrompt",
        "openpi.transforms.ResizeImages",
        "openpi.transforms.TokenizePrompt",
        "openpi.transforms.PadStatesAndActions",
    ],
    "model_outputs": [],
    "sequence_types": {
        "repack_inputs": "tuple",
        "repack_outputs": "tuple",
        "data_inputs": "list",
        "data_outputs": "list",
        "model_inputs": "list",
        "model_outputs": "tuple",
    },
    "droid_input_model_type": "pi05",
    "resize": [224, 224],
    "tokenizer": "openpi.models.tokenizer.PaligemmaTokenizer",
    "discrete_state_input": True,
    "model_action_dim": 32,
    "forbidden_transforms_absent": [
        "openpi.transforms.DeltaActions",
        "openpi.transforms.AbsoluteActions",
    ],
    "output_projection": "DroidOutputs_leading8",
}

# This is deliberately an exact value, not a configurable dataset/category
# selector.  The released checkpoint carries global DROID statistics at this
# path; substituting OXE ``single_arm`` statistics would be a train/eval
# mismatch even though PolaRiS itself is a single-arm simulator.
PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT = {
    "schema_version": 1,
    "checkpoint": {
        "uri": PI05_DROID_CHECKPOINT_URI,
        "content_manifest_sha256": PI05_DROID_CHECKPOINT_MANIFEST_SHA256,
    },
    "normalization": {
        "asset_id": "droid",
        "scope": "checkpoint_global_droid",
        "path": "assets/droid/norm_stats.json",
        "sha256": PI05_DROID_NORM_STATS_SHA256,
        "category_override": "forbidden",
        "rejected_category_substitutions": ["single_arm", "single-arm", "single arm"],
        "reference_probes": PI05_DROID_NATIVE_NORM_REFERENCE_PROBES,
    },
    "policy_input": {
        "state": "7_panda_joint_positions_radians_plus_closed_positive_gripper",
        "state_width": 8,
        "request_state_dtype": "float32",
        "resize": "openpi_image_tools_resize_with_pad_224_v1",
        "resize_semantics": "aspect_preserving_letterbox_with_zero_padding",
        "images": [
            {
                "source": "external",
                "model_slot": "base_0_rgb",
                "shape": [224, 224, 3],
                "dtype": "uint8",
                "masked": False,
                "rotation_degrees": 0,
            },
            {
                "source": "wrist",
                "model_slot": "left_wrist_0_rgb",
                "shape": [224, 224, 3],
                "dtype": "uint8",
                "masked": False,
                "rotation_degrees": 0,
            },
            {
                "source": "zero_blank",
                "model_slot": "right_wrist_0_rgb",
                "shape": [224, 224, 3],
                "dtype": "uint8",
                "masked": True,
                "rotation_degrees": 0,
            },
        ],
    },
    "policy_output": {
        "response_shape": [15, 8],
        "execute_first": 8,
        "policy_frequency_hz": 15,
        "arm": "panda_joint1_through_7_velocity_radians_per_second",
        "gripper": "absolute_closed_positive_binarize_gt_0p5_else_open",
        "action_transform": "none_before_DroidOutputs_leading8_projection",
        "forbidden_action_transforms": [
            "DeltaActions",
            "AbsoluteActions",
            "joint_position_action_interpretation",
        ],
    },
}

PI05_DROID_BASE_CONTROLLER_COMPLETION_PATH = (
    "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris-pi05-jointvelocity/"
    "controller-smoke/90d56b3b8d0a-controller-only-v3/"
    "controller-smoke-1098174.completion.json"
)
PI05_DROID_BASE_CONTROLLER_COMPLETION_SHA256 = (
    "05403d0aabf3ebc8111cecf64d33f56f50a3a5673e7a84653ae096e7f4027ad3"
)
PI05_DROID_BASE_CONTROLLER_COMPLETION_SIZE = 13_947
PI05_DROID_BASE_CONTROLLER_SOURCE_COMMIT = "90d56b3b8d0a93ad7c48319a377d325790b89144"
PI05_DROID_BASE_CONTROLLER_RUNTIME_SHA256 = (
    "495ce92226ad0d1840138fc2b315fc2531d0ff50953fb16d70172080a8ee0b71"
)

PI05_DROID_CONTROLLER_JOB_ID = 1098174
PI05_DROID_ALL_SIX_CONTROLLER_JOB_ID = 1098349
PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_PATH = (
    "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris-pi05-native/"
    "all-six-controller-smoke/20260703T201500Z-93083d2-all6-smoke2/"
    "native-all-six-smoke-1098349.completion.json"
)
PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SHA256 = (
    "a03ffbf0745327ce604db7be5928a94dce910eac91a8766c21f8efcb71fea867"
)
PI05_DROID_ALL_SIX_CONTROLLER_COMPLETION_SIZE = 7_502
PI05_DROID_ALL_SIX_CONTROLLER_SOURCE_COMMIT = "93083d2694b8638de30e970e3bea450526593e7e"
PI05_DROID_ALL_SIX_CONTROLLER_PROFILE = (
    "pi05_droid_native_all_six_l40s_controller_smoke_v1"
)
PI05_DROID_GRIPPER_DRIVE_PROFILE = NATIVE_GRIPPER_DRIVE_PROFILE
PI05_DROID_ALL_SIX_RUNTIME_SHA256 = (
    "9a0597d62debc01fbde064360f9845a28a2df06fd2853ff0b3556dff48c14efc"
)
PI05_DROID_PYXIS_SHA256 = (
    "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"
)
PI05_DROID_HUB_REVISION = "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b"
PI05_DROID_CANARY_ASSETS = {
    "food_bussing/initial_conditions.json": {
        "sha256": "40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de",
        "metadata_sha256": "852dd0345afb7e4d0c7526b5c327086b5132c40624ed97ff6942962126e90534",
    },
    "food_bussing/scene.usda": {
        "sha256": "82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489",
        "metadata_sha256": "accd9b67e90e510eb4ed44a789b9169df058e71ce557164f960de2d62a840e63",
    },
    "nvidia_droid/noninstanceable.usd": {
        "sha256": "d8379925b103963dbf3e7c85bcc4ae101b81b7c1d7dabe7d2e964f41d069ec44",
        "metadata_sha256": "208e0f85fc16fa32ffeca972aea0fd1b33b0c6c2a582e89ff3877823291a7754",
    },
}

# These are the exact controller-semantic files attested by job 1098174.  An
# evaluation-only descendant commit is acceptable only if every digest remains
# identical to the completed controller attestation.
PI05_DROID_CONTROLLER_CRITICAL_PATHS = (
    "scripts/smoke_joint_velocity_controller.py",
    "scripts/polaris/finalize_pi05_droid_jointvelocity_controller_smoke.py",
    "scripts/polaris/l40s_pi05_droid_jointvelocity_controller_smoke.sbatch",
    "scripts/polaris/submit_pi05_droid_jointvelocity_controller_smoke.sh",
    "src/polaris/environments/droid_cfg.py",
    "src/polaris/environments/robot_cfg.py",
    "src/polaris/joint_velocity_runtime.py",
    "src/polaris/joint_velocity_smoke.py",
    "src/polaris/pi05_droid_jointvelocity_contract.py",
)

# These are the exact controller/runtime sources independently exercised by
# accepted all-six job 1098349.  A model-evaluation descendant may change its
# evaluation-only contract/launcher, but every path below must remain byte
# identical to this gate.
PI05_DROID_ALL_SIX_CONTROLLER_CRITICAL_PATHS = (
    "scripts/eval.py",
    "scripts/smoke_pi05_native_all_six_controller.py",
    "scripts/polaris/finalize_pi05_native_all_six_controller_smoke.py",
    "scripts/polaris/l40s_pi05_native_all_six_controller_smoke.sbatch",
    "scripts/polaris/submit_pi05_native_all_six_controller_smoke.sh",
    "src/polaris/environments/droid_cfg.py",
    "src/polaris/environments/manager_based_rl_splat_environment.py",
    "src/polaris/environments/robot_cfg.py",
    "src/polaris/joint_velocity_runtime.py",
    "src/polaris/native_all_six_smoke.py",
    "src/polaris/native_gripper_runtime.py",
    "src/polaris/pi05_droid_jointvelocity_contract.py",
)

# The accepted smoke proved these policy-facing paths were still identical to
# the integrated official-model base.  The evaluation contract itself is
# intentionally excluded because this descendant changes only its pinned gate.
PI05_DROID_ALL_SIX_UNCHANGED_POLICY_IO_PATHS = (
    "scripts/polaris/pi05_droid_native_gcs_manifest.tsv",
    "scripts/polaris/serve_pi05_droid_native_jointvelocity.py",
    "src/polaris/policy/droid_jointvelocity_client.py",
)


def canonical_json_bytes(value: Any) -> bytes:
    """Return strict canonical ASCII JSON with one trailing newline."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_immutable_json(path: Path, value: Any) -> dict[str, Any]:
    """Create one canonical mode-0444 JSON artifact without overwriting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    fsync_directory(path.parent)
    return validate_immutable_json(path)


def validate_immutable_json(path: Path) -> dict[str, Any]:
    """Read and bind one canonical mode-0444, single-link JSON artifact."""

    path = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"Immutable JSON is not readable: {path}") from error
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
            raise ValueError(f"Immutable JSON must be one regular link: {path}")
        if stat.S_IMODE(file_stat.st_mode) != 0o444:
            raise ValueError(f"Immutable JSON must have mode 0444: {path}")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(path, follow_symlinks=False)

    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mode,
            value.st_nlink,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(file_stat) != identity(after) or identity(file_stat) != identity(
        current
    ):
        raise ValueError(f"Immutable JSON changed while being read: {path}")
    try:
        value = json.loads(
            payload,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant is forbidden: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Immutable artifact is not strict JSON: {path}") from error
    if payload != canonical_json_bytes(value):
        raise ValueError(f"Immutable artifact is not canonical JSON: {path}")
    return {
        "path": str(path.resolve()),
        "size": len(payload),
        "sha256": sha256_bytes(payload),
        "mode": "0444",
        "nlink": 1,
        "value": value,
    }


def should_render_expensive(
    *,
    policy_client_name: str,
    render_every_step: bool,
    needs_next_policy_render: bool,
) -> bool:
    """Keep native video cadence separate without changing other clients."""

    if policy_client_name != "DroidJointVelocity":
        return needs_next_policy_render

    if (
        type(render_every_step) is not bool
        or type(needs_next_policy_render) is not bool
    ):
        raise TypeError("Render decisions require exact booleans")
    return render_every_step or needs_next_policy_render


def configure_native_environment_timeout(env_cfg: Any) -> float:
    """Move the simulator timeout past the exact 450-step outer canary horizon."""

    if not hasattr(env_cfg, "episode_length_s"):
        raise ValueError("Native environment config has no episode_length_s")
    env_cfg.episode_length_s = PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS
    return env_cfg.episode_length_s


def _environment_runtime_sha256(value: dict[str, Any]) -> str:
    payload = dict(value)
    payload.pop("sha256", None)
    return sha256_bytes(canonical_json_bytes(payload))


def make_environment_runtime_contract(
    *, configured_episode_length_seconds: Any, live_max_episode_length: Any
) -> dict[str, Any]:
    """Bind the configured and live timeout that prevents terminal auto-reset."""

    if (
        type(configured_episode_length_seconds) is not float
        or not math.isfinite(configured_episode_length_seconds)
        or configured_episode_length_seconds
        != PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS
    ):
        raise ValueError("Native configured episode length does not encode 451 steps")
    if (
        type(live_max_episode_length) is not int
        or live_max_episode_length != PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS
        or live_max_episode_length <= PI05_DROID_NATIVE_EPISODE_STEPS
    ):
        raise ValueError(
            "Native live max_episode_length must be exactly 451 and exceed 450"
        )
    value = {
        "schema_version": 1,
        "profile": PI05_DROID_NATIVE_ENVIRONMENT_RUNTIME_PROFILE,
        "outer_episode_steps": PI05_DROID_NATIVE_EPISODE_STEPS,
        "configured_episode_length_seconds": (
            PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS
        ),
        "live_max_episode_length": PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
        "timeout_margin_steps": (
            PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS
            - PI05_DROID_NATIVE_EPISODE_STEPS
        ),
        "policy_frequency_hz": PI05_DROID_NATIVE_POLICY_HZ,
        "physics_frequency_hz": PI05_DROID_NATIVE_PHYSICS_HZ,
        "decimation": PI05_DROID_NATIVE_DECIMATION,
        "require_terminated_false_for_all_outer_steps": True,
        "require_truncated_false_for_all_outer_steps": True,
        "post_action_observation_required": True,
        "sensor_liveness": {
            "profile": PI05_DROID_NATIVE_SENSOR_LIVENESS_PROFILE,
            "sensor_names": list(PI05_DROID_NATIVE_SENSOR_NAMES),
            "source_property": "isaaclab.sensors.camera.Camera.frame",
            "required_counter_increment_per_outer_step": 1,
            "image_hash_variation_authoritative": False,
        },
    }
    value["sha256"] = _environment_runtime_sha256(value)
    return value


def validate_environment_runtime_contract(value: Any) -> dict[str, Any]:
    """Validate one exact live environment-timeout and sensor-liveness binding."""

    expected = make_environment_runtime_contract(
        configured_episode_length_seconds=(
            PI05_DROID_NATIVE_CONFIGURED_EPISODE_LENGTH_SECONDS
        ),
        live_max_episode_length=PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS,
    )
    if (
        not isinstance(value, dict)
        or value != expected
        or value.get("sha256") != _environment_runtime_sha256(value)
    ):
        raise ValueError("Native environment runtime contract mismatch")
    return json.loads(canonical_json_bytes(value))


def _single_exact_bool(value: Any, field: str) -> bool:
    try:
        value = value.detach().cpu().tolist()
    except AttributeError:
        if hasattr(value, "tolist"):
            value = value.tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{field} must contain exactly one environment")
        value = value[0]
    if type(value) is not bool:
        raise ValueError(f"{field} must be one exact boolean")
    return value


def validate_outer_step_flags(
    terminated: Any, truncated: Any, *, outer_step_index: Any
) -> dict[str, Any]:
    """Reject any terminal or timeout boundary within the outer 450 steps."""

    if (
        type(outer_step_index) is not int
        or not 0 <= outer_step_index < PI05_DROID_NATIVE_EPISODE_STEPS
    ):
        raise ValueError("Native outer step index is invalid")
    terminated_value = _single_exact_bool(terminated, "terminated")
    truncated_value = _single_exact_bool(truncated, "truncated")
    if terminated_value or truncated_value:
        raise ValueError(
            "Native rollout hit an auto-reset boundary before all 450 post-action "
            f"states were captured: step={outer_step_index}, "
            f"terminated={terminated_value}, truncated={truncated_value}"
        )
    return {
        "outer_step_index": outer_step_index,
        "terminated": False,
        "truncated": False,
    }


def validate_terminal_rollout_evidence(
    value: Any, environment_runtime_contract: Any
) -> dict[str, Any]:
    """Validate the true post-action state and rubric captured before close."""

    runtime = validate_environment_runtime_contract(environment_runtime_contract)
    required = {
        "schema_version",
        "profile",
        "environment_runtime_sha256",
        "outer_steps_completed",
        "last_outer_step_index",
        "terminated_false_count",
        "truncated_false_count",
        "environment_before",
        "environment_after",
        "rubric",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Native terminal rollout evidence schema mismatch")
    if (
        value["schema_version"] != 1
        or value["profile"] != PI05_DROID_NATIVE_ENVIRONMENT_RUNTIME_PROFILE
        or value["environment_runtime_sha256"] != runtime["sha256"]
        or value["outer_steps_completed"] != PI05_DROID_NATIVE_EPISODE_STEPS
        or value["last_outer_step_index"] != PI05_DROID_NATIVE_EPISODE_STEPS - 1
        or value["terminated_false_count"] != PI05_DROID_NATIVE_EPISODE_STEPS
        or value["truncated_false_count"] != PI05_DROID_NATIVE_EPISODE_STEPS
    ):
        raise ValueError("Native terminal rollout evidence identity mismatch")
    before = value["environment_before"]
    after = value["environment_after"]
    environment_fields = {
        "live_max_episode_length",
        "episode_length",
        "sim_step_counter",
        "common_step_counter",
        "sensor_frame_counters",
    }
    if (
        not isinstance(before, dict)
        or not isinstance(after, dict)
        or set(before) != environment_fields
        or set(after) != environment_fields
    ):
        raise ValueError("Native terminal environment evidence schema mismatch")
    for name, environment in (("before", before), ("after", after)):
        if (
            environment["live_max_episode_length"]
            != PI05_DROID_NATIVE_INTERNAL_MAX_EPISODE_STEPS
            or type(environment["episode_length"]) is not int
            or type(environment["sim_step_counter"]) is not int
            or type(environment["common_step_counter"]) is not int
            or not isinstance(environment["sensor_frame_counters"], dict)
            or set(environment["sensor_frame_counters"])
            != set(PI05_DROID_NATIVE_SENSOR_NAMES)
            or any(
                type(counter) is not int or counter < 0
                for counter in environment["sensor_frame_counters"].values()
            )
        ):
            raise ValueError(f"Native terminal {name} environment evidence mismatch")
    if before["episode_length"] != 0 or after["episode_length"] != 450:
        raise ValueError("Native terminal episode length proves an auto-reset")
    if (
        after["sim_step_counter"] - before["sim_step_counter"]
        != PI05_DROID_NATIVE_EPISODE_STEPS * PI05_DROID_NATIVE_DECIMATION
        or after["common_step_counter"] - before["common_step_counter"]
        != PI05_DROID_NATIVE_EPISODE_STEPS
    ):
        raise ValueError("Native terminal simulator counters do not cover 450 actions")
    for sensor_name in PI05_DROID_NATIVE_SENSOR_NAMES:
        if (
            after["sensor_frame_counters"][sensor_name]
            - before["sensor_frame_counters"][sensor_name]
            != PI05_DROID_NATIVE_EPISODE_STEPS
        ):
            raise ValueError("Native terminal camera frame counters are not live")
    rubric = value["rubric"]
    if (
        not isinstance(rubric, dict)
        or set(rubric) != {"success", "progress"}
        or type(rubric["success"]) is not bool
        or type(rubric["progress"]) not in (int, float)
        or isinstance(rubric["progress"], bool)
        or not math.isfinite(rubric["progress"])
        or not 0.0 <= rubric["progress"] <= 1.0
    ):
        raise ValueError("Native terminal rubric evidence mismatch")
    return json.loads(canonical_json_bytes(value))


def validate_native_model_eval_contract(value: Any) -> dict[str, Any]:
    """Reject any normalization, image, state, or action-contract substitution."""

    if value != PI05_DROID_NATIVE_MODEL_EVAL_CONTRACT:
        raise ValueError("Official pi05_droid model eval contract mismatch")
    # JSON round-tripping gives callers an isolated, JSON-compatible deep copy.
    return json.loads(canonical_json_bytes(value))


def verify_official_norm_reference_probes(path: Path) -> dict[str, Any]:
    """Verify the global DROID stats and expose action/state semantic probes."""

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Official DROID norm stats must be one regular file")
    if file_sha256(path) != PI05_DROID_NORM_STATS_SHA256:
        raise ValueError("Official DROID norm-stats SHA-256 mismatch")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Official DROID norm stats are not strict JSON") from error
    if not isinstance(value, dict) or set(value) != {"norm_stats"}:
        raise ValueError("Official DROID norm-stats root schema mismatch")
    norm_stats = value["norm_stats"]
    if not isinstance(norm_stats, dict) or set(norm_stats) != {"actions", "state"}:
        raise ValueError("Official DROID norm-stats group schema mismatch")
    for group_name in ("actions", "state"):
        group = norm_stats[group_name]
        if not isinstance(group, dict) or set(group) != {"mean", "std", "q01", "q99"}:
            raise ValueError(f"Official DROID {group_name} stats schema mismatch")
        for statistic in ("mean", "std", "q01", "q99"):
            vector = group[statistic]
            if (
                not isinstance(vector, list)
                or len(vector) != 32
                or any(
                    type(item) not in (int, float) or not math.isfinite(item)
                    for item in vector
                )
            ):
                raise ValueError(
                    f"Official DROID {group_name} {statistic} vector mismatch"
                )
    observed = {
        "actions_q01_first8": norm_stats["actions"]["q01"][:8],
        "actions_q99_first8": norm_stats["actions"]["q99"][:8],
        "state_q01_first8": norm_stats["state"]["q01"][:8],
        "state_q99_first8": norm_stats["state"]["q99"][:8],
    }
    if observed != PI05_DROID_NATIVE_NORM_REFERENCE_PROBES:
        raise ValueError("Official DROID norm reference probes mismatch")
    return {
        "sha256": PI05_DROID_NORM_STATS_SHA256,
        "path_within_checkpoint": "assets/droid/norm_stats.json",
        "scope": "checkpoint_global_droid",
        "asset_id": "droid",
        "category_override": "forbidden",
        "probes": observed,
        "action_semantics": "joint_velocity_no_delta_or_absolute_transform",
        "state_semantics": "panda_joint_position_plus_closed_positive_gripper",
    }


def make_runtime_artifact(
    report: dict[str, Any], environment_runtime_contract: dict[str, Any]
) -> dict[str, Any]:
    environment_runtime = validate_environment_runtime_contract(
        environment_runtime_contract
    )
    return {
        "schema_version": 1,
        "profile": PI05_DROID_NATIVE_CANARY_PROFILE,
        "environment": PI05_DROID_NATIVE_TASK,
        "rollouts": PI05_DROID_NATIVE_ROLLOUTS,
        "episode_steps": PI05_DROID_NATIVE_EPISODE_STEPS,
        "environment_runtime_contract": environment_runtime,
        "runtime_contract": report,
    }


def make_close_ready_artifact(
    *,
    runtime_artifact: dict[str, Any],
    runtime_path: Path,
    metrics_path: Path,
    trace_path: Path,
    video_path: Path,
    environment_runtime_contract: dict[str, Any],
    terminal_rollout: dict[str, Any],
) -> dict[str, Any]:
    """Describe the successful child state immediately before Kit shutdown."""

    if runtime_artifact.get("path") != str(Path(runtime_path).resolve()):
        raise ValueError("Runtime artifact path binding mismatch")
    environment_runtime = validate_environment_runtime_contract(
        environment_runtime_contract
    )
    terminal = validate_terminal_rollout_evidence(terminal_rollout, environment_runtime)
    return {
        "schema_version": 1,
        "profile": PI05_DROID_NATIVE_CANARY_PROFILE,
        "status": "simulation_app_close_pending",
        "environment": PI05_DROID_NATIVE_TASK,
        "rollouts": PI05_DROID_NATIVE_ROLLOUTS,
        "episode_steps": PI05_DROID_NATIVE_EPISODE_STEPS,
        "env_close": "complete",
        "environment_runtime_contract_sha256": environment_runtime["sha256"],
        "terminal_rollout": terminal,
        "runtime_artifact": {
            key: runtime_artifact[key]
            for key in ("path", "size", "sha256", "mode", "nlink")
        },
        "metrics_path": str(Path(metrics_path).resolve()),
        "trace_path": str(Path(trace_path).resolve()),
        "video_path": str(Path(video_path).resolve()),
    }
