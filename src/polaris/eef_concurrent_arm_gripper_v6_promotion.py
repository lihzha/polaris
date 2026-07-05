"""Closed v6 evidence gate for the paired FoodBussing checkpoint canaries.

The cadence-correct controller evidence and real-camera image-contract
evidence are finalized and bound to immutable job-1098975 and job-1098982
attestations.  This schema authorizes only one official-LAP3B and one
reasoning-43075 FLOW rollout on DROID-FoodBussing.  Smoke-suite, standard,
native joint-position, and pi0.5 evaluation remain outside this gate.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping


SCHEMA_VERSION = 2
PROMOTION_PROFILE = "concurrent_arm_gripper_v6_cadence_image_checkpoint_canary_gate_v2"
FINAL_PROMOTION_STATUS = (
    "controller_cadence_and_camera_image_smokes_validated_checkpoint_canaries_only"
)
CONTROLLER_PROFILE = (
    "arm_slew_0p95_gripper_rate0p25_concurrent_arm_velocity_recovery8_"
    "clean2_mimic100_damping1p2_v6"
)
IK_SAFETY_PROFILE = (
    "panda_velocity_physxlimit_solveriter1_residual_recovery8_clean2_"
    "concurrent_arm_gripper_v6"
)

# Controller producer C1.  The ee6 revision appears only as C1's immutable Git
# parent.  It is not an evidence commit, promotion checkout, or launch target.
CONTROLLER_IMPLEMENTATION_COMMIT = "39418400493cdcf8cd8272608980a798f7929a20"
CONTROLLER_IMPLEMENTATION_TREE = "7fc1ff24053e3aeab5ed3e06068089b5aa596bc6"
CONTROLLER_IMPLEMENTATION_PARENT = "ee6d09351bed75e32db93ecf59c039a8e99fac9f"
CONTROLLER_IMPLEMENTATION_SOURCE_SHA256 = (
    (
        "scripts/eval.py",
        "b5464158b8cc996bffd55ee744133ba2d0d3708cca2288059076c829cee8a86f",
    ),
    (
        "scripts/smoke_eef_pose_controller.py",
        "b5b1b621041b74247dbce6488483cdf7d33d6fa7c4002e821cf50ed26609f6a8",
    ),
    (
        "src/polaris/config.py",
        "47f1a5af67e680e7b5762848697bd4c51b3b0f31132968d3551e0b28b0c889b6",
    ),
    (
        "src/polaris/eef_controller_profile.py",
        "fa55f0b1fc1bb9600c5d2d11d39bd670980791374de5a0dfba955e90929496a4",
    ),
    (
        "src/polaris/eef_controller_repair.py",
        "b2a4df4cccf5c7a4efadd9f6ca990e9b9c9eca8230024787c519c94b284d0f76",
    ),
    (
        "src/polaris/eef_gripper_failure_trace.py",
        "f66af5001f8636333f6db00948a64214909a43b7d6afd1af968397dea33280b0",
    ),
    (
        "src/polaris/eef_gripper_runtime.py",
        "0687434bc2c61bb09739be473d5477f7b92d7b7b846a800298a89225f5b4f220",
    ),
    (
        "src/polaris/eef_ik_safety.py",
        "bc34d745705227c1154bdb266a5e7f937739a90dad00e4802eebf2da8c6cd978",
    ),
    (
        "src/polaris/eef_runtime_contract.py",
        "fb7094a37a1b6c676c61cce1a371d1f146db69183d55fd46d9db84c2a8739a8b",
    ),
    (
        "src/polaris/robust_differential_ik.py",
        "a07a62f0ef5aebfb69e214e2c2d11bac197ab1a377686a72b02024e8285b16f4",
    ),
)

# Controller evidence C2 and its exact job-1098975 attestation.
CONTROLLER_EVIDENCE_COMMIT = "be2f608fd72d1441a777264cd6842f00fd5bf6e8"
CONTROLLER_EVIDENCE_TREE = "657f4a90908d7afd2f880954efc3df7566fa44f7"
CONTROLLER_EVIDENCE_PARENT = CONTROLLER_IMPLEMENTATION_COMMIT
CONTROLLER_EVIDENCE_CHANGED_PATHS = (
    "WORKLOG.v6.md",
    "scripts/finalize_eef_pose_controller_v6_smoke.py",
    "tests/test_finalize_eef_pose_controller_v6_smoke.py",
)
CONTROLLER_FINALIZER_PATH = "scripts/finalize_eef_pose_controller_v6_smoke.py"
CONTROLLER_FINALIZER_SHA256 = (
    "953eb9e43c93a4aa3525bcd3001019348d30b38c0279be6ea51cfd88cd6c6d81"
)
CONTROLLER_FINALIZER_SIZE_BYTES = 108_170
CONTROLLER_ATTESTATION_PATH = (
    "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/"
    "controller_concurrent_v6_cadence_smoke/3941840-20260705T053510Z/"
    "smoke-1098975.promotion-attestation.json"
)
CONTROLLER_ATTESTATION_SHA256 = (
    "4b5f53524590711874a06fa3d2f47b1b430df7ff7b82b445d14e44db2c4e1e90"
)
CONTROLLER_ATTESTATION_SIZE_BYTES = 11_481
CONTROLLER_ATTESTATION_MODE = 0o444
CONTROLLER_ATTESTATION_NLINK = 1

# Corrected image implementation and its worklog-only integration tip.
IMAGE_IMPLEMENTATION_COMMIT = "f1d32a3ca73ef8613b1c3e38f31f70fd06637857"
IMAGE_IMPLEMENTATION_TREE = "2ab8437d5144b76b32bba768a62e32667ca8483a"
IMAGE_IMPLEMENTATION_PARENT = CONTROLLER_EVIDENCE_COMMIT
IMAGE_IMPLEMENTATION_CHANGED_PATHS = (
    "src/polaris/environments/manager_based_rl_splat_environment.py",
    "src/polaris/splat_image_contract.py",
    "tests/test_splat_image_contract.py",
    "worklogs/agents/polaris-image-contract-fix-20260704.md",
)
IMAGE_IMPLEMENTATION_SOURCE_SHA256 = (
    (
        "src/polaris/environments/manager_based_rl_splat_environment.py",
        "175bf8a8fcf02418827d9c1913b1b3c0b2373967788fd5a2ff6250a49c3ac592",
    ),
    (
        "src/polaris/splat_image_contract.py",
        "a1bb3f3a9235cc08228c3ac1fc15f0c472c7cdfa1222425cdc40273a3bb64472",
    ),
    (
        "src/polaris/policy/lap_eef_pose_client.py",
        "b853182c2cac34e5a25edc09113c49f49537f2b43dacc784104ee7167473aa5e",
    ),
    (
        "src/polaris/policy/ego_lap_contract.py",
        "e3628052062d40ede220b596266617d806e2f1de10eee0873d28481952a00702",
    ),
    (
        "src/polaris/environments/droid_cfg.py",
        "91f54fdf2dd487294e514534f8fd2513148596caf86b31622d680885be0f7b0d",
    ),
    (
        "src/polaris/policy/droid_jointpos_client.py",
        "49b36504c7ae1038aa5efd4c6a9e133bb7e294e62ff4f3f3ca139c619cffe280",
    ),
    (
        "src/polaris/splat_renderer/splat_renderer.py",
        "b9104be8620738b6fe5bea88950e0a3b721d8d23455402071745d798c662a8aa",
    ),
)
IMAGE_INTEGRATION_COMMIT = "42e266353df71d5906e98975165f8aa021020dad"
IMAGE_INTEGRATION_TREE = "229f0f9b954d32015e01d82985e56a0e9d42f355"
IMAGE_INTEGRATION_PARENT = IMAGE_IMPLEMENTATION_COMMIT
IMAGE_INTEGRATION_CHANGED_PATHS = (
    "worklogs/agents/polaris-image-contract-fix-20260704.md",
)

# Real image-smoke producer C4.
IMAGE_SMOKE_PRODUCER_COMMIT = "9d296361bb323b2e309a3b92a204c102908c61a6"
IMAGE_SMOKE_PRODUCER_TREE = "2e868fd4a31a55c9cedfb3221e4c2bc1fbbb9310"
IMAGE_SMOKE_PRODUCER_PARENT = IMAGE_INTEGRATION_COMMIT
IMAGE_SMOKE_PRODUCER_CHANGED_PATHS = (
    "scripts/smoke_splat_image_contract.py",
    "tests/test_smoke_splat_image_contract.py",
    "worklogs/agents/polaris-image-contract-smoke-20260704.md",
)
IMAGE_SMOKE_PRODUCER_SOURCE_SHA256 = (
    (
        "scripts/smoke_splat_image_contract.py",
        "29db9e302179bb3ca4b05c14cae92e697376bb26066e4583b752bf9f6ce8d202",
    ),
    (
        "tests/test_smoke_splat_image_contract.py",
        "54b626119b6cc81f6050779dbc45ecfa92aa1fbdc12bda7743c2101556435fdf",
    ),
    (
        "worklogs/agents/polaris-image-contract-smoke-20260704.md",
        "4d8eb0785b731d25011f3119c0ab89439ec73bad54b903ebff2d7969be1a4250",
    ),
)

# Host-reviewed image evidence C5 and its exact job-1098982 attestation.
IMAGE_EVIDENCE_COMMIT = "5c9a2c50f564fb58d58777fbe34fb831ba362ec3"
IMAGE_EVIDENCE_TREE = "707a5d7b659e7c4dfc13d19ede9ce8a8077aeec7"
IMAGE_EVIDENCE_PARENT = IMAGE_SMOKE_PRODUCER_COMMIT
IMAGE_EVIDENCE_CHANGED_PATHS = (
    "WORKLOG.v6.md",
    "scripts/finalize_splat_image_contract_smoke.py",
    "tests/test_finalize_splat_image_contract_smoke.py",
)
IMAGE_EVIDENCE_FINALIZER_PATH = "scripts/finalize_splat_image_contract_smoke.py"
IMAGE_EVIDENCE_FINALIZER_SHA256 = (
    "4faf9d6edbea18e2761b333b15e23bab57a6443f81cb31a111d3db91d27b1e7c"
)
IMAGE_EVIDENCE_FINALIZER_SIZE_BYTES = 81_744
IMAGE_ATTESTATION_PATH = (
    "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/"
    "image_contract_smoke/9d29636-20260705T061730Z/"
    "smoke-1098982.image-evidence-attestation.json"
)
IMAGE_ATTESTATION_SHA256 = (
    "f85125e27c00bab0173a2f78642555bc4cdcf7d72ab5bb4cc9c2948cb84e4212"
)
IMAGE_ATTESTATION_SIZE_BYTES = 28_839
IMAGE_ATTESTATION_MODE = 0o444
IMAGE_ATTESTATION_NLINK = 1
IMAGE_SMOKE_JOB_ID = 1_098_982
IMAGE_ATTESTED_RUNTIME_SOURCE_SHA256 = (
    (
        "scripts/smoke_splat_image_contract.py",
        "29db9e302179bb3ca4b05c14cae92e697376bb26066e4583b752bf9f6ce8d202",
    ),
    (
        "src/polaris/environments/droid_cfg.py",
        "91f54fdf2dd487294e514534f8fd2513148596caf86b31622d680885be0f7b0d",
    ),
    (
        "src/polaris/environments/manager_based_rl_splat_environment.py",
        "175bf8a8fcf02418827d9c1913b1b3c0b2373967788fd5a2ff6250a49c3ac592",
    ),
    (
        "src/polaris/policy/ego_lap_contract.py",
        "e3628052062d40ede220b596266617d806e2f1de10eee0873d28481952a00702",
    ),
    (
        "src/polaris/policy/lap_eef_pose_client.py",
        "b853182c2cac34e5a25edc09113c49f49537f2b43dacc784104ee7167473aa5e",
    ),
    (
        "src/polaris/splat_image_contract.py",
        "a1bb3f3a9235cc08228c3ac1fc15f0c472c7cdfa1222425cdc40273a3bb64472",
    ),
    (
        "src/polaris/splat_renderer/splat_renderer.py",
        "b9104be8620738b6fe5bea88950e0a3b721d8d23455402071745d798c662a8aa",
    ),
)

PRESERVED_V5_SOURCE_SHA256 = (
    (
        "src/polaris/eef_velocity_recovery_promotion.py",
        "f98f6d3ae6eb06f0127e3ec686fa70e3bb524ea892582b0ee3461b0dd6d84df4",
    ),
    (
        "src/polaris/eef_velocity_recovery_standard_promotion.py",
        "8cb836645dde741876ff5d10b285761bc8f47c822732e9a2e1c469fb79ee0e06",
    ),
)

OFFICIAL_LAP3B_REVISION = "601db9c1ab4bcaf6dddb160c7b2dec589a67b730"
OFFICIAL_LAP3B_URI = f"hf://lihzha/LAP-3B@{OFFICIAL_LAP3B_REVISION}"
OFFICIAL_LAP3B_CONTENT_MANIFEST_SHA256 = (
    "567cc3ff7d20f3f03913a6f11c3fa151f789e1c0118ed5af0eea24d9cc48f20e"
)
REASONING_43075_URI = (
    "gs://v6_east1d/checkpoints/lap_oxe_magic_soup_reasoning_full/"
    "oxe_magic_soup_reasoning_full_v2_flow_pred0_cf0_ckpt25_"
    "v6_32_b512_s42_20260630/43075"
)
REASONING_43075_INFERENCE_SUBSET_SHA256 = (
    "bb9ea5bb041f689a08f914cac7dfe5d061c822ddbe87e292f9c7878a9d3bfc4d"
)

POLARIS_HUB_REVISION = "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b"
FOODBUSSING_SCENE_SHA256 = (
    "82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489"
)
FOODBUSSING_INITIAL_CONDITIONS_SHA256 = (
    "40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de"
)
PYXIS_IMAGE_SHA256 = "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"


# Exact gzip-compressed job-1098975 promotion-attestation bytes.  This is the
# cadence-correct capture; no superseded controller-smoke bytes are retained.
_CONTROLLER_ATTESTATION_GZIP_BASE64 = b"""
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
"""


def _duplicate_keys_rejected(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _nonfinite_rejected(value: str) -> None:
    raise ValueError(f"Non-finite JSON number: {value}")


def _validate_json_tree(value: Any, field: str = "root") -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value at {field}")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{field}[{index}]")
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"Non-string key at {field}")
            _validate_json_tree(item, f"{field}.{key}")
        return
    raise ValueError(f"Non-JSON or non-exact type at {field}: {type(value).__name__}")


def _strict_json(data: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            data,
            object_pairs_hook=_duplicate_keys_rejected,
            parse_constant=_nonfinite_rejected,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid strict JSON for {field}") from exc
    if type(value) is not dict:
        raise ValueError(f"{field} must be an exact JSON object")
    _validate_json_tree(value, field)
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    _validate_json_tree(value)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _controller_attestation_bytes() -> bytes:
    try:
        compressed = base64.b64decode(
            b"".join(_CONTROLLER_ATTESTATION_GZIP_BASE64.split()), validate=True
        )
        data = gzip.decompress(compressed)
    except (ValueError, gzip.BadGzipFile) as exc:
        raise RuntimeError("Embedded controller attestation is invalid") from exc
    if len(data) != CONTROLLER_ATTESTATION_SIZE_BYTES:
        raise RuntimeError("Embedded controller attestation size drift")
    if hashlib.sha256(data).hexdigest() != CONTROLLER_ATTESTATION_SHA256:
        raise RuntimeError("Embedded controller attestation digest drift")
    return data


def canonical_controller_smoke_attestation_bytes() -> bytes:
    """Return the exact cadence-correct job-1098975 attestation bytes."""

    return _controller_attestation_bytes()


def _validate_controller_attestation_semantics(value: Mapping[str, Any]) -> None:
    producer = value.get("producer")
    reviewer = value.get("reviewer")
    coverage = value.get("coverage_limits")
    summary = value.get("validation_summary")
    if (
        value.get("schema_version") != 1
        or value.get("profile")
        != "concurrent_arm_gripper_v6_cadence_smoke_job1098975_v1"
        or value.get("passed") is not True
        or value.get("finalized") is not True
        or type(producer) is not dict
        or type(reviewer) is not dict
        or type(coverage) is not dict
        or type(summary) is not dict
    ):
        raise ValueError("Controller-smoke attestation schema drift")
    if producer != {
        "controller_profile": CONTROLLER_PROFILE,
        "ik_safety_profile": IK_SAFETY_PROFILE,
        "polaris_commit": CONTROLLER_IMPLEMENTATION_COMMIT,
        "polaris_parent": CONTROLLER_IMPLEMENTATION_PARENT,
        "polaris_repo": (
            "/lustre/fsw/portfolios/nvr/users/lzha/src/"
            "PolaRiS-concurrent-v6-cadence-3941840-20260705T053510Z"
        ),
        "polaris_tree": CONTROLLER_IMPLEMENTATION_TREE,
        "source_sha256": dict(CONTROLLER_IMPLEMENTATION_SOURCE_SHA256),
    }:
        raise ValueError("Controller-smoke producer identity drift")
    finalizer = reviewer.get("finalizer")
    if (
        reviewer.get("evidence_commit") != CONTROLLER_EVIDENCE_COMMIT
        or reviewer.get("evidence_tree") != CONTROLLER_EVIDENCE_TREE
        or reviewer.get("evidence_parent") != CONTROLLER_EVIDENCE_PARENT
        or reviewer.get("changed_paths") != list(CONTROLLER_EVIDENCE_CHANGED_PATHS)
        or type(finalizer) is not dict
        or finalizer.get("sha256") != CONTROLLER_FINALIZER_SHA256
        or finalizer.get("size_bytes") != CONTROLLER_FINALIZER_SIZE_BYTES
    ):
        raise ValueError("Controller-smoke evidence identity drift")
    expected_false = {
        "camera_image_contract_validated",
        "checkpoint_loaded",
        "normalization_validated",
        "policy_serving_validated",
        "task_success_metric_validated",
    }
    if any(coverage.get(field) is not False for field in expected_false):
        raise ValueError("Controller-smoke coverage overclaim")
    if (
        summary.get("total_controller_apply_calls") != 5856
        or summary.get("safety_report_count") != 17
        or summary.get("controller_aborts") != 0
        or summary.get("recovery_events") != 0
        or summary.get("concurrent_closed_fresh_dls_applies") != 80
    ):
        raise ValueError("Controller-smoke validation summary drift")


def _image_evidence_identity_fields() -> dict[str, int | str]:
    return {
        "image_evidence_commit": IMAGE_EVIDENCE_COMMIT,
        "image_evidence_tree": IMAGE_EVIDENCE_TREE,
        "image_evidence_parent": IMAGE_EVIDENCE_PARENT,
        "image_evidence_finalizer_sha256": IMAGE_EVIDENCE_FINALIZER_SHA256,
        "image_evidence_finalizer_size_bytes": IMAGE_EVIDENCE_FINALIZER_SIZE_BYTES,
        "image_attestation_path": IMAGE_ATTESTATION_PATH,
        "image_attestation_sha256": IMAGE_ATTESTATION_SHA256,
        "image_attestation_size_bytes": IMAGE_ATTESTATION_SIZE_BYTES,
    }


def image_evidence_finalized() -> bool:
    """Return true only for the reviewed C5/job-1098982 identity."""

    return _image_evidence_identity_fields() == {
        "image_evidence_commit": "5c9a2c50f564fb58d58777fbe34fb831ba362ec3",
        "image_evidence_tree": "707a5d7b659e7c4dfc13d19ede9ce8a8077aeec7",
        "image_evidence_parent": "9d296361bb323b2e309a3b92a204c102908c61a6",
        "image_evidence_finalizer_sha256": (
            "4faf9d6edbea18e2761b333b15e23bab57a6443f81cb31a111d3db91d27b1e7c"
        ),
        "image_evidence_finalizer_size_bytes": 81_744,
        "image_attestation_path": (
            "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/"
            "image_contract_smoke/9d29636-20260705T061730Z/"
            "smoke-1098982.image-evidence-attestation.json"
        ),
        "image_attestation_sha256": (
            "f85125e27c00bab0173a2f78642555bc4cdcf7d72ab5bb4cc9c2948cb84e4212"
        ),
        "image_attestation_size_bytes": 28_839,
    }


def canonical_image_contract_expectation() -> dict[str, Any]:
    """Return the exact static contract job 1098982 must attest."""

    return {
        "job_id": IMAGE_SMOKE_JOB_ID,
        "profile": "polaris_foodbussing_splat_image_contract_smoke_v1",
        "scope": "image_contract_only_no_checkpoint_policy_action_metric_or_canary",
        "environment": "DROID-FoodBussing",
        "instruction": "Put all the foods in the bowl",
        "initial_condition_index": 0,
        "hub_revision": POLARIS_HUB_REVISION,
        "scene_sha256": FOODBUSSING_SCENE_SHA256,
        "initial_conditions_sha256": FOODBUSSING_INITIAL_CONDITIONS_SHA256,
        "pyxis_image_sha256": PYXIS_IMAGE_SHA256,
        "production_path": "ManagerBasedRLSplatEnv.custom_render",
        "camera_keys": ["external_cam", "wrist_cam"],
        "native_shape": [720, 1280, 3],
        "native_dtype": "uint8",
        "preprocessed_shape": [224, 224, 3],
        "resized_content_shape": [126, 224, 3],
        "padding_rows": {"top": 49, "bottom": 49},
        "renderer_conversion": (
            "float_clip_0_1_mul255_truncating_uint8_no_spatial_resample_v1"
        ),
        "robot_compositing": "np_where_robot_mask_over_splat_v1",
        "resize_profile": (
            "tf_bilinear_half_pixel_antialias_false_uint8_round_symmetric_zero_pad_v1"
        ),
        "wrist_operation_order": ["resize_with_pad", "rotate_180"],
        "operation_order_probe_noncommuting": True,
        "msgpack_roundtrip_exact": True,
        "removed_resize_counterfactual": (
            "cv2_default_linear_half_down_up_must_change_pixels_and_stay_off_live_path"
        ),
        "official_lap3b_route": {
            "model_image_keys": ["base_0_rgb", "left_wrist_0_rgb"],
            "model_image_order": ["external", "wrist"],
            "model_image_resolution": [224, 224],
        },
        "reasoning_43075_route": {
            "model_image_keys": ["camera_0_rgb", "camera_1_rgb", "camera_2_rgb"],
            "model_image_order": ["wrist", "external", "blank"],
            "model_image_resolution": [224, 224],
            "blank_image": {
                "shape": [224, 224, 3],
                "dtype": "uint8",
                "value": 0,
            },
        },
    }


def _validate_image_attestation_semantics(value: Mapping[str, Any]) -> None:
    """Validate the exact job-1098982 host-finalizer evidence envelope."""

    expected_top_level = {
        "schema_version",
        "profile",
        "status",
        "scope",
        "producer",
        "reviewer",
        "job",
        "runtime_inputs",
        "capture",
        "semantic_evidence",
        "authorizations",
    }
    if (
        type(value) is not dict
        or set(value) != expected_top_level
        or value.get("schema_version") != 1
        or value.get("profile")
        != "polaris_foodbussing_image_contract_evidence_job1098982_v1"
        or value.get("status") != "evidence_bundle_validated"
        or value.get("scope") != "standalone_image_boundary_evidence_only"
    ):
        raise ValueError("Image-smoke attestation schema drift")

    if value["producer"] != {
        "commit": IMAGE_SMOKE_PRODUCER_COMMIT,
        "tree": IMAGE_SMOKE_PRODUCER_TREE,
        "parent": IMAGE_SMOKE_PRODUCER_PARENT,
        "repo": (
            "/lustre/fsw/portfolios/nvr/users/lzha/src/"
            "PolaRiS-image-contract-smoke-9d29636-20260705T061008Z"
        ),
        "source_sha256": dict(IMAGE_ATTESTED_RUNTIME_SOURCE_SHA256),
    }:
        raise ValueError("Image-smoke producer identity drift")

    if value["reviewer"] != {
        "commit": IMAGE_EVIDENCE_COMMIT,
        "tree": IMAGE_EVIDENCE_TREE,
        "parent": IMAGE_EVIDENCE_PARENT,
        "repo": (
            "/lustre/fsw/portfolios/nvr/users/lzha/src/"
            "PolaRiS-image-evidence-5c9a2c5-20260705T063554Z"
        ),
        "changed_paths": list(IMAGE_EVIDENCE_CHANGED_PATHS),
        "finalizer": {
            "path": (
                "/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/src/"
                "PolaRiS-image-evidence-5c9a2c5-20260705T063554Z/"
                "scripts/finalize_splat_image_contract_smoke.py"
            ),
            "sha256": IMAGE_EVIDENCE_FINALIZER_SHA256,
            "size_bytes": IMAGE_EVIDENCE_FINALIZER_SIZE_BYTES,
            "mode": "0644",
            "nlink": 1,
            "mtime_ns": 1_783_233_365_000_000_000,
            "ctime_ns": 1_783_233_365_000_000_000,
        },
    }:
        raise ValueError("Image-smoke evidence identity drift")

    job = value["job"]
    if (
        type(job) is not dict
        or set(job) != {"job_id", "job_name", "scheduler", "sacct_snapshot"}
        or job.get("job_id") != str(IMAGE_SMOKE_JOB_ID)
        or job.get("job_name") != "pol_img_9d29636"
        or job.get("sacct_snapshot")
        != {
            "path": (
                "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/"
                "image_contract_smoke/9d29636-20260705T061730Z/"
                "sacct-terminal-1098982.json"
            ),
            "sha256": (
                "58671acba5ae4756bc9f0b5328f3f41f713fd99e4fa8dd6a538d94fe7f0a0672"
            ),
            "size_bytes": 1_830,
            "mode": "0444",
            "nlink": 1,
            "mtime_ns": 1_783_233_523_000_000_000,
            "ctime_ns": 1_783_233_523_000_000_000,
        }
    ):
        raise ValueError("Image-smoke scheduler identity drift")
    scheduler = job["scheduler"]
    expected_scheduler_ids = {
        "allocation": "1098982",
        "batch": "1098982.batch",
        "extern": "1098982.extern",
        "srun": "1098982.0",
    }
    if type(scheduler) is not dict or set(scheduler) != set(expected_scheduler_ids):
        raise ValueError("Image-smoke scheduler schema drift")
    for step, expected_job_id in expected_scheduler_ids.items():
        record = scheduler[step]
        if (
            type(record) is not dict
            or record.get("job_id") != expected_job_id
            or record.get("state") != "COMPLETED"
            or record.get("exit_code") != "0:0"
            or record.get("node") != "pool0-00010"
        ):
            raise ValueError("Image-smoke scheduler terminal-state drift")

    semantics = value["semantic_evidence"]
    if type(semantics) is not dict or set(semantics) != {
        "contracts",
        "environment",
        "pixel_evidence",
        "production_path",
    }:
        raise ValueError("Image-smoke semantic-evidence schema drift")
    expected_contracts = {
        "renderer_conversion": {
            "bgr_conversion": False,
            "channel_order": "RGB",
            "formula": "(clip(raw_float_rgb,0,1)*255).astype(uint8)",
            "pixel_exact": True,
            "shape_preserved": True,
        },
        "robot_compositing": {
            "formula": "np.where(robot_mask,sim_rgb,native_splat_rgb)",
            "native_shape": [720, 1280, 3],
            "pixel_exact": True,
        },
        "ego_lap_preprocessing": {
            "actual_client_class": (
                "polaris.policy.lap_eef_pose_client.EgoLAPEefPoseClient"
            ),
            "call_events": [
                "resize:external:720x1280->224x224",
                "resize:wrist:720x1280->224x224",
                "rotate180:wrist:224x224->224x224",
            ],
            "constructor_bypassed_no_network": True,
            "method": "_build_request",
            "native_shape": [720, 1280, 3],
            "operation_order_probe": {
                "differing_values": 99,
                "input_shape": [5, 8, 3],
                "production_matches_resize_then_rotate": True,
                "profile": "odd_5x8_to_7x7_asymmetric_padding_v1",
                "resize_then_rotate_sha256": (
                    "19f595391cdbf268f22969f676e77f373a3223efbc1c5ebb19effbddc6d81f47"
                ),
                "rotate_then_resize_sha256": (
                    "d9da8384bd84ede8ab081e1e04f12ad42759ad78e870096b2a2cc2b8c072e7a9"
                ),
                "target_shape": [7, 7, 3],
            },
            "padding_rows": {"bottom": 49, "top": 49},
            "pixel_exact_request_binding": True,
            "preprocessed_shape": [224, 224, 3],
            "resized_content_shape": [126, 224, 3],
            "wrist_operation_order": "resize_pad_then_rotate_180",
        },
        "msgpack_roundtrip": {
            "exact_arrays": True,
            "exact_image_bytes": True,
            "implementation": "openpi_client.msgpack_numpy",
            "packed_sha256": (
                "18bd00fdac85142b748444e181ff2cef0e50a432114c4ee594e029ce4932d60a"
            ),
        },
        "removed_resize_counterfactual": {
            "live_path": False,
            "profile": "removed_cv2_default_linear_half_down_up_v1",
            "required_to_change_pixels": True,
        },
    }
    if semantics["contracts"] != expected_contracts:
        raise ValueError("Image-smoke camera/preprocessing contract drift")

    environment = semantics["environment"]
    if (
        type(environment) is not dict
        or set(environment)
        != {
            "camera_sensor_keys",
            "hub_metadata",
            "hub_revision",
            "id",
            "initial_condition_index",
            "initial_conditions_file",
            "instruction",
            "renderer_camera_keys",
            "runtime_class",
            "scene_file",
        }
        or environment.get("id") != "DROID-FoodBussing"
        or environment.get("instruction") != "Put all the foods in the bowl"
        or environment.get("initial_condition_index") != 0
        or environment.get("hub_revision") != POLARIS_HUB_REVISION
        or environment.get("camera_sensor_keys") != ["external_cam", "wrist_cam"]
        or environment.get("renderer_camera_keys") != ["external_cam", "wrist_cam"]
        or environment.get("runtime_class")
        != (
            "polaris.environments.manager_based_rl_splat_environment."
            "ManagerBasedRLSplatEnv"
        )
        or environment.get("scene_file", {}).get("sha256") != FOODBUSSING_SCENE_SHA256
        or environment.get("initial_conditions_file", {}).get("sha256")
        != FOODBUSSING_INITIAL_CONDITIONS_SHA256
    ):
        raise ValueError("Image-smoke FoodBussing environment drift")
    if semantics["production_path"] != {
        "bound_render_splat_is_production_method": True,
        "events": [
            "ManagerBasedRLSplatEnv.render_splat.enter",
            "SplatRenderer.render",
            "ManagerBasedRLSplatEnv.render_splat.exit",
            "ManagerBasedRLSplatEnv.get_robot_from_sim",
        ],
        "get_robot_from_sim_calls": 1,
        "render_splat_calls": 1,
        "renderer_render_calls": 1,
    }:
        raise ValueError("Image-smoke production-path drift")

    pixel_evidence = semantics["pixel_evidence"]
    if (
        type(pixel_evidence) is not dict
        or set(pixel_evidence)
        != {
            "artifact_manifest_sha256",
            "camera_evidence",
            "contact_sheet_rgb_sha256",
            "msgpack",
        }
        or pixel_evidence.get("artifact_manifest_sha256")
        != "d14bdb03f67b2b105d21c8544ef19c1671c1dc437b9240fc60c1e3e0798d763b"
        or pixel_evidence.get("contact_sheet_rgb_sha256")
        != "f1b2e0e3931f49e1f393a5560e4df7df5a73091f6fe9b155f8cc3236a4f5ee87"
    ):
        raise ValueError("Image-smoke pixel-evidence identity drift")
    camera_evidence = pixel_evidence["camera_evidence"]
    expected_camera_hashes = {
        "external_cam": {
            "native_rgb_sha256": (
                "923d770193c6e481b2f2b976025022ce805a667771a03225043e96d92190ca65"
            ),
            "composited_rgb_sha256": (
                "3986855828044b721e3076235605d404205933c051ec8a8f791e02509065c972"
            ),
            "preprocessed_rgb_sha256": (
                "7fb64c7b5a8921042a847d7d3311a7ee7bfd7d4a01c7dc3094d33bc1b1c9afca"
            ),
        },
        "wrist_cam": {
            "native_rgb_sha256": (
                "97baa38b2570df5794395d94493ce9d8c21d1a98a642edcd59814b4b1bfb25cc"
            ),
            "composited_rgb_sha256": (
                "f0c988734f431015b133d707a15080c8f563e1fa24c2ca4fd4e865fdff236ba2"
            ),
            "preprocessed_rgb_sha256": (
                "34d9195f72355475dbed2b7d103ab4d215f8b46e3401674c452d6609d95bda3f"
            ),
        },
    }
    if type(camera_evidence) is not dict or set(camera_evidence) != set(
        expected_camera_hashes
    ):
        raise ValueError("Image-smoke camera pixel schema drift")
    for camera, expected_hashes in expected_camera_hashes.items():
        actual = camera_evidence[camera]
        if type(actual) is not dict or any(
            actual.get(field) != expected for field, expected in expected_hashes.items()
        ):
            raise ValueError("Image-smoke camera pixel hash drift")
        counterfactual = actual.get("counterfactual")
        if (
            type(counterfactual) is not dict
            or counterfactual.get("changed_pixels", 0) <= 0
            or counterfactual.get("changed_values", 0) <= 0
        ):
            raise ValueError("Image-smoke removed-resize counterfactual drift")
    if pixel_evidence["msgpack"] != {
        "external_image_sha256": expected_camera_hashes["external_cam"][
            "preprocessed_rgb_sha256"
        ],
        "wrist_image_sha256": expected_camera_hashes["wrist_cam"][
            "preprocessed_rgb_sha256"
        ],
        "observation_keys": [
            "base_0_rgb",
            "cartesian_position",
            "gripper_position",
            "left_wrist_0_rgb",
            "state",
        ],
        "request_keys": [
            "dataset_name",
            "eef_frame",
            "frame_description",
            "has_wrist_image",
            "is_bimanual",
            "observation",
            "prompt",
            "rotation_applied",
            "state_type",
        ],
        "serialized_request_sha256": (
            "18bd00fdac85142b748444e181ff2cef0e50a432114c4ee594e029ce4932d60a"
        ),
    }:
        raise ValueError("Image-smoke request serialization drift")

    if value["authorizations"] != {
        "benchmark_result": False,
        "canary": False,
        "checkpoint_evaluation": False,
        "controller_behavior": False,
        "policy_serving": False,
        "promotion": False,
        "smoke_suite": False,
        "standard_suite": False,
        "task_metric": False,
    }:
        raise ValueError("Image-smoke standalone-evidence authorization drift")


def canonical_paired_checkpoint_canary_request() -> dict[str, Any]:
    """Return the only checkpoint request the finalized gate may authorize."""

    return {
        "eval_scale": "canary",
        "stage": "paired_official_and_reasoning_foodbussing_canaries",
        "benchmark": "polaris_droid_suite_v1",
        "task": "DROID-FoodBussing",
        "checkpoint_roles": ["official_lap3b", "reasoning_43075"],
        "checkpoints": {
            "official_lap3b": {
                "uri": OFFICIAL_LAP3B_URI,
                "revision": OFFICIAL_LAP3B_REVISION,
                "content_manifest_sha256": (OFFICIAL_LAP3B_CONTENT_MANIFEST_SHA256),
                "checkpoint_profile": "original_lap_public_3b_v1",
                "policy_type": "flow",
                "flow_num_steps": 10,
                "response_horizon": 16,
                "execution_horizon": 8,
                "model_image_keys": ["base_0_rgb", "left_wrist_0_rgb"],
                "model_image_order": ["external", "wrist"],
                "legacy_image_order": True,
                "native_image_resolution": [720, 1280],
                "model_image_resolution": [224, 224],
                "state_encoding": "EEF_R6",
                "state_layout": "xyz+r6_first_two_rows+gripper_open",
                "state_layout_mode": "public_lap_train_matched_rows_v1",
                "language_action_frame": "robot base frame",
            },
            "reasoning_43075": {
                "uri": REASONING_43075_URI,
                "step": 43075,
                "inference_subset_profile": "policy-inference-params-assets-v1",
                "inference_subset_sha256": (REASONING_43075_INFERENCE_SUBSET_SHA256),
                "checkpoint_profile": "manifest_v1_canonical",
                "policy_type": "flow",
                "flow_num_steps": 10,
                "response_horizon": 16,
                "execution_horizon": 8,
                "model_image_keys": [
                    "camera_0_rgb",
                    "camera_1_rgb",
                    "camera_2_rgb",
                ],
                "model_image_order": ["wrist", "external", "blank"],
                "legacy_image_order": False,
                "native_image_resolution": [720, 1280],
                "model_image_resolution": [224, 224],
                "blank_image": {
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                    "value": 0,
                },
                "state_encoding": "EEF_R6",
                "state_layout": "xyz+r6_first_two_columns+gripper_open",
                "state_layout_mode": "manifest_train_matched_columns_v1",
                "language_action_frame": "egocentric frame",
            },
        },
        "shared_train_eval_contract": {
            "policy_client": "EgoLAPEefPose",
            "image_color_space": "RGB",
            "image_dtype": "uint8",
            "renderer_conversion": (
                "float_clip_0_1_mul255_truncating_uint8_no_spatial_resample_v1"
            ),
            "resize_profile": (
                "tf_bilinear_half_pixel_antialias_false_uint8_round_"
                "symmetric_zero_pad_v1"
            ),
            "wrist_image_preprocessing": {
                "operation_order": ["resize_with_pad", "rotate_180"],
                "rotation_degrees": 180,
            },
            "normalization": {
                "source": "checkpoint_assets",
                "type": "bounds_q99",
                "scope": "global",
                "configured_policy_category": "single_arm",
                "effective_selected_category": None,
                "compute_dtype": "float32",
                "formula_profile": "q99_train_matched_v1",
                "input_formula_id": "q99_input_eps1e-8_clip_zero0_v1",
                "output_formula_id": ("q99_output_eps1e-8_zeroq01_extrapolate_v1"),
            },
            "response_semantics": "cumulative_delta_targets",
            "numeric_action_frame": "robot_base",
        },
        "rollouts_per_checkpoint": 1,
        "total_rollouts": 2,
        "control_mode": "absolute_end_effector_pose",
        "eef_frame": "panda_link8_relative_to_panda_link0",
        "controller_profile": CONTROLLER_PROFILE,
        "ik_safety_profile": IK_SAFETY_PROFILE,
        "native_joint_position_or_pi05_authorized": False,
    }


def canonical_eef_concurrent_arm_gripper_v6_promotion_evidence() -> dict[str, Any]:
    """Return the exact finalized dual-evidence promotion manifest."""

    if not image_evidence_finalized():
        raise RuntimeError("Finalized image-smoke evidence identity drift")
    request = canonical_paired_checkpoint_canary_request()
    controller_attestation = _strict_json(
        _controller_attestation_bytes(), "embedded controller attestation"
    )
    _validate_controller_attestation_semantics(controller_attestation)
    return {
        "schema_version": SCHEMA_VERSION,
        "profile": PROMOTION_PROFILE,
        "status": FINAL_PROMOTION_STATUS,
        "lineage": {
            "controller_implementation": {
                "commit": CONTROLLER_IMPLEMENTATION_COMMIT,
                "tree": CONTROLLER_IMPLEMENTATION_TREE,
                "parent": CONTROLLER_IMPLEMENTATION_PARENT,
                "historical_parent_is_not_authorized": True,
                "source_sha256": dict(CONTROLLER_IMPLEMENTATION_SOURCE_SHA256),
            },
            "controller_evidence": {
                "commit": CONTROLLER_EVIDENCE_COMMIT,
                "tree": CONTROLLER_EVIDENCE_TREE,
                "parent": CONTROLLER_EVIDENCE_PARENT,
                "changed_paths": list(CONTROLLER_EVIDENCE_CHANGED_PATHS),
                "finalizer_path": CONTROLLER_FINALIZER_PATH,
                "finalizer_sha256": CONTROLLER_FINALIZER_SHA256,
                "finalizer_size_bytes": CONTROLLER_FINALIZER_SIZE_BYTES,
            },
            "image_implementation": {
                "commit": IMAGE_IMPLEMENTATION_COMMIT,
                "tree": IMAGE_IMPLEMENTATION_TREE,
                "parent": IMAGE_IMPLEMENTATION_PARENT,
                "changed_paths": list(IMAGE_IMPLEMENTATION_CHANGED_PATHS),
                "source_sha256": dict(IMAGE_IMPLEMENTATION_SOURCE_SHA256),
            },
            "image_integration_tip": {
                "commit": IMAGE_INTEGRATION_COMMIT,
                "tree": IMAGE_INTEGRATION_TREE,
                "parent": IMAGE_INTEGRATION_PARENT,
                "changed_paths": list(IMAGE_INTEGRATION_CHANGED_PATHS),
                "runtime_semantics_changed": False,
            },
            "image_smoke_producer": {
                "commit": IMAGE_SMOKE_PRODUCER_COMMIT,
                "tree": IMAGE_SMOKE_PRODUCER_TREE,
                "parent": IMAGE_SMOKE_PRODUCER_PARENT,
                "changed_paths": list(IMAGE_SMOKE_PRODUCER_CHANGED_PATHS),
                "source_sha256": dict(IMAGE_SMOKE_PRODUCER_SOURCE_SHA256),
            },
            "image_evidence": {
                "commit": IMAGE_EVIDENCE_COMMIT,
                "tree": IMAGE_EVIDENCE_TREE,
                "parent": IMAGE_EVIDENCE_PARENT,
                "expected_parent": IMAGE_SMOKE_PRODUCER_COMMIT,
                "changed_paths": list(IMAGE_EVIDENCE_CHANGED_PATHS),
                "finalizer_path": IMAGE_EVIDENCE_FINALIZER_PATH,
                "finalizer_sha256": IMAGE_EVIDENCE_FINALIZER_SHA256,
                "finalizer_size_bytes": IMAGE_EVIDENCE_FINALIZER_SIZE_BYTES,
                "finalized": True,
            },
            "preserved_v5_source_sha256": dict(PRESERVED_V5_SOURCE_SHA256),
            "promotion_changes_runtime_behavior": False,
        },
        "controller_smoke": {
            "attestation_identity": {
                "path": CONTROLLER_ATTESTATION_PATH,
                "sha256": CONTROLLER_ATTESTATION_SHA256,
                "size_bytes": CONTROLLER_ATTESTATION_SIZE_BYTES,
                "mode": "0444",
                "nlink": CONTROLLER_ATTESTATION_NLINK,
            },
            "attestation": controller_attestation,
            "claims": {
                "controller_cadence_and_safety_validated": True,
                "camera_image_contract_validated": False,
                "checkpoint_loaded": False,
                "normalization_validated": False,
                "policy_serving_validated": False,
                "task_success_metric_validated": False,
            },
        },
        "image_smoke": {
            "job_id": IMAGE_SMOKE_JOB_ID,
            "attestation_identity": {
                "path": IMAGE_ATTESTATION_PATH,
                "sha256": IMAGE_ATTESTATION_SHA256,
                "size_bytes": IMAGE_ATTESTATION_SIZE_BYTES,
                "mode": "0444",
                "nlink": IMAGE_ATTESTATION_NLINK,
            },
            "expected_contract": canonical_image_contract_expectation(),
            "claims": {
                "camera_image_contract_validated": True,
                "image_order_or_resolution_validated": True,
                "checkpoint_loaded": False,
                "normalization_validated": False,
                "policy_serving_validated": False,
                "task_success_metric_validated": False,
            },
        },
        "paired_checkpoint_contract": request,
        "authorization": {
            "evidence_finalized": True,
            "pending_replacement_fields": [],
            "allowed_eval_scales": ["canary"],
            "allowed_tasks": ["DROID-FoodBussing"],
            "allowed_checkpoint_roles": ["official_lap3b", "reasoning_43075"],
            "next_eval_scale": "canary",
            "paired_checkpoint_canary_request": request,
            "canary_authorized": True,
            "smoke_suite_authorized": False,
            "standard_authorized": False,
            "native_joint_position_or_pi05_authorized": False,
            "requires_both_immutable_attestations": True,
            "requires_exact_source_identity_validation": True,
            "controller_or_evaluator_behavior_change_authorized": False,
        },
    }


PROMOTION_EVIDENCE_SHA256 = _canonical_sha256(
    canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
)
EXPECTED_PROMOTION_EVIDENCE_SHA256 = (
    "5a03e09fc5dd8d3c3d196909695d58e7ae589a99d3db9fe730877481858ab301"
)
if PROMOTION_EVIDENCE_SHA256 != EXPECTED_PROMOTION_EVIDENCE_SHA256:
    raise RuntimeError("Canonical finalized v6 dual-evidence manifest drift")

PAIRED_CANARY_REQUEST_SHA256 = _canonical_sha256(
    canonical_paired_checkpoint_canary_request()
)
EXPECTED_PAIRED_CANARY_REQUEST_SHA256 = (
    "6963d2cc0f3c02ee9a4e5f2f3c3718a027bbbfe36a97e884c48e83e94122be28"
)
if PAIRED_CANARY_REQUEST_SHA256 != EXPECTED_PAIRED_CANARY_REQUEST_SHA256:
    raise RuntimeError("Canonical v6 paired canary request drift")


def validate_eef_concurrent_arm_gripper_v6_promotion_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject structural, value, exact-type, or canonical-digest drift."""

    _validate_json_tree(value)
    expected = canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
    if type(value) is not dict or value != expected:
        raise ValueError("V6 dual-evidence promotion manifest drift")
    if _canonical_sha256(value) != PROMOTION_EVIDENCE_SHA256:
        raise ValueError("V6 dual-evidence promotion digest drift")
    return deepcopy(expected)


def validate_paired_checkpoint_canary_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject anything except the exact paired FLOW FoodBussing request."""

    _validate_json_tree(value, "canary request")
    expected = canonical_paired_checkpoint_canary_request()
    if type(value) is not dict or value != expected:
        raise ValueError("V6 paired checkpoint-canary request drift")
    if _canonical_sha256(value) != PAIRED_CANARY_REQUEST_SHA256:
        raise ValueError("V6 paired checkpoint-canary request digest drift")
    return deepcopy(expected)


def eef_concurrent_arm_gripper_v6_eval_scale_allowed(eval_scale: str) -> bool:
    """Allow only the canary scale while finalized identities remain exact."""

    return (
        image_evidence_finalized()
        and type(eval_scale) is str
        and eval_scale == "canary"
    )


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _validate_immutable_file(
    path: Path,
    *,
    expected_path: str,
    expected_sha256: str,
    expected_size: int,
    expected_mode: int,
    expected_nlink: int,
    field: str,
    allow_content_addressed_mirror: bool,
    expected_bytes: bytes | None = None,
) -> bytes:
    if not isinstance(path, Path):
        raise ValueError(f"{field} path must be pathlib.Path")
    if type(allow_content_addressed_mirror) is not bool:
        raise ValueError(f"{field} mirror selection must be an exact boolean")
    if not allow_content_addressed_mirror and str(path) != expected_path:
        raise ValueError(f"{field} path drift")
    try:
        before_lstat = path.lstat()
    except OSError as exc:
        raise ValueError(f"{field} is unavailable") from exc
    if not stat.S_ISREG(before_lstat.st_mode) or stat.S_ISLNK(before_lstat.st_mode):
        raise ValueError(f"{field} is not regular")
    if stat.S_IMODE(before_lstat.st_mode) != expected_mode:
        raise ValueError(f"{field} mode drift")
    if before_lstat.st_nlink != expected_nlink:
        raise ValueError(f"{field} link-count drift")
    if before_lstat.st_size != expected_size:
        raise ValueError(f"{field} size drift")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError(f"{field} open failed") from exc
    try:
        before_fstat = os.fstat(descriptor)
        if _stat_identity(before_fstat) != _stat_identity(before_lstat):
            raise ValueError(f"{field} changed before read")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after_fstat = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after_lstat = path.lstat()
    except OSError as exc:
        raise ValueError(f"{field} changed after read") from exc
    if _stat_identity(before_fstat) != _stat_identity(after_fstat) or _stat_identity(
        before_lstat
    ) != _stat_identity(after_lstat):
        raise ValueError(f"{field} changed during read")
    data = b"".join(chunks)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"{field} digest drift")
    if expected_bytes is not None and data != expected_bytes:
        raise ValueError(f"{field} byte drift")
    return data


def validate_controller_smoke_attestation(
    path: Path, *, allow_content_addressed_mirror: bool = False
) -> dict[str, Any]:
    """Freshly bind and semantically validate the job-1098975 attestation."""

    data = _validate_immutable_file(
        path,
        expected_path=CONTROLLER_ATTESTATION_PATH,
        expected_sha256=CONTROLLER_ATTESTATION_SHA256,
        expected_size=CONTROLLER_ATTESTATION_SIZE_BYTES,
        expected_mode=CONTROLLER_ATTESTATION_MODE,
        expected_nlink=CONTROLLER_ATTESTATION_NLINK,
        field="Controller-smoke attestation",
        allow_content_addressed_mirror=allow_content_addressed_mirror,
        expected_bytes=_controller_attestation_bytes(),
    )
    value = _strict_json(data, "controller-smoke attestation")
    _validate_controller_attestation_semantics(value)
    return value


def validate_image_smoke_attestation(
    path: Path, *, allow_content_addressed_mirror: bool = False
) -> dict[str, Any]:
    """Freshly bind and semantically validate the job-1098982 attestation."""

    if not image_evidence_finalized():
        raise ValueError("Finalized image-smoke evidence identity drift")
    data = _validate_immutable_file(
        path,
        expected_path=IMAGE_ATTESTATION_PATH,
        expected_sha256=IMAGE_ATTESTATION_SHA256,
        expected_size=IMAGE_ATTESTATION_SIZE_BYTES,
        expected_mode=IMAGE_ATTESTATION_MODE,
        expected_nlink=IMAGE_ATTESTATION_NLINK,
        field="Image-smoke attestation",
        allow_content_addressed_mirror=allow_content_addressed_mirror,
    )
    value = _strict_json(data, "image-smoke attestation")
    _validate_image_attestation_semantics(value)
    return value


def _validate_source_file(path: Path, expected_sha256: str, field: str) -> str:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError(f"Missing {field}: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{field} is not a regular file: {path}")
    if before.st_nlink != 1:
        raise ValueError(f"{field} must have one hard link: {path}")
    data = _validate_immutable_file(
        path,
        expected_path=str(path),
        expected_sha256=expected_sha256,
        expected_size=before.st_size,
        expected_mode=stat.S_IMODE(before.st_mode),
        expected_nlink=1,
        field=field,
        allow_content_addressed_mirror=False,
    )
    return hashlib.sha256(data).hexdigest()


def validate_v6_inherited_source_identity(repo_root: Path) -> dict[str, Any]:
    """Validate every exact inherited runtime and evidence source."""

    if not isinstance(repo_root, Path):
        raise ValueError("Repository root must be pathlib.Path")
    source_sets = {
        "controller_implementation_source_sha256": dict(
            CONTROLLER_IMPLEMENTATION_SOURCE_SHA256
        ),
        "image_implementation_source_sha256": dict(IMAGE_IMPLEMENTATION_SOURCE_SHA256),
        "image_smoke_producer_source_sha256": dict(IMAGE_SMOKE_PRODUCER_SOURCE_SHA256),
        "preserved_v5_source_sha256": dict(PRESERVED_V5_SOURCE_SHA256),
    }
    validated: dict[str, dict[str, str]] = {}
    for set_name, sources in source_sets.items():
        validated[set_name] = {
            relative: _validate_source_file(
                repo_root / relative, digest, f"V6 {set_name}"
            )
            for relative, digest in sources.items()
        }
    validated["controller_evidence"] = {
        "finalizer_sha256": _validate_source_file(
            repo_root / CONTROLLER_FINALIZER_PATH,
            CONTROLLER_FINALIZER_SHA256,
            "V6 controller evidence finalizer",
        )
    }
    return validated


def validate_v6_promotion_source_identity(repo_root: Path) -> dict[str, Any]:
    """Validate inherited sources plus the exact C5 image finalizer."""

    if not image_evidence_finalized():
        raise ValueError("Finalized image-smoke evidence identity drift")
    validated = validate_v6_inherited_source_identity(repo_root)
    validated["image_evidence"] = {
        "finalizer_sha256": _validate_source_file(
            repo_root / IMAGE_EVIDENCE_FINALIZER_PATH,
            IMAGE_EVIDENCE_FINALIZER_SHA256,
            "V6 image evidence finalizer",
        )
    }
    return validated


def validate_and_authorize_paired_checkpoint_canaries(
    evidence: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    controller_attestation_path: Path,
    image_attestation_path: Path,
    repo_root: Path,
    allow_content_addressed_attestation_mirrors: bool = False,
) -> dict[str, Any]:
    """Authorize only the pair after both attestations and sources are exact."""

    if not image_evidence_finalized():
        raise ValueError("Finalized image-smoke evidence identity drift")
    validated_evidence = validate_eef_concurrent_arm_gripper_v6_promotion_evidence(
        evidence
    )
    validated_request = validate_paired_checkpoint_canary_request(request)
    validate_controller_smoke_attestation(
        controller_attestation_path,
        allow_content_addressed_mirror=allow_content_addressed_attestation_mirrors,
    )
    validate_image_smoke_attestation(
        image_attestation_path,
        allow_content_addressed_mirror=allow_content_addressed_attestation_mirrors,
    )
    source_identity = validate_v6_promotion_source_identity(repo_root)
    authorization = validated_evidence["authorization"]
    if authorization != {
        **authorization,
        "evidence_finalized": True,
        "pending_replacement_fields": [],
        "allowed_eval_scales": ["canary"],
        "allowed_tasks": ["DROID-FoodBussing"],
        "allowed_checkpoint_roles": ["official_lap3b", "reasoning_43075"],
        "next_eval_scale": "canary",
        "canary_authorized": True,
        "smoke_suite_authorized": False,
        "standard_authorized": False,
        "native_joint_position_or_pi05_authorized": False,
    }:
        raise ValueError("Closed evidence does not authorize only paired canaries")
    return {
        "authorized": True,
        "eval_scale": "canary",
        "stage": validated_request["stage"],
        "request": validated_request,
        "promotion_evidence_sha256": PROMOTION_EVIDENCE_SHA256,
        "controller_smoke_attestation_sha256": CONTROLLER_ATTESTATION_SHA256,
        "image_smoke_attestation_sha256": IMAGE_ATTESTATION_SHA256,
        "verified_controller_attestation_path": str(controller_attestation_path),
        "verified_image_attestation_path": str(image_attestation_path),
        "content_addressed_attestation_mirrors_used": (
            allow_content_addressed_attestation_mirrors
        ),
        "source_identity": source_identity,
        "validation_claims": {
            "camera_image_contract_validated": True,
            "image_order_or_resolution_validated": True,
            "checkpoint_loaded": False,
            "normalization_validated": False,
            "policy_serving_validated": False,
            "task_success_metric_validated": False,
        },
        "native_joint_position_or_pi05_authorized": False,
    }
