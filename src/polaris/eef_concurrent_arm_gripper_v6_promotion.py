"""Closed evidence gate for the v6 paired FoodBussing checkpoint canaries.

This module changes no controller, evaluator, policy, or simulator behavior.  It
binds the reviewed job-1098922 standalone controller-smoke attestation and
authorizes only the next lifecycle stage: one official-LAP3B and one reasoning
checkpoint rollout on ``DROID-FoodBussing``.  Smoke-suite and standard scales
remain blocked.  In particular, the controller smoke makes no claim that a
checkpoint loaded or that image, normalization, policy-serving, or task-success
contracts were validated.
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


PROMOTION_PROFILE = "concurrent_arm_gripper_v6_checkpoint_canary_gate_v1"
PROMOTION_STATUS = "controller_smoke_validated_checkpoint_canaries_only"
PRODUCER_COMMIT = "6e4b7c5be5ff6db670970774be3250c5d5ffa4d2"
PRODUCER_TREE = "328063f148832f7050634b09b9db22bc5d1d5095"
PRODUCER_PARENT = "b6dec3d0c053066d65f6998cf1ecc33fc6e6e9ff"
EVIDENCE_COMMIT = "f4a27ce2bdbbaf2b87a38b4850390f9697ce8f9e"
EVIDENCE_TREE = "4c4ce225bdfd57564e2e90db7657f9dc807a93f8"
EVIDENCE_PARENT = PRODUCER_COMMIT
FINALIZER_PATH = "scripts/finalize_eef_pose_controller_v6_smoke.py"
FINALIZER_SHA256 = "f9ab24398286d5e4db2af816cfa86c9b0b355c13eeb246e307331b5e14720c4c"
FINALIZER_SIZE_BYTES = 106_246
CONTROLLER_PROFILE = (
    "arm_slew_0p95_gripper_rate0p25_concurrent_arm_velocity_recovery8_"
    "clean2_mimic100_damping1p2_v6"
)
IK_SAFETY_PROFILE = (
    "panda_velocity_physxlimit_solveriter1_residual_recovery8_clean2_"
    "concurrent_arm_gripper_v6"
)

CONTROLLER_SMOKE_ATTESTATION_PATH = (
    "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/"
    "controller_concurrent_v6_smoke/6e4b7c5-20260705T021644Z/"
    "smoke-1098922.promotion-attestation.json"
)
CONTROLLER_SMOKE_ATTESTATION_SHA256 = (
    "c359e978bf4aede7555fd3d6118a2abf5f7f4c2e5cf058326d7c3304bda2305a"
)
CONTROLLER_SMOKE_ATTESTATION_SIZE_BYTES = 10_423
CONTROLLER_SMOKE_ATTESTATION_MODE = 0o444
CONTROLLER_SMOKE_ATTESTATION_NLINK = 1

AUTHORIZED_EVAL_SCALES = ("canary",)
NEXT_EVAL_SCALE = "canary"
NEXT_STAGE = "paired_official_and_reasoning_foodbussing_canaries"
CANARY_TASK = "DROID-FoodBussing"
CHECKPOINT_ROLES = ("official_lap3b", "reasoning_43075")
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

PRODUCER_SOURCE_SHA256 = (
    (
        "scripts/eval.py",
        "b5464158b8cc996bffd55ee744133ba2d0d3708cca2288059076c829cee8a86f",
    ),
    (
        "scripts/smoke_eef_pose_controller.py",
        "30e0e29c70c470b366cbbf8afa4ac61fa2e1cd05d30f5d0cd4dc4fb2d5cbe63e",
    ),
    (
        "src/polaris/config.py",
        "47f1a5af67e680e7b5762848697bd4c51b3b0f31132968d3551e0b28b0c889b6",
    ),
    (
        "src/polaris/eef_controller_profile.py",
        "e549608d618ad109cb35e7e30e0e5ac452091dd97fd1f6f10bc9be502c8b6608",
    ),
    (
        "src/polaris/eef_controller_repair.py",
        "1eb7771cc2f3f2f457949208c6cdbcbf6bc20fede431e77908012f1f69d1001a",
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
        "83b25b04a43c36d5ba5f5b7ac5ba3481b6e864ec98c4be8c3a79bedfa458d0f0",
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

SEALED_PROVENANCE = (
    (
        "sacct",
        "0444",
        1823,
        "13ec9313b8a593463e23a649798871338ca59e672a530c43849dd9125938996c",
    ),
    (
        "saved_job_script",
        "0444",
        9548,
        "a74f46d5e6d1e359b8df9bc02209e1b08c10b400cfd710826c6203fdeec55669",
    ),
    (
        "slurm_log",
        "0444",
        44136,
        "64179ecdc1ae32b51fe12a88b11987df955e7053a03e2b26cfad30c08a0621f6",
    ),
    (
        "source_identity",
        "0444",
        800,
        "52c9a5c00506886e68dd394950fb84e80cbd9bfb18c91290e3abd201162561ad",
    ),
)

# Gzip is used only to keep the exact reviewed 10,423-byte JSON tractable in
# source.  Import-time size/SHA checks and strict parsing make this a closed,
# byte-for-byte attestation pin rather than a second interpretation of it.
_CONTROLLER_SMOKE_ATTESTATION_GZIP_BASE64 = b"""
H4sIAAAAAAACA+VaW2/cxhV+968I9KzVzv1iwE9tkAJNkSANUCBFQcxVYswlGZIrWQn83/sN967VurbjIDbqB9vi
zJwz8537OfrtxVdfXblhqrML03j18qvf8AGf6rap21S5aUrj5Ka6a/drWF11MeHnKyKEuLrefe3ddFe+Lpv1OA1p
mceHZd8NU+6auhuX7f2wXI9pGJfNr3duOaRx3UwjdjRuqMcq3btmGbp2GrqmSUOF/4b1MKR2qu5VNa6612mpkvA6
yAUjTBFN5I+EUSXET8t5eUGJNZaxm7tunBZHV7/5ecT99/cc7xyTqtyUSqe8MIqlbK0LQbNovaBUZ+eCsMEI6002
xhstpE9EZudlMCYSTrQ9fvtY/5oq/wiWoEu1JvPC28361eAeviD4cNuLiGnLs+HUBuuyS5FmngJnXhPuNc/aa0GM
jzSomJmRSjrrqIzE5sBVcFJeQAxkwfwUs+Ti4xeI2s188YsABkK4sI5Lww3JWRDpY7QseK6SpUkTF50oLIOF3klp
ldM2OCGNJIoDgecB5JqdoDe6+xSrnztfjWGo++lZIJUgHwtk49ZtuCs/nkC5+wHgHaE6A1Vtcaye4ngzejeFu+fA
clpkoaJMKtLEJYwxZusDYYzYRD0xgRIvCAk5akoMU0ExwnNMKUiplL0AlpXiVNfGZj2sqqa7vQDTR+vbnvA5TjNG
09DscNkrUreensNCCaptCjFQl2BxkuZEmYOOUGqNBi5SQnskd4Qn5gFFdpGTQIwjitGsLmAhBOXqFIxuPYRU1RFW
UE+PnxqST2uC810Xu7vuMdzC9gyMksF1SVghzMkYlZSJkVthJcneiGRI8IgB2VMTLGWWJO58ZIRSBQLUxQswGrL1
+C+2QF6F7j4N7jZVTb2qj0NrcCssVPWqLM4PR+itgEId3ZQiNmbXjGkrDRhZeN13NeBoOhfP1pv6PlVAuukeAN50
B3jvuiZWYejGsW5vq84D+/vnzw1pvuUjhFDwvrC1TW8mbP1lXQ/wKbe4ZAGyd/OPXc51qF1TuTZik4PbK1xz10W/
3twguBaSBkw7et2wwmt/nWPzxYdDP+rwWJUbFSKXto0hIU+J9S2CfTG0W9zJP25dDvzfs9t7JAizc9yeK3fvh1SN
w7qtVmmCD57cCctpWO9ITG58XY3rENI4ls1DHc5vt1eDXLflrSdEgN04PvkydHEd0nCkJgeTwGKumxl1B3cyNumh
Ir2V1S1ce48NA1iTnslj4yk771PTBdjFXtCmCk1yLatWUEo4T1JFt+qBL+0ZzG0noRoPdDnh4BHrHii5A8n+7nF8
M+t2NXYNiNdTGig4jXVcQx/OWD652+7uB647jxC6FajOPm9j9z7JnFX0ShOridbCwwNKEmTEdycie0qhd4VNoeBV
RIIS4QclJ0pFJbOy1oRMUwic5wAPkGzOTykMqe8+wM0PYfk9Tv5Q/3NxeObiXi0uea6n/MBjBpkzgzCfqTCGs6yL
l+LCE+tt9Ix5PJlGSewuj9o5671/27vqTdgfl8W93vSPMxZSIIhIZLMhWKt8zhExI+kSArh3LJLINWJqcIwZQ6Ql
WgXDbEjJOKPykevbEt9YWUq5WFQ6CvlbjpwkkkBAkyA0QYqjgvfZIHUULiiaHUs0RCIRqDKyxBBFDCJ7FmXwSfHj
ZAcIb7EqwSLXt1sWQmfqpMtKw5WTpL3Uihnk9Fb7KIKknnuSOaWcWfh6LiVNxDPjCfJ469XzLMqTzg1wyzJJYRUx
UVHjIiIOcjeE3jQ/VqJukMhOKJI6nZEfq4wUJVioMGHBeIWD78USCggHu+VIk9da0xBY5pllIVF8WAZRqRB98Fn5
wEhOMQmOFFJbYghlGcwt7keou8xxZ4XZ1UhVUlWC0e6dWSmXJc5no7jisBbYICFWGKcEo8IS6wTy/ohtkboMfLnV
MTletNiT/80V7naqVzt+CMhaQNsDg254KL3mwE1oHqXQGsWFZREVBsK0coi3zEIrEexlll5kpISX+e392c4SAhcR
JR3sizEdKJXCw77wXggyW67B2VkSXSQkCUNYSj6z6GbArX6HBLcv2sf1HZSokKxwXDvqkaTCrhSkmSi+UCgJFcW7
WWrwVJkjfoK5GxGQ35WbGP88w6FDiJ0QxHJOxeGUMFy/3rI03DPpiYCIUHtF6V0BSju4U8eFwT2SUSKhzAjwpyZw
1Bk+xVxKjUgyuXqSzxxFgoue/DzpR5Dd5mTVPZ1fgcLuvk4PJ6HuzrUlbpdEsiRK/9499l/f/fD3b7/75uZe3azi
ufvZxdbnPNA+eSxw7E+WlsC4LH9X7314Pvufrb/F3ZFrhnQUpLJw0CHk3NF7l5k32nHjBeo1bgmMwmLRZLtzZgcS
hyj1oXFuT+JThakdwcX2MUfxikMr1U9nnHcBSwSBt0PTIqKJltAolmA5XisJS4rBEO1Kz2BHYAf7cByswmwz7aZ3
gsDHKKL1/s/1u8uP1fsdLg2t12XPO+oWSp+iB63/OYVp/qFqeris+/4TQbr8eDU+lDPZOs8Et4aZUicLeDGXDS3V
n1EIOwi6UgbK4cCYUIhRmnPqZaJCMwTlcKmDhZpRqCceYOvbTlNUV7e45VzMXJYo0gnotPg4iV48/D4SfZc5BIfi
au9Mt/8uSsK0COvoKF9k+H8n6c34y/h8fyJKxAyU3N576Fxw2QuHvIVkUSpMhehJk0CgBDFUmxTiQPWO9EYqz5lw
l7ph0A/KiwhOCvNSU1W7omouZN6JuEAceyfi5DLiSllBPz3ipazam8vf1n55/KTl/KSb9Rjdc1gbFiLS1yQYs1x6
1OwBSZfTXsFtSqQjyGklw25RVIVhF6PI1OCfkD0Lc6kXhDSGnsI8Qiviujn1UA4FdnjaBy/fQ+jWGyd+5CH2vA4n
Ed6AzzhnH3UDHG9f0evQr19RdX2LheVt+f/1Kq1eWfXNdQsxvaLHdFLjehSNKIhhdnHu+VF9vN6WgvKqeJkF0Qsi
fqT2JeMvhT6h8gb1WtjpwEtyvFYK4nomsg3ZTxdbt9pUgufdq+Ot7ZZ833UNWUAYVB0v92XcsEXy6rT1N2cHv6wR
nn83XOMENueAUPOSkyf7Nv2Mv3z3j++//frHr/96tV18u1eYzSV/t9w/c2nfnMniWOZnix8i5XcL+I8SW3ozpeH/
yV5vti++IMLz1c9fhqUj96dbHvmj5UYuiqy9/zPlJd5XXqcBtHRRZ6IVypNwd5SbU23ZITffZ5Yl5K5Kb3EYN7en
289dPzMEsTa6pmvTeYXZtc1j1XbVUaO8G6q5Sbtpzm6qzjG5ppSYQ3efWoe8/JDGjlCp6bOeNOLWq64IdnG4/wJ6
spt3LOcnXB5zc5T5yCi9cdJygQSVcaeE1dYYTTk3Ad+T0syhcgqCI3WM0VKGzcZadbFIMIx/+NDxC8IUr5m/bF7z
2c8pvxxkyzO+iDnnlwPp6SD0cxiAlnCxnoPNfjp27r17hNNSzk4PJx78ZGS4O1+mheN6tXLDQVgowUvUwOYygux7
xIKA+D/f6Pp8xzw6m4edxzuO+5knJKh6ZktoupIfxHqc6jZMVUzjPAktnZtN7+TimVyms1VsxplPvQXu+mzg5zzU
qiwe1tZ9CV/1ql83Y9q360e3wvfjnRH5y2NBunB8+px9LX+1cm/q1Xp1mBwfRoUOuU8heKM1Mh+KQlpIbVBiPzk6
d6rw15xxVGkYEHZX80HkJeWgpbJMH6Rm6rmjQ7f57azt0Zhu58Nz+KGcGaopCPD90LjNboII4f1K77ZozUmv+C65
2JSR7G2TH0pCAAH1wKJ6cEM7b94rbtcXvqDV3r6ZVbIMuOdbHG1a1ZtGS2lWFyuvtnP7VRFhmd9OO8qHM6m+vYMO
hakM1/HP2k2gWfVwGLjzfWn5gij+ncevVezyuCdy0mZG3gNc2rixhoOUrd1tGAAADOGJhIXaq9N+x6aj6KCc1X7u
vIf19DcAjhVpOwAuHeby+G3OvSthrqauCONYZ08uIo1UJzsvvUgRwk82zrP53fB/Sv3RXs3LLzm9ffH2xX8B3A5k
QbcoAAA=
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


def _attestation_bytes() -> bytes:
    try:
        compressed = base64.b64decode(
            b"".join(_CONTROLLER_SMOKE_ATTESTATION_GZIP_BASE64.split()),
            validate=True,
        )
        data = gzip.decompress(compressed)
    except (ValueError, gzip.BadGzipFile) as exc:
        raise RuntimeError(
            "Embedded v6 controller-smoke attestation is invalid"
        ) from exc
    if len(data) != CONTROLLER_SMOKE_ATTESTATION_SIZE_BYTES:
        raise RuntimeError("Embedded v6 controller-smoke attestation size drift")
    if hashlib.sha256(data).hexdigest() != CONTROLLER_SMOKE_ATTESTATION_SHA256:
        raise RuntimeError("Embedded v6 controller-smoke attestation digest drift")
    return data


def canonical_controller_smoke_attestation_bytes() -> bytes:
    """Return the exact immutable promotion-attestation bytes."""

    return _attestation_bytes()


def _expected_validation_summary() -> dict[str, Any]:
    return {
        "adversarial_apply_calls": 8,
        "adversarial_slew_events": 8,
        "concurrent_apply_calls": 168,
        "concurrent_closed_distinct_desired_poses": 10,
        "concurrent_closed_fresh_dls_applies": 80,
        "controller_aborts": 0,
        "coupled_impulse_failure_samples": 0,
        "delayed_close_apply_calls": 1000,
        "maximum_follower_velocity_rad_s": 0.7730712294578552,
        "maximum_pose_position_error_m": 0.001071291510015726,
        "maximum_pose_rotation_error_deg": 0.4384913281711513,
        "nonfatal_log_findings": [
            "headless_glfw_no_display_warnings",
            "optional_ngx_context_errors",
            "missing_viewport_camera_mesh_asset_warning",
            "eight_active_actuators_plus_five_passive_mimic_dofs_warning",
        ],
        "open_endpoint_samples": 99,
        "ordinary_apply_calls": 4680,
        "ordinary_pose_cases_passed": 13,
        "recovery_events": 0,
        "safety_report_count": 17,
        "total_controller_apply_calls": 5856,
        "total_open_endpoint_samples": 6003,
        "total_post_policy_step_samples": 732,
    }


def _expected_coverage_limits() -> dict[str, Any]:
    return {
        "camera_image_contract_validated": False,
        "checkpoint_loaded": False,
        "live_follower_threshold_crossing_observed": False,
        "live_recovery_event_observed": False,
        "next_required_gate": NEXT_STAGE,
        "normalization_validated": False,
        "policy_serving_validated": False,
        "scene_digest_logged_by_smoke_job": False,
        "scene_post_job_digest_and_pre_srun_metadata_validated": True,
        "task_success_metric_validated": False,
    }


def _expected_sealed_provenance() -> dict[str, Any]:
    root = (
        "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/"
        "controller_concurrent_v6_smoke/6e4b7c5-20260705T021644Z/"
        "promotion-provenance-job-1098922"
    )
    file_names = {
        "sacct": "sacct.json",
        "saved_job_script": "saved-job-script.sbatch",
        "slurm_log": "slurm.out",
        "source_identity": "source-identity.sha256",
    }
    return {
        name: {
            "mode": mode,
            "path": f"{root}/{file_names[name]}",
            "sha256": digest,
            "size_bytes": size_bytes,
        }
        for name, mode, size_bytes, digest in SEALED_PROVENANCE
    }


def _expected_controller_smoke_attestation() -> dict[str, Any]:
    value = _strict_json(_attestation_bytes(), "embedded controller-smoke attestation")
    expected_sources = dict(PRODUCER_SOURCE_SHA256)
    producer = value.get("producer")
    reviewer = value.get("reviewer")
    if type(producer) is not dict or type(reviewer) is not dict:
        raise RuntimeError("Embedded controller-smoke lineage schema drift")
    if producer != {
        "controller_profile": CONTROLLER_PROFILE,
        "ik_safety_profile": IK_SAFETY_PROFILE,
        "polaris_commit": PRODUCER_COMMIT,
        "polaris_parent": PRODUCER_PARENT,
        "polaris_repo": (
            "/lustre/fsw/portfolios/nvr/users/lzha/src/"
            "PolaRiS-concurrent-v6-6e4b7c5-20260705T021644Z"
        ),
        "polaris_tree": PRODUCER_TREE,
        "source_sha256": expected_sources,
    }:
        raise RuntimeError("Embedded controller-smoke producer identity drift")
    if reviewer.get("evidence_commit") != EVIDENCE_COMMIT:
        raise RuntimeError("Embedded controller-smoke evidence commit drift")
    if reviewer.get("evidence_tree") != EVIDENCE_TREE:
        raise RuntimeError("Embedded controller-smoke evidence tree drift")
    if reviewer.get("evidence_parent") != EVIDENCE_PARENT:
        raise RuntimeError("Embedded controller-smoke evidence parent drift")
    finalizer = reviewer.get("finalizer")
    if type(finalizer) is not dict or (
        finalizer.get("sha256") != FINALIZER_SHA256
        or finalizer.get("size_bytes") != FINALIZER_SIZE_BYTES
        or finalizer.get("mode") != "0644"
        or finalizer.get("nlink") != 1
    ):
        raise RuntimeError("Embedded controller-smoke finalizer identity drift")
    if value.get("sealed_provenance") != _expected_sealed_provenance():
        raise RuntimeError("Embedded controller-smoke sealed provenance drift")
    if value.get("validation_summary") != _expected_validation_summary():
        raise RuntimeError("Embedded controller-smoke validation summary drift")
    if value.get("coverage_limits") != _expected_coverage_limits():
        raise RuntimeError("Embedded controller-smoke coverage limits drift")
    return value


def canonical_paired_checkpoint_canary_request() -> dict[str, Any]:
    """Return the only request authorized by this promotion gate."""

    return {
        "eval_scale": NEXT_EVAL_SCALE,
        "stage": NEXT_STAGE,
        "benchmark": "polaris_droid_suite_v1",
        "task": CANARY_TASK,
        "checkpoint_roles": list(CHECKPOINT_ROLES),
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
                "image_resolution": [224, 224],
                "state_encoding": "EEF_R6",
                "state_layout": "xyz+r6_first_two_rows+gripper_open",
                "state_layout_mode": "public_lap_train_matched_rows_v1",
                "frame_description": "robot base frame",
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
                "image_resolution": [224, 224],
                "state_encoding": "EEF_R6",
                "state_layout": "xyz+r6_first_two_columns+gripper_open",
                "state_layout_mode": "manifest_train_matched_columns_v1",
                "frame_description": "egocentric frame",
            },
        },
        "shared_train_eval_contract": {
            "image_color_space": "RGB",
            "image_dtype": "uint8",
            "wrist_image_preprocessing": {
                "operation_order": ["resize_with_pad", "rotate_180"],
                "rotation_degrees": 180,
            },
            "normalization": {
                "source": "checkpoint_assets",
                "type": "bounds_q99",
                "scope": "global",
                "policy_category": "single_arm",
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
    }


def canonical_eef_concurrent_arm_gripper_v6_promotion_evidence() -> dict[str, Any]:
    """Return the closed controller-smoke evidence and bounded authorization."""

    attestation = _expected_controller_smoke_attestation()
    return {
        "schema_version": 1,
        "profile": PROMOTION_PROFILE,
        "status": PROMOTION_STATUS,
        "lineage": {
            "producer_commit": PRODUCER_COMMIT,
            "producer_tree": PRODUCER_TREE,
            "producer_parent": PRODUCER_PARENT,
            "evidence_commit": EVIDENCE_COMMIT,
            "evidence_tree": EVIDENCE_TREE,
            "evidence_parent": EVIDENCE_PARENT,
            "finalizer_path": FINALIZER_PATH,
            "finalizer_sha256": FINALIZER_SHA256,
            "finalizer_size_bytes": FINALIZER_SIZE_BYTES,
            "producer_source_sha256": dict(PRODUCER_SOURCE_SHA256),
            "preserved_v5_source_sha256": dict(PRESERVED_V5_SOURCE_SHA256),
            "controller_or_evaluator_semantics_changed_by_promotion": False,
        },
        "controller_smoke": {
            "attestation_identity": {
                "path": CONTROLLER_SMOKE_ATTESTATION_PATH,
                "sha256": CONTROLLER_SMOKE_ATTESTATION_SHA256,
                "size_bytes": CONTROLLER_SMOKE_ATTESTATION_SIZE_BYTES,
                "mode": "0444",
                "nlink": CONTROLLER_SMOKE_ATTESTATION_NLINK,
            },
            "attestation": attestation,
            "validation_summary": _expected_validation_summary(),
            "coverage_limits": _expected_coverage_limits(),
        },
        "authorization": {
            "allowed_eval_scales": list(AUTHORIZED_EVAL_SCALES),
            "next_eval_scale": NEXT_EVAL_SCALE,
            "next_stage": NEXT_STAGE,
            "paired_checkpoint_canary_request": (
                canonical_paired_checkpoint_canary_request()
            ),
            "canary_authorized": True,
            "smoke_suite_authorized": False,
            "standard_authorized": False,
            "requires_exact_manifest_validation": True,
            "requires_exact_attestation_validation": True,
            "requires_exact_source_identity_validation": True,
            "controller_or_evaluator_behavior_change_authorized": False,
            "controller_smoke_validation_claims": {
                "checkpoint_loaded": False,
                "policy_serving_validated": False,
                "camera_image_contract_validated": False,
                "image_order_or_resolution_validated": False,
                "normalization_validated": False,
                "task_success_metric_validated": False,
            },
        },
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    _validate_json_tree(value)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


PROMOTION_EVIDENCE_SHA256 = _canonical_sha256(
    canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
)
EXPECTED_PROMOTION_EVIDENCE_SHA256 = (
    "714b22a185ff06135cdc84d03a17347943c405b3d782f3a0141455f0194eb937"
)
if PROMOTION_EVIDENCE_SHA256 != EXPECTED_PROMOTION_EVIDENCE_SHA256:
    raise RuntimeError(
        "Canonical v6 checkpoint-canary promotion evidence drift: "
        f"{PROMOTION_EVIDENCE_SHA256}"
    )


def validate_eef_concurrent_arm_gripper_v6_promotion_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject any structural, value, or exact-type drift in the evidence."""

    _validate_json_tree(value)
    expected = canonical_eef_concurrent_arm_gripper_v6_promotion_evidence()
    if type(value) is not dict or value != expected:
        raise ValueError("V6 checkpoint-canary promotion evidence drift")
    if _canonical_sha256(value) != EXPECTED_PROMOTION_EVIDENCE_SHA256:
        raise ValueError("V6 checkpoint-canary promotion digest drift")
    return deepcopy(expected)


def validate_paired_checkpoint_canary_request(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject anything except the exact paired one-rollout FoodBussing request."""

    _validate_json_tree(value, "canary request")
    expected = canonical_paired_checkpoint_canary_request()
    if (
        type(value) is not dict
        or value != expected
        or _canonical_sha256(value) != _canonical_sha256(expected)
    ):
        raise ValueError("V6 paired checkpoint-canary request drift")
    return deepcopy(expected)


def eef_concurrent_arm_gripper_v6_eval_scale_allowed(eval_scale: str) -> bool:
    """Declare only canary scale eligible; this does not itself authorize launch."""

    return type(eval_scale) is str and eval_scale in AUTHORIZED_EVAL_SCALES


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


def validate_controller_smoke_attestation(
    path: Path, *, allow_content_addressed_mirror: bool = False
) -> dict[str, Any]:
    """Freshly bind the exact immutable job-1098922 promotion attestation.

    The production default requires the canonical result path.  Tests and
    offline audits may opt into an explicit content-addressed mirror; the same
    mode, single-link, size, byte, and SHA checks still apply.
    """

    if not isinstance(path, Path):
        raise ValueError("Controller-smoke attestation path must be pathlib.Path")
    if type(allow_content_addressed_mirror) is not bool:
        raise ValueError("Attestation mirror selection must be an exact boolean")
    if (
        not allow_content_addressed_mirror
        and str(path) != CONTROLLER_SMOKE_ATTESTATION_PATH
    ):
        raise ValueError("Controller-smoke promotion attestation path drift")
    try:
        before_lstat = path.lstat()
    except OSError as exc:
        raise ValueError(
            "Controller-smoke promotion attestation is unavailable"
        ) from exc
    if not stat.S_ISREG(before_lstat.st_mode) or stat.S_ISLNK(before_lstat.st_mode):
        raise ValueError("Controller-smoke promotion attestation is not regular")
    if stat.S_IMODE(before_lstat.st_mode) != CONTROLLER_SMOKE_ATTESTATION_MODE:
        raise ValueError("Controller-smoke promotion attestation mode drift")
    if before_lstat.st_nlink != CONTROLLER_SMOKE_ATTESTATION_NLINK:
        raise ValueError("Controller-smoke promotion attestation link-count drift")
    if before_lstat.st_size != CONTROLLER_SMOKE_ATTESTATION_SIZE_BYTES:
        raise ValueError("Controller-smoke promotion attestation size drift")

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError("Controller-smoke promotion attestation open failed") from exc
    try:
        before_fstat = os.fstat(fd)
        if _stat_identity(before_fstat) != _stat_identity(before_lstat):
            raise ValueError(
                "Controller-smoke promotion attestation changed before read"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_fstat = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        after_lstat = path.lstat()
    except OSError as exc:
        raise ValueError(
            "Controller-smoke promotion attestation changed after read"
        ) from exc
    if _stat_identity(before_fstat) != _stat_identity(after_fstat) or _stat_identity(
        before_lstat
    ) != _stat_identity(after_lstat):
        raise ValueError("Controller-smoke promotion attestation changed during read")
    data = b"".join(chunks)
    if hashlib.sha256(data).hexdigest() != CONTROLLER_SMOKE_ATTESTATION_SHA256:
        raise ValueError("Controller-smoke promotion attestation digest drift")
    if data != _attestation_bytes():
        raise ValueError("Controller-smoke promotion attestation byte drift")
    value = _strict_json(data, "controller-smoke promotion attestation")
    if value != _expected_controller_smoke_attestation():
        raise ValueError("Controller-smoke promotion attestation semantic drift")
    return value


def _validate_source_file(path: Path, expected_sha256: str, label: str) -> str:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ValueError(f"Missing {label}: {path}") from exc
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ValueError(f"{label} is not a regular file: {path}")
    if before.st_nlink != 1:
        raise ValueError(f"{label} must have exactly one hard link: {path}")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError(f"Could not securely open {label}: {path}") from exc
    try:
        opened = os.fstat(fd)
        if _stat_identity(before) != _stat_identity(opened):
            raise ValueError(f"{label} changed before read: {path}")
        if opened.st_nlink != 1:
            raise ValueError(f"{label} hard-link count drift: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_opened = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ValueError(f"{label} changed after read: {path}") from exc
    if _stat_identity(before) != _stat_identity(after) or _stat_identity(
        opened
    ) != _stat_identity(after_opened):
        raise ValueError(f"{label} changed during read: {path}")
    if after.st_nlink != 1 or after_opened.st_nlink != 1:
        raise ValueError(f"{label} hard-link count drift: {path}")
    digest = hashlib.sha256(b"".join(chunks)).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{label} digest drift: {path}")
    return digest


def validate_v6_promotion_source_identity(repo_root: Path) -> dict[str, Any]:
    """Bind producer/controller, Commit-A finalizer, and preserved v5 bytes."""

    if not isinstance(repo_root, Path):
        raise ValueError("Repository root must be pathlib.Path")
    producer = {
        relative: _validate_source_file(
            repo_root / relative, digest, "V6 producer source"
        )
        for relative, digest in PRODUCER_SOURCE_SHA256
    }
    finalizer = _validate_source_file(
        repo_root / FINALIZER_PATH, FINALIZER_SHA256, "Commit-A finalizer"
    )
    preserved_v5 = {
        relative: _validate_source_file(
            repo_root / relative, digest, "Preserved v5 promotion source"
        )
        for relative, digest in PRESERVED_V5_SOURCE_SHA256
    }
    return {
        "producer_source_sha256": producer,
        "finalizer_sha256": finalizer,
        "preserved_v5_source_sha256": preserved_v5,
    }


def validate_and_authorize_paired_checkpoint_canaries(
    evidence: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    attestation_path: Path,
    repo_root: Path,
    allow_content_addressed_attestation_mirror: bool = False,
) -> dict[str, Any]:
    """Authorize only the exact next canary pair after all fresh checks pass."""

    validated_evidence = validate_eef_concurrent_arm_gripper_v6_promotion_evidence(
        evidence
    )
    validated_request = validate_paired_checkpoint_canary_request(request)
    validate_controller_smoke_attestation(
        attestation_path,
        allow_content_addressed_mirror=(allow_content_addressed_attestation_mirror),
    )
    source_identity = validate_v6_promotion_source_identity(repo_root)
    authorization = validated_evidence["authorization"]
    if (
        authorization["canary_authorized"] is not True
        or authorization["smoke_suite_authorized"] is not False
        or authorization["standard_authorized"] is not False
        or authorization["allowed_eval_scales"] != ["canary"]
        or authorization["next_stage"] != NEXT_STAGE
    ):
        raise ValueError("Closed evidence does not authorize only paired canaries")
    claims = authorization["controller_smoke_validation_claims"]
    if any(claims.values()):
        raise ValueError("Controller smoke overclaims checkpoint evaluation coverage")
    return {
        "authorized": True,
        "eval_scale": "canary",
        "stage": NEXT_STAGE,
        "request": validated_request,
        "promotion_evidence_sha256": EXPECTED_PROMOTION_EVIDENCE_SHA256,
        "controller_smoke_attestation_sha256": (CONTROLLER_SMOKE_ATTESTATION_SHA256),
        "verified_attestation_path": str(attestation_path),
        "content_addressed_attestation_mirror_used": (
            allow_content_addressed_attestation_mirror
        ),
        "source_identity": source_identity,
        "checkpoint_image_normalization_and_task_validation_claimed": False,
    }
