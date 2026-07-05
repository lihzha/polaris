from __future__ import annotations

import hashlib
import importlib.util
import io
import os
from pathlib import Path
import stat
import sys

import numpy as np
from PIL import Image
import pytest


REPO_ROOT = Path(__file__).parents[1]
FINALIZER_PATH = REPO_ROOT / "scripts/finalize_splat_image_contract_smoke.py"


def _load_finalizer():
    spec = importlib.util.spec_from_file_location(
        "finalize_splat_image_contract_smoke", FINALIZER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


finalizer = _load_finalizer()


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _png_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def _scheduler_lines(job_id: int = 123) -> str:
    return "\n".join(
        [
            f"{job_id}|pol_img_9d29636|nvr_lpr_rvp|batch|COMPLETED|0:0|171|2026-07-04T23:19:11|2026-07-04T23:22:02|pool0-00010|billing=1,cpu=16,gres/gpu=1,mem=96G,node=1|billing=1,cpu=16,gres/gpu=1,mem=96G,node=1",
            f"{job_id}.batch|batch|nvr_lpr_rvp||COMPLETED|0:0|171|2026-07-04T23:19:11|2026-07-04T23:22:02|pool0-00010|cpu=16,gres/gpu=1,mem=96G,node=1|",
            f"{job_id}.extern|extern|nvr_lpr_rvp||COMPLETED|0:0|171|2026-07-04T23:19:11|2026-07-04T23:22:02|pool0-00010|billing=1,cpu=16,gres/gpu=1,mem=96G,node=1|",
            f"{job_id}.0|env|nvr_lpr_rvp||COMPLETED|0:0|160|2026-07-04T23:19:22|2026-07-04T23:22:02|pool0-00010|cpu=16,gres/gpu=1,mem=96G,node=1|",
        ]
    )


def _minimal_context(tmp_path: Path, job_id: int = 123):
    leaf = finalizer.ExpectedLeaf(1, "a" * 64, 0o444)
    return finalizer.RuntimeContext(
        job_id=job_id,
        job_name="pol_img_9d29636",
        result_root=tmp_path / "result",
        raw_path=tmp_path / "result" / f"smoke-{job_id}.raw.json",
        ready_path=tmp_path / "result" / f"smoke-{job_id}.raw.json.ready.json",
        source_identity_path=tmp_path / "result" / f"source-identity-{job_id}.sha256",
        post_srun_path=tmp_path / "result" / f"post-srun-validation-{job_id}.json",
        attestation_path=tmp_path
        / "result"
        / f"smoke-{job_id}.image-evidence-attestation.json",
        producer_repo=tmp_path / "producer",
        evidence_repo=tmp_path / "evidence",
        saved_job_script=tmp_path / "wrapper.sbatch",
        slurm_log=tmp_path / "log.out",
        sacct_snapshot=tmp_path / "sacct.json",
        raw_spec=leaf,
        ready_spec=leaf,
        source_identity_spec=leaf,
        post_srun_spec=leaf,
        saved_job_script_spec=leaf,
        slurm_log_spec=leaf,
        sacct_snapshot_spec=leaf,
        artifact_manifest_sha256="b" * 64,
        srun_start_epoch_ns=1,
    )


def _contract_fixture() -> dict[str, object]:
    return {
        "renderer_conversion": {
            "formula": "(clip(raw_float_rgb,0,1)*255).astype(uint8)",
            "pixel_exact": True,
            "shape_preserved": True,
            "channel_order": "RGB",
            "bgr_conversion": False,
        },
        "robot_compositing": {
            "formula": "np.where(robot_mask,sim_rgb,native_splat_rgb)",
            "pixel_exact": True,
            "native_shape": [720, 1280, 3],
        },
        "ego_lap_preprocessing": {
            "actual_client_class": (
                "polaris.policy.lap_eef_pose_client.EgoLAPEefPoseClient"
            ),
            "constructor_bypassed_no_network": True,
            "method": "_build_request",
            "call_events": [
                "resize:external:720x1280->224x224",
                "resize:wrist:720x1280->224x224",
                "rotate180:wrist:224x224->224x224",
            ],
            "native_shape": [720, 1280, 3],
            "resized_content_shape": [126, 224, 3],
            "preprocessed_shape": [224, 224, 3],
            "padding_rows": {"top": 49, "bottom": 49},
            "wrist_operation_order": "resize_pad_then_rotate_180",
            "operation_order_probe": {
                "profile": "odd_5x8_to_7x7_asymmetric_padding_v1",
                "input_shape": [5, 8, 3],
                "target_shape": [7, 7, 3],
                "resize_then_rotate_sha256": (
                    "19f595391cdbf268f22969f676e77f373a3223efbc1c5ebb19effbddc6d81f47"
                ),
                "rotate_then_resize_sha256": (
                    "d9da8384bd84ede8ab081e1e04f12ad42759ad78e870096b2a2cc2b8c072e7a9"
                ),
                "differing_values": 99,
                "production_matches_resize_then_rotate": True,
            },
            "pixel_exact_request_binding": True,
        },
        "msgpack_roundtrip": {
            "implementation": "openpi_client.msgpack_numpy",
            "exact_arrays": True,
            "exact_image_bytes": True,
            "packed_sha256": "a" * 64,
        },
        "removed_resize_counterfactual": {
            "profile": "removed_cv2_default_linear_half_down_up_v1",
            "live_path": False,
            "required_to_change_pixels": True,
        },
    }


def test_evidence_lineage_and_changed_paths_are_exact() -> None:
    assert finalizer.PRODUCER_COMMIT == "9d296361bb323b2e309a3b92a204c102908c61a6"
    assert finalizer.PRODUCER_TREE == "2e868fd4a31a55c9cedfb3221e4c2bc1fbbb9310"
    assert finalizer.PRODUCER_PARENT == "42e266353df71d5906e98975165f8aa021020dad"
    assert finalizer.EVIDENCE_CHANGED_PATHS == {
        "WORKLOG.v6.md",
        "scripts/finalize_splat_image_contract_smoke.py",
        "tests/test_finalize_splat_image_contract_smoke.py",
    }
    assert len(finalizer.PRODUCER_SOURCE_SHA256) == 7


def test_exact_artifact_name_kind_contract_has_28_leaves() -> None:
    assert len(finalizer.ARTIFACT_KINDS) == 28
    assert set(finalizer.ARTIFACT_KINDS.values()) == {"npy", "png", "msgpack"}
    assert finalizer.ARTIFACT_KINDS["ego_lap_request_msgpack"] == "msgpack"
    for camera in finalizer.CAMERAS:
        assert finalizer.ARTIFACT_KINDS[f"{camera}_renderer_raw_float"] == "npy"
        assert finalizer.ARTIFACT_KINDS[f"{camera}_preprocessed_png"] == "png"


def test_finalizer_does_not_import_or_delegate_to_producer() -> None:
    source = FINALIZER_PATH.read_text()
    assert "import smoke_splat_image_contract" not in source
    assert "validate_passed_raw_payload" not in source
    assert (
        "isaac"
        not in "\n".join(
            line for line in source.splitlines() if line.startswith("import ")
        ).lower()
    )
    assert 'checkpoint_evaluation": False' in source
    assert 'policy_serving": False' in source
    assert 'task_metric": False' in source
    assert 'promotion": False' in source


def test_typed_equal_rejects_bool_integer_aliasing() -> None:
    assert finalizer._typed_equal({"value": True}, {"value": True})
    assert not finalizer._typed_equal({"value": True}, {"value": 1})
    assert not finalizer._typed_equal([1], [1.0])


def test_strict_json_rejects_nonfinite_constants() -> None:
    with pytest.raises(finalizer.VerificationError, match="strict JSON"):
        finalizer._strict_json(b'{"value": NaN}', "probe")


def test_secure_read_binds_mode_size_hash_and_single_link(tmp_path: Path) -> None:
    path = tmp_path / "leaf"
    path.write_bytes(b"evidence")
    path.chmod(0o444)

    data, record = finalizer._secure_read(
        path,
        "leaf",
        required_mode=0o444,
        expected_size=8,
        expected_sha256=hashlib.sha256(b"evidence").hexdigest(),
    )

    assert data == b"evidence"
    assert record.nlink == 1
    assert record.mode == "0444"


def test_secure_read_rejects_symlink_hardlink_and_mode(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    target.chmod(0o444)
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target)
    with pytest.raises(finalizer.VerificationError, match="regular file"):
        finalizer._secure_read(symlink, "symlink", required_mode=0o444)

    hardlink = tmp_path / "hardlink"
    os.link(target, hardlink)
    with pytest.raises(finalizer.VerificationError, match="hard-link"):
        finalizer._secure_read(target, "target", required_mode=0o444)
    hardlink.unlink()
    target.chmod(0o644)
    with pytest.raises(finalizer.VerificationError, match="mode"):
        finalizer._secure_read(target, "target", required_mode=0o444)


def test_secure_read_detects_path_replacement_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "leaf"
    path.write_bytes(b"old")
    path.chmod(0o444)
    original_read = os.read
    replaced = False

    def replacing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        if not replaced:
            replaced = True
            path.rename(tmp_path / "old")
            path.write_bytes(b"new")
            path.chmod(0o444)
        return original_read(descriptor, count)

    monkeypatch.setattr(finalizer.os, "read", replacing_read)
    with pytest.raises(finalizer.VerificationError, match="changed"):
        finalizer._secure_read(path, "leaf", required_mode=0o444)


@pytest.mark.parametrize(
    ("array", "dtype", "descr"),
    [
        (np.arange(12, dtype=np.uint8).reshape(2, 2, 3), "uint8", "|u1"),
        (np.arange(6, dtype=np.int64).reshape(2, 3), "int64", "<i8"),
        (np.linspace(0, 1, 6, dtype=np.float32), "float32", "<f4"),
    ],
)
def test_parse_npy_closes_header_shape_dtype_and_payload(
    array: np.ndarray, dtype: str, descr: str
) -> None:
    parsed = finalizer.parse_npy(_npy_bytes(array), "array")
    assert parsed.dtype == dtype
    assert parsed.descr == descr
    assert parsed.shape == array.shape
    assert parsed.data == array.tobytes()


def test_parse_npy_rejects_trailing_payload() -> None:
    data = _npy_bytes(np.zeros((2, 2, 3), dtype=np.uint8)) + b"x"
    with pytest.raises(finalizer.VerificationError, match="payload size"):
        finalizer.parse_npy(data, "array")


def test_parse_png_decodes_all_lossless_rgb_pixels_and_crc() -> None:
    array = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    data = _png_bytes(array)
    parsed = finalizer.parse_png(data, "image")
    assert (parsed.height, parsed.width) == (5, 7)
    assert parsed.pixels == array.tobytes()

    corrupted = bytearray(data)
    corrupted[-8] ^= 1
    with pytest.raises(finalizer.VerificationError, match="CRC"):
        finalizer.parse_png(bytes(corrupted), "image")


def test_msgpack_decoder_and_request_validation_are_pixel_exact() -> None:
    from openpi_client import msgpack_numpy

    external = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    wrist = external[::-1, ::-1].copy()
    state = np.arange(10, dtype=np.float32)
    request = {
        "observation": {
            "base_0_rgb": external,
            "left_wrist_0_rgb": wrist,
            "cartesian_position": state[:9],
            "gripper_position": state[9:],
            "state": state,
        },
        "prompt": "Put all the foods in the bowl",
        "frame_description": "image contract smoke; no checkpoint or policy server",
        "eef_frame": "panda_link8",
        "dataset_name": "droid",
        "state_type": "eef_pose",
        "has_wrist_image": True,
        "is_bimanual": False,
        "rotation_applied": True,
    }
    packed = msgpack_numpy.packb(request)

    summary = finalizer._validate_msgpack_request(
        packed, external.tobytes(), wrist.tobytes()
    )

    assert (
        summary["external_image_sha256"]
        == hashlib.sha256(external.tobytes()).hexdigest()
    )
    tampered = wrist.copy()
    tampered[0, 0, 0] ^= 1
    with pytest.raises(finalizer.VerificationError, match="pixels"):
        finalizer._validate_msgpack_request(
            packed, external.tobytes(), tampered.tobytes()
        )


def test_independent_resize_matches_actual_client_and_wrist_order() -> None:
    from polaris.policy.lap_eef_pose_client import (
        preprocess_lap_wrist_image,
        resize_lap_image,
    )

    image = (
        np.arange(720 * 1280 * 3, dtype=np.uint32).reshape(720, 1280, 3) % 251
    ).astype(np.uint8)
    independent, content, remainder = finalizer.resize_with_pad_rgb8(
        image.tobytes(), 720, 1280, 224, 224
    )
    assert content == (126, 224)
    assert remainder == (49, 0)
    np.testing.assert_array_equal(
        np.frombuffer(independent, dtype=np.uint8).reshape(224, 224, 3),
        resize_lap_image(image),
    )
    np.testing.assert_array_equal(
        np.frombuffer(finalizer.rotate_rgb8_180(independent), dtype=np.uint8).reshape(
            224, 224, 3
        ),
        preprocess_lap_wrist_image(image, rotate_180=True),
    )


def test_noncommuting_order_probe_is_reconstructed_exactly() -> None:
    expected = {
        "profile": "odd_5x8_to_7x7_asymmetric_padding_v1",
        "input_shape": [5, 8, 3],
        "target_shape": [7, 7, 3],
        "resize_then_rotate_sha256": "19f595391cdbf268f22969f676e77f373a3223efbc1c5ebb19effbddc6d81f47",
        "rotate_then_resize_sha256": "d9da8384bd84ede8ab081e1e04f12ad42759ad78e870096b2a2cc2b8c072e7a9",
        "differing_values": 99,
        "production_matches_resize_then_rotate": True,
    }
    assert finalizer._validate_order_probe(expected) == expected
    expected["differing_values"] = 98
    with pytest.raises(finalizer.VerificationError, match="probe drift"):
        finalizer._validate_order_probe(expected)


def test_contract_schema_is_independently_closed() -> None:
    fixture = _contract_fixture()
    validated = finalizer._validate_contracts(fixture)
    assert validated["renderer_conversion"]["channel_order"] == "RGB"
    assert (
        validated["ego_lap_preprocessing"]["operation_order_probe"]["differing_values"]
        == 99
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("renderer_conversion", "pixel_exact", False),
        ("renderer_conversion", "channel_order", "BGR"),
        ("robot_compositing", "pixel_exact", False),
        ("ego_lap_preprocessing", "constructor_bypassed_no_network", False),
        ("ego_lap_preprocessing", "wrist_operation_order", "rotate_then_resize"),
        ("ego_lap_preprocessing", "padding_rows", {"top": 48, "bottom": 50}),
        ("removed_resize_counterfactual", "live_path", True),
        ("removed_resize_counterfactual", "required_to_change_pixels", False),
    ],
)
def test_contract_schema_rejects_semantic_tamper(
    section: str, key: str, value: object
) -> None:
    fixture = _contract_fixture()
    fixture[section][key] = value
    with pytest.raises(finalizer.VerificationError):
        finalizer._validate_contracts(fixture)


def test_environment_binds_scene_initial_condition_zero_and_hub_metadata() -> None:
    value = {
        "id": "DROID-FoodBussing",
        "runtime_class": (
            "polaris.environments.manager_based_rl_splat_environment."
            "ManagerBasedRLSplatEnv"
        ),
        "scene_file": {
            "path": str(finalizer.SCENE_PATH),
            "size_bytes": 14914,
            "sha256": finalizer.SCENE_SHA256,
            "mode": "0640",
        },
        "initial_conditions_file": {
            "path": str(finalizer.INITIAL_CONDITIONS_PATH),
            "size_bytes": 173951,
            "sha256": finalizer.INITIAL_CONDITIONS_SHA256,
            "mode": "0640",
        },
        "initial_condition_index": 0,
        "instruction": "Put all the foods in the bowl",
        "hub_revision": finalizer.HUB_REVISION,
        "hub_metadata": {
            "initial_conditions": {
                "path": str(finalizer.INITIAL_METADATA_PATH),
                "size_bytes": 101,
                "sha256": finalizer.INITIAL_METADATA_SHA256,
                "mode": "0640",
            },
            "scene": {
                "path": str(finalizer.SCENE_METADATA_PATH),
                "size_bytes": 101,
                "sha256": finalizer.SCENE_METADATA_SHA256,
                "mode": "0640",
            },
        },
        "camera_sensor_keys": ["external_cam", "wrist_cam"],
        "renderer_camera_keys": ["external_cam", "wrist_cam"],
    }
    assert finalizer._validate_environment(value) == value
    value["initial_condition_index"] = 1
    with pytest.raises(finalizer.VerificationError, match="identity drift"):
        finalizer._validate_environment(value)


def test_renderer_conversion_compositing_and_counterfactual_are_independent() -> None:
    raw = np.linspace(-0.1, 1.1, 720 * 1280 * 3, dtype=np.float32).reshape(
        finalizer.NATIVE_SHAPE
    )
    native = (np.clip(raw, 0.0, 1.0) * 255).astype(np.uint8)
    robot = np.full(finalizer.NATIVE_SHAPE, 17, dtype=np.uint8)
    mask = np.zeros((720, 1280, 1), dtype=np.int64)
    mask[20:40, 30:50] = 1
    composited = np.where(mask, robot, native)
    old = native.copy()
    old[0, 0] ^= 1
    difference = np.abs(native.astype(np.int16) - old.astype(np.int16)).astype(np.uint8)
    arrays = {
        "raw": finalizer.parse_npy(_npy_bytes(raw), "raw"),
        "native": finalizer.parse_npy(_npy_bytes(native), "native"),
        "robot": finalizer.parse_npy(_npy_bytes(robot), "robot"),
        "mask": finalizer.parse_npy(_npy_bytes(mask), "mask"),
        "composited": finalizer.parse_npy(_npy_bytes(composited), "composited"),
        "old": finalizer.parse_npy(_npy_bytes(old), "old"),
        "difference": finalizer.parse_npy(_npy_bytes(difference), "difference"),
    }
    assert (
        finalizer._validate_renderer_conversion(
            arrays["raw"], arrays["native"], "external_cam"
        )
        > 0
    )
    assert finalizer._validate_compositing(
        arrays["native"],
        arrays["robot"],
        arrays["mask"],
        arrays["composited"],
        "external_cam",
    ) == (400, 921200)
    metrics = finalizer._validate_counterfactual(
        arrays["native"], arrays["old"], arrays["difference"], "external_cam"
    )
    assert metrics["changed_values"] == 3
    tampered = bytearray(arrays["composited"].data)
    tampered[-1] ^= 1
    bad = finalizer.NpyArray(
        descr="|u1",
        dtype="uint8",
        shape=finalizer.NATIVE_SHAPE,
        itemsize=1,
        data=bytes(tampered),
    )
    with pytest.raises(finalizer.VerificationError, match="compositing pixel"):
        finalizer._validate_compositing(
            arrays["native"], arrays["robot"], arrays["mask"], bad, "external_cam"
        )


def test_scheduler_parser_requires_exact_complete_zero_four_row_lifecycle(
    tmp_path: Path,
) -> None:
    context = _minimal_context(tmp_path)
    parsed = finalizer._parse_sacct_lines(_scheduler_lines(), context)
    assert set(parsed) == {"allocation", "batch", "extern", "srun"}
    assert parsed["srun"]["elapsed_seconds"] == 160

    failed = _scheduler_lines().replace("|COMPLETED|0:0|171|", "|FAILED|1:0|171|", 1)
    with pytest.raises(finalizer.VerificationError, match="state"):
        finalizer._parse_sacct_lines(failed, context)
    with pytest.raises(finalizer.VerificationError, match="rows"):
        finalizer._parse_sacct_lines(_scheduler_lines() + "\n123.1|extra", context)


def test_publish_is_nonoverwriting_mode_0444_single_link(tmp_path: Path) -> None:
    path = tmp_path / "attestation.json"
    payload = b'{"schema_version": 1}\n'
    finalizer._publish_nonoverwriting(path, payload)
    status = path.stat()
    assert path.read_bytes() == payload
    assert stat.S_IMODE(status.st_mode) == 0o444
    assert status.st_nlink == 1
    with pytest.raises(FileExistsError):
        finalizer._publish_nonoverwriting(path, payload)


def test_cli_exposes_only_separate_finalize_and_verify_modes() -> None:
    parser = finalizer._parser()
    mode = next(action for action in parser._actions if action.dest == "mode")
    assert tuple(mode.choices) == ("finalize", "verify")
    options = {option for action in parser._actions for option in action.option_strings}
    for required in (
        "--job-id",
        "--result-root",
        "--expected-artifact-manifest-sha256",
        "--sacct-snapshot",
        "--expected-evidence-commit",
        "--expected-evidence-tree",
        "--expected-finalizer-sha256",
    ):
        assert required in options


def test_attestation_claim_boundary_is_explicitly_false() -> None:
    source = FINALIZER_PATH.read_text()
    for field in (
        "checkpoint_evaluation",
        "policy_serving",
        "task_metric",
        "benchmark_result",
        "controller_behavior",
        "canary",
        "smoke_suite",
        "standard_suite",
        "promotion",
    ):
        assert f'"{field}": False' in source
    assert "camera_2" not in source
    assert "reasoning blank" not in source.lower()
