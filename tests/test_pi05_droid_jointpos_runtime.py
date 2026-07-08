import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import polaris.pi05_droid_jointpos_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]
GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"
ELF_FIXTURE = Path("/lib/x86_64-linux-gnu/libz.so.1")
CANONICAL_ICD_BYTES = (
    b"{\n"
    b'    "file_format_version" : "1.0.1",\n'
    b'    "ICD": {\n'
    b'        "library_path": "libGLX_nvidia.so.0",\n'
    b'        "api_version" : "1.4.312"\n'
    b"    }\n"
    b"}\n"
)


def _execution_environment():
    value = {
        "nvidia_smi": {
            "query": list(runtime.PI05_DROID_JOINTPOS_NVIDIA_SMI_QUERY),
            "uuid": GPU_UUID,
            "name": runtime.PI05_DROID_JOINTPOS_NVIDIA_GPU_NAME,
            "driver_version": runtime.PI05_DROID_JOINTPOS_NVIDIA_DRIVER_VERSION,
        },
        "vulkan": {
            "vk_driver_files": (runtime.PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH),
            "icd": {
                "path": runtime.PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH,
                "size": runtime.PI05_DROID_JOINTPOS_VULKAN_ICD_SIZE,
                "sha256": runtime.PI05_DROID_JOINTPOS_VULKAN_ICD_SHA256,
            },
        },
        "graphics_runtime": _graphics_runtime(),
    }
    return value


def _graphics_runtime():
    libraries = [
        {
            "path": path,
            "size": size,
            "sha256": sha256,
            "elf_gnu_build_id": build_id,
            "maps_device": "0:1",
            "maps_inode": index + 1,
        }
        for index, (path, size, sha256, build_id) in enumerate(
            runtime.PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES
        )
    ]
    value = {
        "profile": runtime.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE,
        "proc_maps_path": runtime.PI05_DROID_JOINTPOS_GRAPHICS_PROC_MAPS_PATH,
        "environment": {
            "LD_LIBRARY_PATH": (
                runtime.PI05_DROID_JOINTPOS_GRAPHICS_EXPECTED_LD_LIBRARY_PATH
            ),
            "NVIDIA_VISIBLE_DEVICES": GPU_UUID,
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            **{
                name: None
                for name in runtime.PI05_DROID_JOINTPOS_GRAPHICS_FORBIDDEN_ENVIRONMENT
            },
        },
        "libraries": libraries,
    }
    value["graphics_runtime_sha256"] = runtime._graphics_runtime_sha256(value)
    return value


def _write_canonical_icd(path: Path):
    path.write_bytes(CANONICAL_ICD_BYTES)
    assert path.stat().st_size == runtime.PI05_DROID_JOINTPOS_VULKAN_ICD_SIZE
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        runtime.PI05_DROID_JOINTPOS_VULKAN_ICD_SHA256
    )


def _write_maps(path: Path, library: Path, *, deleted=False, device=None, inode=None):
    metadata = library.lstat()
    device = (
        device or f"{os.major(metadata.st_dev):02x}:{os.minor(metadata.st_dev):02x}"
    )
    inode = metadata.st_ino if inode is None else inode
    suffix = " (deleted)" if deleted else ""
    path.write_text(
        f"7f000000-7f001000 r-xp 00000000 {device} {inode} {library}{suffix}\n",
        encoding="utf-8",
    )


def _prepare_graphics_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict]:
    library = tmp_path / "libvulkan.so.1"
    shutil.copyfile(ELF_FIXTURE, library)
    metadata = library.stat()
    descriptor = os.open(library, os.O_RDONLY)
    try:
        build_id = runtime._elf_gnu_build_id(descriptor, metadata.st_size)
    finally:
        os.close(descriptor)
    identity = (
        str(library),
        metadata.st_size,
        hashlib.sha256(library.read_bytes()).hexdigest(),
        build_id,
    )
    monkeypatch.setattr(
        runtime, "PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES", (identity,)
    )
    maps = tmp_path / "maps"
    _write_maps(maps, library)
    monkeypatch.setattr(
        runtime, "PI05_DROID_JOINTPOS_GRAPHICS_PROC_MAPS_PATH", str(maps)
    )
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", GPU_UUID)
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "all")
    for name in runtime.PI05_DROID_JOINTPOS_GRAPHICS_FORBIDDEN_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    device = f"{os.major(metadata.st_dev):x}:{os.minor(metadata.st_dev):x}"
    value = {
        "profile": runtime.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE,
        "proc_maps_path": str(maps),
        "environment": runtime._graphics_environment(),
        "libraries": [
            {
                "path": str(library),
                "size": metadata.st_size,
                "sha256": identity[2],
                "elf_gnu_build_id": build_id,
                "maps_device": device,
                "maps_inode": metadata.st_ino,
            }
        ],
    }
    value["graphics_runtime_sha256"] = runtime._graphics_runtime_sha256(value)
    monkeypatch.setattr(
        runtime,
        "PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_SHA256",
        value["graphics_runtime_sha256"],
    )
    return maps, library, value


def _runtime_report():
    report = {
        "schema_version": runtime.PI05_DROID_JOINTPOS_RUNTIME_SCHEMA_VERSION,
        "profile": runtime.PI05_DROID_JOINTPOS_PROFILE,
        "status": "pass",
        "execution_environment": _execution_environment(),
        "boundary": {
            "profile": runtime.PI05_DROID_JOINTPOS_BOUNDARY_PROFILE,
            "outer_steps": 450,
            "internal_max_episode_steps": 451,
            "returned_terminal_flags": "all_false",
            "terminal_rubric_source": "post_action_450_pre_autoreset_info",
        },
        "timing": {
            "physics_dt_seconds": 1 / 120,
            "physics_frequency_hz": 120,
            "decimation": 8,
            "policy_frequency_hz": 15,
        },
        "joint_names": list(runtime.PANDA_ARM_JOINT_NAMES),
        "action": {
            "term_class": runtime._ACTION_TERM_CLASS,
            "cfg_class": runtime._ACTION_CFG_CLASS,
            "base_class": runtime._ACTION_BASE_CLASS,
            "cfg_base_class": runtime._ACTION_CFG_BASE_CLASS,
            "preserve_order": True,
            "scale": 1.0,
            "offset": 0.0,
            "use_default_offset": False,
            "clip": None,
            "semantic": "absolute_joint_position_observation_only_no_guard",
            "setter_calls_per_outer_step": 8,
        },
        "observation": {
            "term_order": [
                "arm_joint_pos",
                "gripper_pos",
                "eef_pos",
                "eef_quat",
            ],
            "enable_corruption": False,
            "concatenate_terms": False,
            "state_layout": {
                "arm_joint_indices": list(range(7)),
                "gripper_joint_index": 7,
                "historical_filter_order_equivalent": True,
            },
            "terms": {
                "arm_joint_pos": {
                    "function": (
                        "polaris.environments.pi05_droid_jointpos_cfg."
                        "ordered_arm_joint_position"
                    ),
                    "noise": None,
                    "clip": None,
                },
                "gripper_pos": {
                    "function": (
                        "polaris.environments.pi05_droid_jointpos_cfg."
                        "closed_positive_gripper_position"
                    ),
                    "noise": {
                        "class": ("isaaclab.utils.noise.noise_cfg.GaussianNoiseCfg"),
                        "mean": 0.0,
                        "std": 0.05,
                        "active": False,
                    },
                    "clip": [0.0, 1.0],
                },
                "eef_pos": {
                    "function": "polaris.environments.droid_cfg.eef_pos",
                    "noise": None,
                    "clip": None,
                },
                "eef_quat": {
                    "function": "polaris.environments.droid_cfg.eef_quat",
                    "noise": None,
                    "clip": None,
                },
            },
        },
        "configured_actuators": {
            "panda_shoulder": {
                "joint_names_expr": ["panda_joint[1-4]"],
                "stiffness": 400.0,
                "damping": 80.0,
                "effort_limit": 87.0,
                "velocity_limit": 2.175,
            },
            "panda_forearm": {
                "joint_names_expr": ["panda_joint[5-7]"],
                "stiffness": 400.0,
                "damping": 80.0,
                "effort_limit": 12.0,
                "velocity_limit": 2.61,
            },
        },
        "live_actuator_and_limits": {
            "joint_stiffness": runtime._EXPECTED_STIFFNESS.tolist(),
            "joint_damping": runtime._EXPECTED_DAMPING.tolist(),
            "joint_effort_limits": runtime._EXPECTED_EFFORT.tolist(),
            "joint_velocity_limits": runtime._EXPECTED_VELOCITY.tolist(),
            "hard_joint_position_limits": runtime._EXPECTED_HARD_LIMITS.tolist(),
            "soft_joint_position_limits": runtime._EXPECTED_SOFT_LIMITS.tolist(),
        },
        "direct_physx_actuator_and_limits": {
            "joint_stiffness": runtime._EXPECTED_STIFFNESS.tolist(),
            "joint_damping": runtime._EXPECTED_DAMPING.tolist(),
            "joint_effort_limits": runtime._EXPECTED_EFFORT.tolist(),
            "joint_velocity_limits": runtime._EXPECTED_VELOCITY.tolist(),
            "hard_joint_position_limits": runtime._EXPECTED_HARD_LIMITS.tolist(),
        },
        "cameras": {
            name: {"shape": [720, 1280, 3], "dtype": "uint8"}
            for name in runtime.PI05_DROID_JOINTPOS_SENSOR_NAMES
        },
        "gripper": {
            "action_class": runtime._GRIPPER_ACTION_CLASS,
            "joint_name": "finger_joint",
            "threshold": "closed_if_gt_0p5_else_open",
            "open_target_rad": 0.0,
            "closed_target_rad": float(np.float32(np.pi / 4)),
            "observation": (
                "finger_joint_position_divided_by_pi_over_4_closed_positive"
            ),
        },
    }
    report["runtime_sha256"] = runtime.canonical_sha256(report)
    return report


def _rehash(report):
    report = copy.deepcopy(report)
    report.pop("runtime_sha256", None)
    report["runtime_sha256"] = runtime.canonical_sha256(report)
    return report


def test_execution_recorder_observes_exact_upstream_eight_hold_path():
    recorder = runtime.JointPositionExecutionRecorder()
    target = np.arange(7, dtype=np.float32)[None]
    recorder.begin_policy_step(target, target.copy())
    for _ in range(8):
        recorder.record_apply_target(target.copy())
    report = recorder.finish_policy_step(target.copy())
    assert report["apply_target_hold_count"] == 8
    assert report["processing"].endswith("no_clip")
    assert report["post_step_articulation_target"] == target[0].tolist()


def test_execution_recorder_rejects_processing_or_hold_drift():
    recorder = runtime.JointPositionExecutionRecorder()
    target = np.zeros((1, 7), dtype=np.float32)
    recorder.begin_policy_step(target, target + np.float32(0.1))
    for _ in range(8):
        recorder.record_apply_target(target)
    with pytest.raises(ValueError, match="processing changed"):
        recorder.finish_policy_step(target)

    recorder = runtime.JointPositionExecutionRecorder()
    recorder.begin_policy_step(target, target)
    for _ in range(7):
        recorder.record_apply_target(target)
    with pytest.raises(ValueError, match="exactly eight"):
        recorder.finish_policy_step(target)


def test_execution_environment_capture_binds_single_gpu_and_exact_icd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _maps, _library, graphics = _prepare_graphics_capture(tmp_path, monkeypatch)
    icd = tmp_path / "nvidia_icd.json"
    _write_canonical_icd(icd)
    monkeypatch.setattr(
        runtime, "PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH", str(icd)
    )
    monkeypatch.setenv("VK_DRIVER_FILES", str(icd))

    def fake_run(argv, **kwargs):
        assert argv == list(runtime.PI05_DROID_JOINTPOS_NVIDIA_SMI_QUERY)
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 30,
        }
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=f"{GPU_UUID}, NVIDIA L40S, 580.105.08\n",
            stderr="",
        )

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    captured = runtime.capture_jointpos_execution_environment()
    expected = _execution_environment()
    expected["graphics_runtime"] = graphics
    assert captured == expected
    assert runtime.validate_jointpos_execution_environment(captured) == captured


def test_execution_environment_capture_rejects_ambient_or_filesystem_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("VK_DRIVER_FILES", raising=False)
    with pytest.raises(ValueError, match="VK_DRIVER_FILES"):
        runtime.capture_jointpos_execution_environment()

    target = tmp_path / "target.json"
    target.write_bytes(b"not the canonical ICD")
    symlink = tmp_path / "nvidia_icd.json"
    symlink.symlink_to(target)
    monkeypatch.setattr(
        runtime, "PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH", str(symlink)
    )
    monkeypatch.setenv("VK_DRIVER_FILES", str(symlink))
    with pytest.raises(ValueError, match="regular non-symlink"):
        runtime.capture_jointpos_execution_environment()

    monkeypatch.setattr(
        runtime, "PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH", str(target)
    )
    monkeypatch.setenv("VK_DRIVER_FILES", str(target))
    with pytest.raises(ValueError, match="Vulkan ICD bytes mismatch"):
        runtime.capture_jointpos_execution_environment()


def test_execution_environment_capture_rejects_multiple_or_wrong_gpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    canonical_icd = tmp_path / "nvidia_icd.json"
    _write_canonical_icd(canonical_icd)
    monkeypatch.setattr(
        runtime,
        "PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH",
        str(canonical_icd),
    )
    monkeypatch.setenv("VK_DRIVER_FILES", str(canonical_icd))

    def two_gpu_run(argv, **_kwargs):
        row = f"{GPU_UUID}, NVIDIA L40S, 580.105.08\n"
        return subprocess.CompletedProcess(argv, 0, stdout=row + row, stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", two_gpu_run)
    with pytest.raises(ValueError, match="exactly one NVIDIA GPU"):
        runtime.capture_jointpos_execution_environment()

    def wrong_gpu_run(argv, **_kwargs):
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=f"{GPU_UUID}, NVIDIA A100-SXM4-80GB, 580.105.08\n",
            stderr="",
        )

    monkeypatch.setattr(runtime.subprocess, "run", wrong_gpu_run)
    with pytest.raises(ValueError, match="NVIDIA identity mismatch"):
        runtime.capture_jointpos_execution_environment()


def test_execution_environment_pure_validator_is_closed():
    expected = _execution_environment()
    assert runtime.validate_jointpos_execution_environment(expected) == expected
    for mutate, message in (
        (
            lambda value: value["nvidia_smi"].update({"driver_version": "580.95.05"}),
            "NVIDIA identity",
        ),
        (
            lambda value: value["nvidia_smi"].update({"uuid": "GPU-not-a-real-uuid"}),
            "NVIDIA identity",
        ),
        (
            lambda value: value["vulkan"].update(
                {"vk_driver_files": "/tmp/nvidia_icd.json"}
            ),
            "Vulkan runtime identity",
        ),
        (
            lambda value: value["vulkan"]["icd"].update({"sha256": "0" * 64}),
            "Vulkan runtime identity",
        ),
        (
            lambda value: value["vulkan"].update({"unexpected": True}),
            "Vulkan runtime schema",
        ),
        (
            lambda value: value["graphics_runtime"]["libraries"][0].update(
                {"sha256": "0" * 64}
            ),
            "graphics-library identity",
        ),
        (
            lambda value: value["graphics_runtime"]["libraries"][0].update(
                {"path": "/tmp/libvulkan.so.1"}
            ),
            "graphics-library identity",
        ),
        (
            lambda value: value["graphics_runtime"]["libraries"][0].update(
                {"elf_gnu_build_id": "0" * 40}
            ),
            "graphics-library identity",
        ),
        (
            lambda value: value["graphics_runtime"]["environment"].update(
                {"LD_PRELOAD": "/tmp/injected.so"}
            ),
            "override environment",
        ),
        (
            lambda value: value["graphics_runtime"]["environment"].update(
                {"NVIDIA_VISIBLE_DEVICES": "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"}
            ),
            "override environment",
        ),
        (
            lambda value: value["graphics_runtime"].update(
                {"graphics_runtime_sha256": "0" * 64}
            ),
            "graphics-runtime SHA-256",
        ),
    ):
        tampered = copy.deepcopy(expected)
        mutate(tampered)
        with pytest.raises(ValueError, match=message):
            runtime.validate_jointpos_execution_environment(tampered)


def test_production_graphics_table_and_canonical_digest_are_closed():
    value = _graphics_runtime()
    assert len(runtime.PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES) == 15
    assert len(value["libraries"]) == 15
    assert value["graphics_runtime_sha256"] == (
        "f3ee6c8027f0cfea3c0f4875c2d3c0aba4c8cf41f8bde040a0bf236b81133a84"
    )
    assert (
        value["graphics_runtime_sha256"]
        == runtime.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_SHA256
    )


def test_mapped_graphics_capture_binds_maps_file_hash_and_build_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _maps, _library, expected = _prepare_graphics_capture(tmp_path, monkeypatch)
    captured = runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)
    assert captured == expected
    assert (
        runtime._validate_graphics_runtime(captured, expected_gpu_uuid=GPU_UUID)
        == captured
    )


@pytest.mark.parametrize("mode", ["missing", "extra", "deleted"])
def test_mapped_graphics_capture_rejects_missing_extra_or_deleted_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
):
    maps, library, _expected = _prepare_graphics_capture(tmp_path, monkeypatch)
    if mode == "missing":
        maps.write_text("", encoding="utf-8")
        message = "set mismatch"
    elif mode == "extra":
        extra = tmp_path / "libcuda.so.580.105.08"
        shutil.copyfile(ELF_FIXTURE, extra)
        current = maps.read_text(encoding="utf-8")
        extra_maps = tmp_path / "extra.maps"
        _write_maps(extra_maps, extra)
        maps.write_text(
            current + extra_maps.read_text(encoding="utf-8"), encoding="utf-8"
        )
        message = "set mismatch"
    else:
        _write_maps(maps, library, deleted=True)
        message = "has been deleted"
    with pytest.raises(ValueError, match=message):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


@pytest.mark.parametrize("field", ["device", "inode"])
def test_mapped_graphics_capture_rejects_maps_stat_identity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
):
    maps, library, _expected = _prepare_graphics_capture(tmp_path, monkeypatch)
    if field == "device":
        _write_maps(maps, library, device="ffff:ffff")
    else:
        _write_maps(maps, library, inode=library.stat().st_ino + 1)
    with pytest.raises(ValueError, match="differs from process maps"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


def test_mapped_graphics_capture_rejects_symlink_backing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    maps, library, _expected = _prepare_graphics_capture(tmp_path, monkeypatch)
    target = tmp_path / "python-elf"
    library.rename(target)
    library.symlink_to(target)
    metadata = target.stat()
    descriptor = os.open(target, os.O_RDONLY)
    try:
        build_id = runtime._elf_gnu_build_id(descriptor, metadata.st_size)
    finally:
        os.close(descriptor)
    monkeypatch.setattr(
        runtime,
        "PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES",
        (
            (
                str(library),
                metadata.st_size,
                hashlib.sha256(target.read_bytes()).hexdigest(),
                build_id,
            ),
        ),
    )
    _write_maps(maps, library)
    with pytest.raises(ValueError, match="regular non-symlink"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


@pytest.mark.parametrize(
    "name",
    ("LD_LIBRARY_PATH", *runtime.PI05_DROID_JOINTPOS_GRAPHICS_FORBIDDEN_ENVIRONMENT),
)
def test_mapped_graphics_capture_rejects_loader_override_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    monkeypatch.setenv(name, "/tmp/injected")
    with pytest.raises(ValueError, match="override environment"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


def test_mapped_graphics_capture_rejects_expected_hash_or_build_id_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _maps, library, expected = _prepare_graphics_capture(tmp_path, monkeypatch)
    original = runtime.PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES[0]
    for index, replacement in ((2, "0" * 64), (3, "0" * 40)):
        changed = list(original)
        changed[index] = replacement
        monkeypatch.setattr(
            runtime,
            "PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES",
            (tuple(changed),),
        )
        with pytest.raises(ValueError, match="graphics-library identity"):
            runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)
    assert library.is_file()
    assert expected["libraries"][0]["path"] == str(library)


def test_runtime_validator_is_closed_over_every_live_contract_surface():
    report = _runtime_report()
    assert runtime.validate_jointpos_runtime_report(report) == report
    digest_tampered = copy.deepcopy(report)
    digest_tampered["execution_environment"]["nvidia_smi"]["uuid"] = (
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime.validate_jointpos_runtime_report(digest_tampered)
    for mutate, message in (
        (
            lambda value: value["observation"]["terms"]["gripper_pos"].update(
                {"clip": None}
            ),
            "observation",
        ),
        (
            lambda value: value["configured_actuators"]["panda_shoulder"].update(
                {"stiffness": 399.0}
            ),
            "configured actuator",
        ),
        (
            lambda value: value["direct_physx_actuator_and_limits"].update(
                {"joint_stiffness": [[399.0] * 7]}
            ),
            "direct PhysX",
        ),
        (
            lambda value: value["gripper"].update({"threshold": "open_positive"}),
            "gripper",
        ),
    ):
        tampered = copy.deepcopy(report)
        mutate(tampered)
        tampered = _rehash(tampered)
        with pytest.raises(ValueError, match=message):
            runtime.validate_jointpos_runtime_report(tampered)


def test_runtime_artifact_publication_is_immutable_and_no_replace(tmp_path: Path):
    destination = tmp_path / "runtime.json"
    report = _runtime_report()
    artifact = runtime.publish_jointpos_runtime(destination, report)
    assert artifact["mode"] == "0444"
    assert artifact["execution_environment"] == report["execution_environment"]
    assert destination.stat().st_nlink == 1
    assert (
        json.loads(destination.read_text())["runtime_sha256"]
        == report["runtime_sha256"]
    )
    with pytest.raises(FileExistsError):
        runtime.publish_jointpos_runtime(destination, report)


def test_timeout_configuration_keeps_internal_step_beyond_outer_horizon():
    cfg = type("Cfg", (), {"episode_length_s": 0.0})()
    seconds = runtime.configure_jointpos_timeout(cfg)
    assert seconds == 451 / 15
    assert cfg.episode_length_s == seconds


def test_jointpos_observation_cfg_preserves_historical_clip_noise_and_eef_terms():
    source = (ROOT / "src/polaris/environments/pi05_droid_jointpos_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "noise=noise.GaussianNoiseCfg(std=0.05)" in source
    assert "clip=(0.0, 1.0)" in source
    assert "eef_pos = ObsTerm(func=eef_pos)" in source
    assert "eef_quat = ObsTerm(func=eef_quat)" in source
    assert "self.enable_corruption = False" in source
    assert "self.concatenate_terms = False" in source
