import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import polaris.pi05_droid_jointpos_runtime as runtime
from polaris.pi05_droid_jointpos_image_contract import (
    IMAGE_PROFILE,
    source_contract,
    static_image_contract,
)


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
        "proc_environ_path": runtime.PI05_DROID_JOINTPOS_GRAPHICS_PROC_ENVIRON_PATH,
        "initial_environment": {
            "LD_LIBRARY_PATH": None,
            "NVIDIA_VISIBLE_DEVICES": GPU_UUID,
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "VK_DRIVER_FILES": (runtime.PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH),
            **{
                name: None
                for name in runtime.PI05_DROID_JOINTPOS_GRAPHICS_FORBIDDEN_ENVIRONMENT
            },
            **{
                name: None
                for name in (
                    runtime.PI05_DROID_JOINTPOS_GRAPHICS_VULKAN_SDK_REQUIRED_ABSENT_ENVIRONMENT
                )
            },
        },
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
            **{
                name: None
                for name in (
                    runtime.PI05_DROID_JOINTPOS_GRAPHICS_VULKAN_SDK_REQUIRED_ABSENT_ENVIRONMENT
                )
            },
        },
        "cv2_loader": {
            "profile": runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_PROFILE,
            "module": {
                **dict(runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_MODULE_IDENTITY),
                "native_maps_device": "0:1",
                "native_maps_inode": 1000,
            },
            "loader_search_safety": {
                "profile": (
                    runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_SEARCH_SAFETY_PROFILE
                ),
                "working_directory": "/immutable/polaris",
                "working_directory_binding": ("equals_runtime_module_repository_root"),
                "working_directory_read_only": True,
                "normalized_cv2_binary_path": "/.venv/lib/python3.11/lib64",
                "normalized_cv2_binary_path_exists": False,
                "working_directory_library_candidates": [],
            },
            "files": [
                {"path": path, "size": size, "sha256": sha256}
                for path, size, sha256 in (
                    runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_FILES
                )
            ],
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
    cv2_dir = tmp_path / "cv2"
    cv2_dir.mkdir()
    cv2_loader_identities = []
    for name, payload in (
        ("__init__.py", b"cv2-init"),
        ("config-3.py", b"cv2-config-3"),
        ("config.py", b"cv2-config"),
        ("cv2.abi3.so", b"cv2-native"),
        ("load_config_py3.py", b"cv2-load-config"),
        ("version.py", b"cv2-version"),
    ):
        path = cv2_dir / name
        path.write_bytes(payload)
        cv2_loader_identities.append(
            (str(path), len(payload), hashlib.sha256(payload).hexdigest())
        )
    monkeypatch.setattr(
        runtime,
        "PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_FILES",
        tuple(cv2_loader_identities),
    )
    module_identity = dict(runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_MODULE_IDENTITY)
    module_identity.update(
        {
            "python_module_path": str(cv2_dir / "__init__.py"),
            "python_module_spec_origin": str(cv2_dir / "__init__.py"),
            "native_module_path": str(cv2_dir / "cv2.abi3.so"),
            "native_module_spec_origin": str(cv2_dir / "cv2.abi3.so"),
            "load_config_module_path": str(cv2_dir / "load_config_py3.py"),
            "version_module_path": str(cv2_dir / "version.py"),
            "higher_priority_config_path": str(cv2_dir / "config-3.11.py"),
            "selected_config_path": str(cv2_dir / "config-3.py"),
        }
    )
    monkeypatch.setattr(
        runtime,
        "PI05_DROID_JOINTPOS_GRAPHICS_CV2_MODULE_IDENTITY",
        tuple(module_identity.items()),
    )
    native_metadata = (cv2_dir / "cv2.abi3.so").stat()
    module_capture = {
        **module_identity,
        "native_maps_device": (
            f"{os.major(native_metadata.st_dev):x}:{os.minor(native_metadata.st_dev):x}"
        ),
        "native_maps_inode": native_metadata.st_ino,
    }
    monkeypatch.setattr(
        runtime,
        "_capture_graphics_cv2_module_identity",
        lambda: copy.deepcopy(module_capture),
    )
    loader_search_safety = {
        "profile": (
            runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_SEARCH_SAFETY_PROFILE
        ),
        "working_directory": str(tmp_path),
        "working_directory_binding": "equals_runtime_module_repository_root",
        "working_directory_read_only": True,
        "normalized_cv2_binary_path": "/.venv/lib/python3.11/lib64",
        "normalized_cv2_binary_path_exists": False,
        "working_directory_library_candidates": [],
    }
    monkeypatch.setattr(
        runtime,
        "_capture_graphics_cv2_loader_search_safety",
        lambda: copy.deepcopy(loader_search_safety),
    )
    maps = tmp_path / "maps"
    _write_maps(maps, library)
    monkeypatch.setattr(
        runtime, "PI05_DROID_JOINTPOS_GRAPHICS_PROC_MAPS_PATH", str(maps)
    )
    environ = tmp_path / "environ"
    environ.write_bytes(
        (
            f"NVIDIA_VISIBLE_DEVICES={GPU_UUID}\0"
            "NVIDIA_DRIVER_CAPABILITIES=all\0"
            f"VK_DRIVER_FILES={runtime.PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH}\0"
        ).encode("utf-8")
    )
    monkeypatch.setattr(
        runtime, "PI05_DROID_JOINTPOS_GRAPHICS_PROC_ENVIRON_PATH", str(environ)
    )
    monkeypatch.setenv(
        "LD_LIBRARY_PATH",
        runtime.PI05_DROID_JOINTPOS_GRAPHICS_EXPECTED_LD_LIBRARY_PATH,
    )
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", GPU_UUID)
    monkeypatch.setenv("NVIDIA_DRIVER_CAPABILITIES", "all")
    for name in runtime.PI05_DROID_JOINTPOS_GRAPHICS_FORBIDDEN_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    for (
        name
    ) in runtime.PI05_DROID_JOINTPOS_GRAPHICS_VULKAN_SDK_REQUIRED_ABSENT_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    device = f"{os.major(metadata.st_dev):x}:{os.minor(metadata.st_dev):x}"
    value = {
        "profile": runtime.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE,
        "proc_maps_path": str(maps),
        "proc_environ_path": str(environ),
        "initial_environment": runtime._initial_graphics_environment(),
        "environment": runtime._graphics_environment(),
        "cv2_loader": {
            "profile": runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_PROFILE,
            "module": module_capture,
            "loader_search_safety": loader_search_safety,
            "files": [
                {"path": path, "size": size, "sha256": sha256}
                for path, size, sha256 in cv2_loader_identities
            ],
        },
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
        "image_contract": {
            "static": static_image_contract(),
            "sources": source_contract(),
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
            "term_order": ["arm_joint_pos", "gripper_pos"],
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
            },
        },
        "configured_actuators": {
            "capture_phase": runtime.PI05_DROID_JOINTPOS_POST_MUTATION_CFG_PHASE,
            "groups": {
                "panda_shoulder": {
                    "joint_names_expr": ["panda_joint[1-4]"],
                    "stiffness": 400.0,
                    "damping": 80.0,
                    "effort_limit": 87.0,
                    "effort_limit_sim": 87.0,
                    "velocity_limit": None,
                    "velocity_limit_sim": None,
                },
                "panda_forearm": {
                    "joint_names_expr": ["panda_joint[5-7]"],
                    "stiffness": 400.0,
                    "damping": 80.0,
                    "effort_limit": 12.0,
                    "effort_limit_sim": 12.0,
                    "velocity_limit": None,
                    "velocity_limit_sim": None,
                },
                "gripper": {
                    "joint_names_expr": ["finger_joint"],
                    "stiffness": None,
                    "damping": None,
                    "effort_limit": 200.0,
                    "effort_limit_sim": 200.0,
                    "velocity_limit": None,
                    "velocity_limit_sim": None,
                },
            },
        },
        "resolved_actuator_and_limits": {
            "provenance": runtime.PI05_DROID_JOINTPOS_EFFECTIVE_LIMIT_PROVENANCE,
            "default_joint_vel_limits_attribute_present": False,
            "groups": {
                "panda_shoulder": {
                    "joint_names": list(runtime.PANDA_ARM_JOINT_NAMES[:4]),
                    "joint_indices": [0, 1, 2, 3],
                    "stiffness": runtime._EXPECTED_STIFFNESS[:, :4].tolist(),
                    "damping": runtime._EXPECTED_DAMPING[:, :4].tolist(),
                    "effort_limit": runtime._EXPECTED_EFFORT[:, :4].tolist(),
                    "effort_limit_sim": runtime._EXPECTED_EFFORT[:, :4].tolist(),
                    "velocity_limit": runtime._EXPECTED_SIM_VELOCITY[:, :4].tolist(),
                    "velocity_limit_sim": runtime._EXPECTED_SIM_VELOCITY[
                        :, :4
                    ].tolist(),
                },
                "panda_forearm": {
                    "joint_names": list(runtime.PANDA_ARM_JOINT_NAMES[4:]),
                    "joint_indices": [4, 5, 6],
                    "stiffness": runtime._EXPECTED_STIFFNESS[:, 4:].tolist(),
                    "damping": runtime._EXPECTED_DAMPING[:, 4:].tolist(),
                    "effort_limit": runtime._EXPECTED_EFFORT[:, 4:].tolist(),
                    "effort_limit_sim": runtime._EXPECTED_EFFORT[:, 4:].tolist(),
                    "velocity_limit": runtime._EXPECTED_SIM_VELOCITY[:, 4:].tolist(),
                    "velocity_limit_sim": runtime._EXPECTED_SIM_VELOCITY[
                        :, 4:
                    ].tolist(),
                },
                "gripper": {
                    "joint_names": ["finger_joint"],
                    "joint_indices": [7],
                    "stiffness": runtime._EXPECTED_FINGER_STIFFNESS.tolist(),
                    "damping": runtime._EXPECTED_FINGER_DAMPING.tolist(),
                    "effort_limit": runtime._EXPECTED_FINGER_EFFORT.tolist(),
                    "effort_limit_sim": runtime._EXPECTED_FINGER_EFFORT.tolist(),
                    "velocity_limit": (runtime._EXPECTED_FINGER_SIM_VELOCITY.tolist()),
                    "velocity_limit_sim": (
                        runtime._EXPECTED_FINGER_SIM_VELOCITY.tolist()
                    ),
                },
            },
        },
        "live_actuator_and_limits": {
            "joint_stiffness": runtime._EXPECTED_STIFFNESS.tolist(),
            "joint_damping": runtime._EXPECTED_DAMPING.tolist(),
            "joint_effort_limits": runtime._EXPECTED_EFFORT.tolist(),
            "joint_velocity_limits": runtime._EXPECTED_SIM_VELOCITY.tolist(),
            "hard_joint_position_limits": runtime._EXPECTED_HARD_LIMITS.tolist(),
            "soft_joint_position_limits": runtime._EXPECTED_SOFT_LIMITS.tolist(),
        },
        "direct_physx_actuator_and_limits": {
            "joint_stiffness": runtime._EXPECTED_STIFFNESS.tolist(),
            "joint_damping": runtime._EXPECTED_DAMPING.tolist(),
            "joint_effort_limits": runtime._EXPECTED_EFFORT.tolist(),
            "joint_velocity_limits": runtime._EXPECTED_SIM_VELOCITY.tolist(),
            "hard_joint_position_limits": runtime._EXPECTED_HARD_LIMITS.tolist(),
        },
        "cameras": {
            name: {
                "shape": [720, 1280, 3],
                "dtype": "uint8",
                "image_profile": IMAGE_PROFILE,
                "final_hash_validated": True,
            }
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
            "live_actuator_and_limits": {
                "joint_stiffness": runtime._EXPECTED_FINGER_STIFFNESS.tolist(),
                "joint_damping": runtime._EXPECTED_FINGER_DAMPING.tolist(),
                "joint_effort_limits": runtime._EXPECTED_FINGER_EFFORT.tolist(),
                "joint_velocity_limits": (
                    runtime._EXPECTED_FINGER_SIM_VELOCITY.tolist()
                ),
                "hard_joint_position_limits": (
                    runtime._EXPECTED_FINGER_HARD_LIMITS.tolist()
                ),
                "soft_joint_position_limits": (
                    runtime._EXPECTED_FINGER_SOFT_LIMITS.tolist()
                ),
            },
            "direct_physx_actuator_and_limits": {
                "joint_stiffness": runtime._EXPECTED_FINGER_STIFFNESS.tolist(),
                "joint_damping": runtime._EXPECTED_FINGER_DAMPING.tolist(),
                "joint_effort_limits": runtime._EXPECTED_FINGER_EFFORT.tolist(),
                "joint_velocity_limits": (
                    runtime._EXPECTED_FINGER_SIM_VELOCITY.tolist()
                ),
                "hard_joint_position_limits": (
                    runtime._EXPECTED_FINGER_HARD_LIMITS.tolist()
                ),
            },
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


def test_effective_limit_constants_bind_usd_physx_float32_values():
    assert runtime.PI05_DROID_JOINTPOS_EFFECTIVE_LIMIT_PROVENANCE == (
        "legacy_velocity_limit_ignored_backcompat_usd_physx_max_joint_velocity"
    )
    np.testing.assert_array_equal(
        runtime._EXPECTED_SIM_VELOCITY,
        np.full((1, 7), 10.0, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        runtime._EXPECTED_FINGER_SIM_VELOCITY,
        np.asarray([[8.726646423339844]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        runtime._EXPECTED_FINGER_HARD_LIMITS,
        np.asarray([[[0.0, 0.7853981852531433]]], dtype=np.float32),
    )


def test_resolved_actuator_capture_requires_exact_names_indices_and_float32():
    actuator = types.SimpleNamespace(
        joint_names=["finger_joint"],
        joint_indices=np.asarray([7], dtype=np.int64),
        stiffness=runtime._EXPECTED_FINGER_STIFFNESS.copy(),
        damping=runtime._EXPECTED_FINGER_DAMPING.copy(),
        effort_limit=runtime._EXPECTED_FINGER_EFFORT.copy(),
        effort_limit_sim=runtime._EXPECTED_FINGER_EFFORT.copy(),
        velocity_limit=runtime._EXPECTED_FINGER_SIM_VELOCITY.copy(),
        velocity_limit_sim=runtime._EXPECTED_FINGER_SIM_VELOCITY.copy(),
    )
    report = runtime._resolved_actuator_report(
        actuator,
        expected_names=("finger_joint",),
        expected_indices=(7,),
        expected_stiffness=runtime._EXPECTED_FINGER_STIFFNESS,
        expected_damping=runtime._EXPECTED_FINGER_DAMPING,
        expected_effort=runtime._EXPECTED_FINGER_EFFORT,
        expected_velocity=runtime._EXPECTED_FINGER_SIM_VELOCITY,
        field="test finger",
    )
    assert report["joint_indices"] == [7]
    assert report["velocity_limit"] == [[8.726646423339844]]

    actuator.velocity_limit = actuator.velocity_limit.astype(np.float64)
    with pytest.raises(ValueError, match="velocity limit mismatch"):
        runtime._resolved_actuator_report(
            actuator,
            expected_names=("finger_joint",),
            expected_indices=(7,),
            expected_stiffness=runtime._EXPECTED_FINGER_STIFFNESS,
            expected_damping=runtime._EXPECTED_FINGER_DAMPING,
            expected_effort=runtime._EXPECTED_FINGER_EFFORT,
            expected_velocity=runtime._EXPECTED_FINGER_SIM_VELOCITY,
            field="test finger",
        )


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
    environ_path = Path(runtime.PI05_DROID_JOINTPOS_GRAPHICS_PROC_ENVIRON_PATH)
    environ_path.write_bytes(
        environ_path.read_bytes().replace(
            b"VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json\0",
            f"VK_DRIVER_FILES={icd}\0".encode(),
        )
    )
    graphics["initial_environment"]["VK_DRIVER_FILES"] = str(icd)
    graphics["graphics_runtime_sha256"] = runtime._graphics_runtime_sha256(graphics)
    monkeypatch.setattr(
        runtime,
        "PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_SHA256",
        graphics["graphics_runtime_sha256"],
    )

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
            lambda value: value["graphics_runtime"]["cv2_loader"]["files"][0].update(
                {"sha256": "0" * 64}
            ),
            "cv2-loader identity",
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
    assert runtime.PI05_DROID_JOINTPOS_RUNTIME_SCHEMA_VERSION == 5
    assert runtime.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE == (
        "l401_pyxis_nvidia_580_105_08_mapped_graphics_v5"
    )
    assert runtime.PI05_DROID_JOINTPOS_GRAPHICS_PROC_ENVIRON_PATH == (
        "/proc/self/environ"
    )
    assert len(runtime.PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES) == 15
    assert len(value["libraries"]) == 15
    assert runtime.PI05_DROID_JOINTPOS_GRAPHICS_EXPECTED_LD_LIBRARY_PATH == (
        "/.venv/lib/python3.11/site-packages/cv2/../../lib64:"
    )
    assert runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_PROFILE == (
        "opencv_python_headless_4_11_0_86_linux_loader_v1"
    )
    assert runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_SEARCH_SAFETY_PROFILE == (
        "cv2_empty_loader_element_readonly_workdir_v1"
    )
    assert dict(runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_MODULE_IDENTITY) == {
        "python_module_path": "/.venv/lib/python3.11/site-packages/cv2/__init__.py",
        "python_module_spec_origin": (
            "/.venv/lib/python3.11/site-packages/cv2/__init__.py"
        ),
        "native_module_path": ("/.venv/lib/python3.11/site-packages/cv2/cv2.abi3.so"),
        "native_module_spec_origin": (
            "/.venv/lib/python3.11/site-packages/cv2/cv2.abi3.so"
        ),
        "load_config_module_path": (
            "/.venv/lib/python3.11/site-packages/cv2/load_config_py3.py"
        ),
        "version_module_path": ("/.venv/lib/python3.11/site-packages/cv2/version.py"),
        "opencv_version": "4.11.0",
        "package_version": "4.11.0.86",
        "ci_build": True,
        "headless": True,
        "contrib": False,
        "rolling": False,
        "python_executable": "/.venv/bin/python",
        "python_implementation": "cpython",
        "python_cache_tag": "cpython-311",
        "python_major": 3,
        "python_minor": 11,
        "higher_priority_config_path": (
            "/.venv/lib/python3.11/site-packages/cv2/config-3.11.py"
        ),
        "higher_priority_config_exists": False,
        "selected_config_path": ("/.venv/lib/python3.11/site-packages/cv2/config-3.py"),
    }
    assert runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_FILES == (
        (
            "/.venv/lib/python3.11/site-packages/cv2/__init__.py",
            6_612,
            "936bd94c5a5debf0212fc751af79d3a163652f3e850259df2159db6aa3ed8ad8",
        ),
        (
            "/.venv/lib/python3.11/site-packages/cv2/config-3.py",
            724,
            "9a7aadf724b822001f5e963b01fd4e375d45e2cbbe328d2ca7b42a440c083a1c",
        ),
        (
            "/.venv/lib/python3.11/site-packages/cv2/config.py",
            111,
            "974e2d4096ee1a9a9a341df1bf33e16973683bf1ac733006de27ff2f23bc584d",
        ),
        (
            "/.venv/lib/python3.11/site-packages/cv2/cv2.abi3.so",
            66_106_617,
            "68fee49d266a95e730c1cb17d913a39a93ab5c50bee1581600f453026f9c7b8d",
        ),
        (
            "/.venv/lib/python3.11/site-packages/cv2/load_config_py3.py",
            262,
            "03dc1f11374a667c9b7db10dd5276d640b0c2d5b2e7866b4ccef772732bd3852",
        ),
        (
            "/.venv/lib/python3.11/site-packages/cv2/version.py",
            92,
            "3b07492169e6079940f716162368c51b6dbee45be36c9c46fb6f9e56b0449739",
        ),
    )
    assert runtime.PI05_DROID_JOINTPOS_GRAPHICS_FORBIDDEN_ENVIRONMENT[-2:] == (
        "QT_QPA_PLATFORM_PLUGIN_PATH",
        "QT_QPA_FONTDIR",
    )
    assert (
        runtime.PI05_DROID_JOINTPOS_GRAPHICS_VULKAN_SDK_REQUIRED_ABSENT_ENVIRONMENT
        == (
            "VK_SDK_PATH",
            "VULKAN_SDK",
            "VK_LAYER_PATH",
            "VK_INSTANCE_LAYERS",
            "VULKAN_HEADERS_INSTALL_DIR",
        )
    )
    assert {
        name: value["environment"][name]
        for name in (
            runtime.PI05_DROID_JOINTPOS_GRAPHICS_VULKAN_SDK_REQUIRED_ABSENT_ENVIRONMENT
        )
    } == {
        name: None
        for name in (
            runtime.PI05_DROID_JOINTPOS_GRAPHICS_VULKAN_SDK_REQUIRED_ABSENT_ENVIRONMENT
        )
    }
    assert (
        "/usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.580.105.08",
        39_422_584,
        "1ed129c4f703547fe5f8961dada7d53cb2981404fabdbfa9b9b3e3d83a04f6ac",
        "6257a5b3887eab41edd54343ea3623c373ab8e8e",
    ) in runtime.PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES
    assert value["graphics_runtime_sha256"] == (
        "f08522c21472c7a42a726952f0bdce5119a31744cd35cb4cb81e37a24faa2eb7"
    )
    assert (
        value["graphics_runtime_sha256"]
        == runtime.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_SHA256
    )


def test_graphics_digest_excludes_only_bound_per_process_identities():
    value = _graphics_runtime()
    baseline = runtime._graphics_runtime_sha256(value)
    value["environment"]["NVIDIA_VISIBLE_DEVICES"] = (
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    )
    value["initial_environment"]["NVIDIA_VISIBLE_DEVICES"] = (
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    )
    value["cv2_loader"]["module"]["native_maps_device"] = "ff:1"
    value["cv2_loader"]["module"]["native_maps_inode"] += 1
    value["cv2_loader"]["loader_search_safety"]["working_directory"] = (
        "/another/immutable/polaris"
    )
    for item in value["libraries"]:
        item["maps_device"] = "ff:1"
        item["maps_inode"] += 1
    assert runtime._graphics_runtime_sha256(value) == baseline


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


def test_cv2_native_maps_identity_normalizes_device_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    native = tmp_path / "cv2.abi3.so"
    maps = tmp_path / "maps"
    maps.write_text(
        f"7f000000-7f001000 r-xp 00000000 09:00 123 {native}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runtime, "PI05_DROID_JOINTPOS_GRAPHICS_PROC_MAPS_PATH", str(maps)
    )
    assert runtime._parse_proc_maps_identity_for_path(str(native)) == ("9:0", 123)


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


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        "/.venv/lib/python3.11/site-packages/cv2/../../lib64",
        "/.venv/lib/python3.11/site-packages/cv2/../../lib64:/tmp/injected",
        "/tmp/injected:/.venv/lib/python3.11/site-packages/cv2/../../lib64:",
    ],
)
def test_mapped_graphics_capture_requires_exact_cv2_loader_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    if value is None:
        monkeypatch.delenv("LD_LIBRARY_PATH")
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", value)
    with pytest.raises(ValueError, match='"LD_LIBRARY_PATH"'):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


def test_mapped_graphics_capture_rejects_cv2_loader_file_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    path = Path(runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_FILES[0][0])
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="cv2-loader identity"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


def test_mapped_graphics_capture_rejects_cv2_loader_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    path = Path(runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_FILES[0][0])
    target = tmp_path / "cv2-loader-target.py"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="regular and non-symlink"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


def test_mapped_graphics_capture_rejects_missing_cv2_loader_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    path = Path(runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_FILES[0][0])
    path.unlink()
    with pytest.raises(ValueError, match="Cannot inspect graphics cv2-loader file"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


@pytest.mark.parametrize(
    "field,replacement",
    [("path", "/tmp/cv2.py"), ("size", 1), ("sha256", "0" * 64)],
)
def test_mapped_graphics_validator_rejects_cv2_loader_identity_drift(
    field: str, replacement: int | str
):
    value = _graphics_runtime()
    value["cv2_loader"]["files"][0][field] = replacement
    with pytest.raises(ValueError, match="cv2-loader identity"):
        runtime._validate_graphics_runtime(value, expected_gpu_uuid=GPU_UUID)


def test_mapped_graphics_validator_rejects_cv2_loader_order_drift():
    value = _graphics_runtime()
    value["cv2_loader"]["files"].reverse()
    with pytest.raises(ValueError, match="cv2-loader identity"):
        runtime._validate_graphics_runtime(value, expected_gpu_uuid=GPU_UUID)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("python_module_path", "/tmp/cv2/__init__.py"),
        ("native_module_path", "/tmp/cv2.abi3.so"),
        ("opencv_version", "4.10.0"),
        ("package_version", "4.10.0.84"),
        ("python_cache_tag", "cpython-312"),
        ("higher_priority_config_exists", True),
    ],
)
def test_mapped_graphics_validator_rejects_loaded_cv2_module_drift(
    field: str, replacement: str | bool
):
    value = _graphics_runtime()
    value["cv2_loader"]["module"][field] = replacement
    with pytest.raises(ValueError, match="cv2-loader identity"):
        runtime._validate_graphics_runtime(value, expected_gpu_uuid=GPU_UUID)


def test_live_cv2_capture_rejects_wrong_loaded_module_origin(
    monkeypatch: pytest.MonkeyPatch,
):
    cv2 = types.ModuleType("cv2")
    cv2.__file__ = "/tmp/cv2/__init__.py"
    cv2.__spec__ = types.SimpleNamespace(origin=cv2.__file__)
    cv2.__version__ = "4.11.0"
    native = types.ModuleType("cv2")
    native.__file__ = "/tmp/cv2/cv2.abi3.so"
    native.__spec__ = types.SimpleNamespace(origin=native.__file__)
    version = types.ModuleType("cv2.version")
    version.__file__ = "/tmp/cv2/version.py"
    version.opencv_version = "4.11.0.86"
    version.ci_build = True
    version.headless = True
    version.contrib = False
    version.rolling = False
    load_config = types.ModuleType("cv2.load_config_py3")
    load_config.__file__ = "/tmp/cv2/load_config_py3.py"
    cv2._native = native
    cv2.version = version
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "cv2.load_config_py3", load_config)
    with pytest.raises(ValueError, match="Live cv2 module identity"):
        runtime._capture_graphics_cv2_module_identity()


def test_mapped_graphics_validator_rejects_cv2_loader_profile_or_extra_key():
    value = _graphics_runtime()
    value["cv2_loader"]["profile"] = "unbound"
    with pytest.raises(ValueError, match="cv2-loader identity"):
        runtime._validate_graphics_runtime(value, expected_gpu_uuid=GPU_UUID)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("working_directory_read_only", False),
        ("normalized_cv2_binary_path_exists", True),
        ("normalized_cv2_binary_path", "/tmp/lib64"),
        ("working_directory_library_candidates", ["libcuda.so"]),
    ],
)
def test_mapped_graphics_validator_rejects_cv2_loader_search_safety_drift(
    field: str, replacement: bool | str | list[str]
):
    value = _graphics_runtime()
    value["cv2_loader"]["loader_search_safety"][field] = replacement
    with pytest.raises(ValueError, match="cv2-loader search safety"):
        runtime._validate_graphics_runtime(value, expected_gpu_uuid=GPU_UUID)
    value = _graphics_runtime()
    value["cv2_loader"]["unexpected"] = True
    with pytest.raises(ValueError, match="cv2-loader schema"):
        runtime._validate_graphics_runtime(value, expected_gpu_uuid=GPU_UUID)
    value = _graphics_runtime()
    value["cv2_loader"]["module"]["unexpected"] = True
    with pytest.raises(ValueError, match="cv2-loader identity"):
        runtime._validate_graphics_runtime(value, expected_gpu_uuid=GPU_UUID)


@pytest.mark.parametrize(
    "name,value",
    [
        ("LD_LIBRARY_PATH", ""),
        ("LD_LIBRARY_PATH", "/tmp/injected"),
        ("QT_QPA_PLATFORM_PLUGIN_PATH", "/tmp/plugins"),
        ("QT_QPA_FONTDIR", "/tmp/fonts"),
        ("VK_LAYER_PATH", ""),
    ],
)
def test_mapped_graphics_capture_rejects_nonabsent_initial_loader_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    environ_path = Path(runtime.PI05_DROID_JOINTPOS_GRAPHICS_PROC_ENVIRON_PATH)
    payload = environ_path.read_bytes() + f"{name}={value}\0".encode()
    environ_path.write_bytes(payload)
    with pytest.raises(ValueError, match="initial graphics environment mismatch"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


def test_initial_environment_parser_rejects_duplicate_or_truncated_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    environ_path = tmp_path / "environ"
    monkeypatch.setattr(
        runtime, "PI05_DROID_JOINTPOS_GRAPHICS_PROC_ENVIRON_PATH", str(environ_path)
    )
    environ_path.write_bytes(b"A=1\0A=2\0")
    with pytest.raises(ValueError, match="duplicate keys"):
        runtime._read_proc_initial_environment()
    environ_path.write_bytes(b"A=1")
    with pytest.raises(ValueError, match="truncated"):
        runtime._read_proc_initial_environment()


def test_graphics_capture_rejects_live_environment_drift_during_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    first = runtime._graphics_environment()
    second = copy.deepcopy(first)
    second["QT_QPA_FONTDIR"] = "/tmp/drift"
    values = iter((first, second))
    monkeypatch.setattr(runtime, "_graphics_environment", lambda: next(values))
    with pytest.raises(ValueError, match="environment changed during capture"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


def test_graphics_capture_rejects_initial_environment_drift_during_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    first = runtime._initial_graphics_environment()
    second = copy.deepcopy(first)
    second["LD_LIBRARY_PATH"] = ""
    values = iter((first, second))
    monkeypatch.setattr(runtime, "_initial_graphics_environment", lambda: next(values))
    with pytest.raises(ValueError, match="initial graphics environment changed"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


def test_graphics_capture_rejects_maps_drift_during_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    first = runtime._parse_graphics_proc_maps()
    second = {path: (device, inode + 1) for path, (device, inode) in first.items()}
    values = iter((first, second))
    monkeypatch.setattr(runtime, "_parse_graphics_proc_maps", lambda: next(values))
    with pytest.raises(ValueError, match="graphics maps changed during capture"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


def test_graphics_capture_rejects_cv2_module_drift_during_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    first = runtime._capture_graphics_cv2_module_identity()
    second = copy.deepcopy(first)
    second["native_maps_inode"] += 1
    values = iter((first, second))
    monkeypatch.setattr(
        runtime, "_capture_graphics_cv2_module_identity", lambda: next(values)
    )
    with pytest.raises(ValueError, match="cv2 module identity changed"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


def test_graphics_capture_rejects_cv2_loader_search_safety_drift_during_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    first = runtime._capture_graphics_cv2_loader_search_safety()
    second = copy.deepcopy(first)
    second["working_directory_read_only"] = False
    values = iter((first, second))
    monkeypatch.setattr(
        runtime, "_capture_graphics_cv2_loader_search_safety", lambda: next(values)
    )
    with pytest.raises(ValueError, match="loader search safety changed"):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


@pytest.mark.parametrize(
    "name",
    runtime.PI05_DROID_JOINTPOS_GRAPHICS_VULKAN_SDK_REQUIRED_ABSENT_ENVIRONMENT,
)
@pytest.mark.parametrize("value", ["", "/tmp/injected"])
def test_mapped_graphics_capture_requires_vulkan_sdk_environment_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: str
):
    _prepare_graphics_capture(tmp_path, monkeypatch)
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=rf'"{name}"'):
        runtime._capture_graphics_runtime(expected_gpu_uuid=GPU_UUID)


@pytest.mark.parametrize(
    "name",
    runtime.PI05_DROID_JOINTPOS_GRAPHICS_VULKAN_SDK_REQUIRED_ABSENT_ENVIRONMENT,
)
def test_mapped_graphics_validator_rejects_missing_absent_environment_key(name: str):
    value = _graphics_runtime()
    value["environment"].pop(name)
    with pytest.raises(ValueError, match=rf'"{name}".*"actual_present":false'):
        runtime._validate_graphics_runtime(value, expected_gpu_uuid=GPU_UUID)


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
            lambda value: value["configured_actuators"]["groups"][
                "panda_shoulder"
            ].update({"stiffness": 399.0}),
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


@pytest.mark.parametrize(
    ("group", "field"),
    [
        (group, field)
        for group in ("panda_shoulder", "panda_forearm", "gripper")
        for field in (
            "joint_names_expr",
            "stiffness",
            "damping",
            "effort_limit",
            "effort_limit_sim",
            "velocity_limit",
            "velocity_limit_sim",
        )
    ],
)
def test_runtime_validator_rejects_each_post_mutation_cfg_field(group, field):
    report = _runtime_report()
    report["configured_actuators"]["groups"][group].pop(field)
    with pytest.raises(ValueError, match="configured actuator"):
        runtime.validate_jointpos_runtime_report(_rehash(report))


@pytest.mark.parametrize(
    ("group", "field"),
    [
        (group, field)
        for group in ("panda_shoulder", "panda_forearm", "gripper")
        for field in (
            "joint_names",
            "joint_indices",
            "stiffness",
            "damping",
            "effort_limit",
            "effort_limit_sim",
            "velocity_limit",
            "velocity_limit_sim",
        )
    ],
)
def test_runtime_validator_rejects_each_resolved_actuator_field(group, field):
    report = _runtime_report()
    report["resolved_actuator_and_limits"]["groups"][group].pop(field)
    with pytest.raises(ValueError, match="resolved actuator"):
        runtime.validate_jointpos_runtime_report(_rehash(report))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("provenance", "unbound"),
        ("default_joint_vel_limits_attribute_present", True),
    ],
)
def test_runtime_validator_rejects_effective_limit_provenance_or_api_drift(
    field, replacement
):
    report = _runtime_report()
    report["resolved_actuator_and_limits"][field] = replacement
    with pytest.raises(ValueError, match="resolved actuator"):
        runtime.validate_jointpos_runtime_report(_rehash(report))


def test_runtime_validator_rejects_post_mutation_capture_phase_drift():
    report = _runtime_report()
    report["configured_actuators"]["capture_phase"] = "pre_constructor"
    with pytest.raises(ValueError, match="configured actuator"):
        runtime.validate_jointpos_runtime_report(_rehash(report))


@pytest.mark.parametrize(
    ("surface", "field"),
    [
        (surface, field)
        for surface, fields in (
            (
                "live_actuator_and_limits",
                (
                    "joint_stiffness",
                    "joint_damping",
                    "joint_effort_limits",
                    "joint_velocity_limits",
                    "hard_joint_position_limits",
                    "soft_joint_position_limits",
                ),
            ),
            (
                "direct_physx_actuator_and_limits",
                (
                    "joint_stiffness",
                    "joint_damping",
                    "joint_effort_limits",
                    "joint_velocity_limits",
                    "hard_joint_position_limits",
                ),
            ),
        )
        for field in fields
    ],
)
def test_runtime_validator_rejects_each_arm_live_or_physx_field(surface, field):
    report = _runtime_report()
    report[surface].pop(field)
    with pytest.raises(ValueError, match="live|direct PhysX"):
        runtime.validate_jointpos_runtime_report(_rehash(report))


@pytest.mark.parametrize(
    ("surface", "field"),
    [
        (surface, field)
        for surface, fields in (
            (
                "live_actuator_and_limits",
                (
                    "joint_stiffness",
                    "joint_damping",
                    "joint_effort_limits",
                    "joint_velocity_limits",
                    "hard_joint_position_limits",
                    "soft_joint_position_limits",
                ),
            ),
            (
                "direct_physx_actuator_and_limits",
                (
                    "joint_stiffness",
                    "joint_damping",
                    "joint_effort_limits",
                    "joint_velocity_limits",
                    "hard_joint_position_limits",
                ),
            ),
        )
        for field in fields
    ],
)
def test_runtime_validator_rejects_each_finger_live_or_physx_field(surface, field):
    report = _runtime_report()
    report["gripper"][surface].pop(field)
    with pytest.raises(ValueError, match="gripper"):
        runtime.validate_jointpos_runtime_report(_rehash(report))


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


def test_jointpos_observation_cfg_preserves_clip_noise_and_official_state_terms():
    source = (ROOT / "src/polaris/environments/pi05_droid_jointpos_cfg.py").read_text(
        encoding="utf-8"
    )
    assert "noise=noise.GaussianNoiseCfg(std=0.05)" in source
    assert "clip=(0.0, 1.0)" in source
    assert "eef_pos = ObsTerm" not in source
    assert "eef_quat = ObsTerm" not in source
    assert "self.enable_corruption = False" in source
    assert "self.concatenate_terms = False" in source
