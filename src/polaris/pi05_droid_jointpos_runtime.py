"""Portable live-runtime evidence for native absolute joint-position evaluation.

This module deliberately has no Isaac Lab imports.  The audited action term lives in
``polaris.environments.pi05_droid_jointpos_cfg`` and calls the pure recorder below;
the remaining helpers inspect the live environment through its public/runtime
surfaces.  None of the checks alter, clip, or reject a policy target.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
from typing import Any

import numpy as np

from polaris.pi05_droid_jointpos_serving_contract import (
    PI05_DROID_JOINTPOS_NVIDIA_DRIVER_VERSION,
    PI05_DROID_JOINTPOS_NVIDIA_GPU_NAME,
    PI05_DROID_JOINTPOS_NVIDIA_SMI_QUERY,
    PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH,
    PI05_DROID_JOINTPOS_VULKAN_ICD_SHA256,
    PI05_DROID_JOINTPOS_VULKAN_ICD_SIZE,
)


PANDA_ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
PI05_DROID_JOINTPOS_PROFILE = "openpi_pi05_droid_native_joint_position_v1"
PI05_DROID_JOINTPOS_RUNTIME_SCHEMA_VERSION = 3
PI05_DROID_JOINTPOS_TRACE_SCHEMA_VERSION = 4
PI05_DROID_JOINTPOS_RUNTIME_MARKER = "POLARIS_PI05_DROID_JOINTPOS_RUNTIME="
PI05_DROID_JOINTPOS_OUTER_STEPS = 450
PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS = 451
PI05_DROID_JOINTPOS_DECIMATION = 8
PI05_DROID_JOINTPOS_PHYSICS_HZ = 120
PI05_DROID_JOINTPOS_POLICY_HZ = 15
PI05_DROID_JOINTPOS_SENSOR_NAMES = ("external_cam", "wrist_cam")
PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE = (720, 1280, 3)
PI05_DROID_JOINTPOS_BOUNDARY_PROFILE = "outer450_internal451_no_autoreset"
PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE = (
    "l401_pyxis_nvidia_580_105_08_mapped_graphics_v1"
)
PI05_DROID_JOINTPOS_GRAPHICS_PROC_MAPS_PATH = "/proc/self/maps"
PI05_DROID_JOINTPOS_GRAPHICS_EXPECTED_LD_LIBRARY_PATH: str | None = None
PI05_DROID_JOINTPOS_GRAPHICS_FORBIDDEN_ENVIRONMENT = (
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_PROFILE",
    "LD_DEBUG",
    "LD_DEBUG_OUTPUT",
    "VK_ADD_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "VK_LAYER_PATH",
    "VK_ADD_LAYER_PATH",
    "VK_INSTANCE_LAYERS",
    "VK_DEVICE_LAYERS",
    "VK_LOADER_DRIVERS_SELECT",
    "VK_LOADER_DRIVERS_DISABLE",
    "VK_LOADER_LAYERS_ENABLE",
    "VK_LOADER_LAYERS_DISABLE",
    "VK_LOADER_LAYERS_ALLOW",
    "__GLX_VENDOR_LIBRARY_NAME",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "LIBGL_DRIVERS_PATH",
)
PI05_DROID_JOINTPOS_GRAPHICS_CANDIDATE_BASENAME_PREFIXES = (
    "libvulkan.so",
    "libGLX_nvidia.so",
    "libEGL_nvidia.so",
    "libnvidia-",
    "libcuda.so",
    "libnvoptix.so",
    "libnvcuvid.so",
)
# Filled from the immutable l401/Pyxis mapped-ELF probe.  Each row is
# (absolute path, size, SHA-256, ELF GNU build-id).  Validation rejects an
# empty, partial, additional, or reordered runtime set.
PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES: tuple[
    tuple[str, int, str, str], ...
] = (
    (
        "/.venv/lib/python3.11/site-packages/isaacsim/extscache/"
        "omni.gpu_foundation-0.0.0+69cbf6ad.lx64.r.cp311/bin/deps/libvulkan.so.1",
        481_760,
        "39f1c73a2e0f94eb56924dddd8f367457dcfaf3483e3a49818fc8615467d61fc",
        "87aff1f6a98b7cf3e6285b054ded9e730dfa2312",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.580.105.08",
        1_296_640,
        "7b5c89d6df5208a659f16caf223c9b281126a7b6ef452d24adbae0b1d16b675a",
        "ba9bba6327ea3086fc9831d779241d602f23e6ad",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.580.105.08",
        1_211_968,
        "b136b6d09ff4875084ebe013327118aa6c6ef71b8d0d4d35bcc0ad62769b0232",
        "176d5b4cc2d4457d79ce869525d452ab57de6b77",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libcuda.so.580.105.08",
        96_276_264,
        "df9183549feb062f4195e6cf130e0ef372de4a59e59dbe51554ad8c3c5b167db",
        "d04acd84d257002c11b30728b30cb93b8dd99dda",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libnvcuvid.so.580.105.08",
        20_749_880,
        "1fd11e0fc9ae216cb12712ed7c4f153b32a6ca3430031dd954785c2a48bfb58b",
        "7f0fc034863a13e9f190a9a0939c8ec4edde2d08",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libnvidia-eglcore.so.580.105.08",
        35_068_336,
        "7867799dfcf2c09ddeeb99e6af3a510520ab1db4471db8ecf3f902035ef41720",
        "c818fe996e96e84d896fbc9fda4ecf9ac00476ae",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.580.105.08",
        37_045_864,
        "9fc7c0a0b1372b81c42ed0c22b1268c5e293bca0e6a66772737459b3018a7a58",
        "ad37450d0a04c9a5ae4a3c84aed544d8a897a9d4",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libnvidia-glsi.so.580.105.08",
        603_352,
        "46802ed72be2486b0fd1da997b5a98408822f9bcd8afd025ad36944197dddb69",
        "fce2d7cefdab4fab1725aca3d73667088dc49530",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libnvidia-glvkspirv.so.580.105.08",
        9_553_072,
        "672b5af2e6397b7c6e25e6522f88ad846da24587ca0de366cee1e99039a6a94f",
        "f7c5465a74e3000a4dbb1ce985690b8858c3fd6e",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libnvidia-gpucomp.so.580.105.08",
        72_261_240,
        "413df629c1ac4eb735c6a1e9146372f474d6f646bbe491d746820ec10c3181d7",
        "fcee19c145a134be8100bd9091bcb304cae7a5ce",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.105.08",
        2_283_608,
        "bee24a7507366126cdf7441c7fec4705e85c3a61c9669e08d70d7d86a8ed6f99",
        "670be7d086f6c76c668ea3cc46ba4ab43de6b3b7",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.580.105.08",
        105_513_888,
        "ddd5c845a5e02542004137f49ba65f17be0968e44f921c6181e4639a962f42d2",
        "4d24c89475327c251418da2a4a7d62f8a9bb4e87",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libnvidia-tls.so.580.105.08",
        18_632,
        "8c8d984462b870e711a2f5e83fe6b5f0f8693b9df4eabaabbc5d30e902a53748",
        "66341c9a22e978a4c0ab70dbb1192b1d95d42f4c",
    ),
    (
        "/usr/lib/x86_64-linux-gnu/libnvoptix.so.580.105.08",
        105_212_368,
        "fe382acfd994661e75d9c515f7ca31b34f72a213eb04cc515ab36f0f7729d40f",
        "f1c9a28967f20238957e98dd417dc93f9a3f8198",
    ),
)
PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_SHA256 = (
    "cd0ae19f2ea2cbdd0b8371796acad34c6d1b36d38c26aca68e8715b663c2f9f5"
)

_ACTION_TERM_CLASS = (
    "polaris.environments.pi05_droid_jointpos_cfg.AuditedDroidJointPositionAction"
)
_ACTION_CFG_CLASS = (
    "polaris.environments.pi05_droid_jointpos_cfg.AuditedDroidJointPositionActionCfg"
)
_ACTION_BASE_CLASS = "isaaclab.envs.mdp.actions.joint_actions.JointPositionAction"
_ACTION_CFG_BASE_CLASS = "isaaclab.envs.mdp.actions.actions_cfg.JointPositionActionCfg"
_GRIPPER_ACTION_CLASS = (
    "polaris.environments.droid_cfg.BinaryJointPositionZeroToOneAction"
)
_EXPECTED_STIFFNESS = np.full((1, 7), 400.0, dtype=np.float32)
_EXPECTED_DAMPING = np.full((1, 7), 80.0, dtype=np.float32)
_EXPECTED_EFFORT = np.asarray([[87.0] * 4 + [12.0] * 3], dtype=np.float32)
_EXPECTED_VELOCITY = np.asarray([[2.175] * 4 + [2.61] * 3], dtype=np.float32)
_EXPECTED_HARD_LIMITS = np.asarray(
    [
        [
            (-2.8973000049591064, 2.8973000049591064),
            (-1.7627999782562256, 1.7627999782562256),
            (-2.8973000049591064, 2.8973000049591064),
            (-3.0717999935150146, -0.0697999969124794),
            (-2.8973000049591064, 2.8973000049591064),
            (-0.017500000074505806, 3.752500057220459),
            (-2.8973000049591064, 2.8973000049591064),
        ]
    ],
    dtype=np.float32,
)
_EXPECTED_SOFT_LIMITS = np.asarray(
    [
        [
            (-2.8973000049591064, 2.8973000049591064),
            (-1.7627999782562256, 1.7627999782562256),
            (-2.8973000049591064, 2.8973000049591064),
            (-3.0717999935150146, -0.06979990005493164),
            (-2.8973000049591064, 2.8973000049591064),
            (-0.017499923706054688, 3.752500057220459),
            (-2.8973000049591064, 2.8973000049591064),
        ]
    ],
    dtype=np.float32,
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _class_path(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _numpy(value: Any, *, field: str) -> np.ndarray:
    try:
        value = value.detach().cpu().numpy()
    except AttributeError:
        value = np.asarray(value)
    result = np.asarray(value)
    if (
        not np.issubdtype(result.dtype, np.number)
        or np.issubdtype(result.dtype, np.bool_)
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{field} must be finite numeric data")
    return result


def _float32_values(value: Any, expected: np.ndarray, *, field: str) -> list[Any]:
    actual = _numpy(value, field=field)
    expected = np.asarray(expected, dtype=np.float32)
    if actual.dtype != np.float32 or not np.array_equal(actual, expected):
        raise ValueError(
            f"{field} mismatch: expected {expected.tolist()}, got "
            f"dtype={actual.dtype} values={actual.tolist()}"
        )
    return actual.tolist()


def _one_nonnegative_integer(value: Any, *, field: str) -> int:
    actual = _numpy(value, field=field)
    if actual.shape != (1,) or not np.issubdtype(actual.dtype, np.integer):
        raise ValueError(f"{field} must be one integer tensor")
    result = int(actual[0])
    if result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _capture_vulkan_icd() -> dict[str, Any]:
    """Read the mounted ICD without following a symlink or accepting drift."""

    path = PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"Cannot inspect the canonical Vulkan ICD: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("Vulkan ICD must be a regular non-symlink file")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            "Cannot open the canonical Vulkan ICD without symlinks"
        ) from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError("Vulkan ICD changed before it could be read")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    finally:
        os.close(descriptor)

    try:
        after = os.lstat(path)
    except OSError as error:
        raise ValueError("Vulkan ICD disappeared after it was read") from error
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or after.st_size != size
    ):
        raise ValueError("Vulkan ICD changed while it was read")
    sha256 = digest.hexdigest()
    if (
        size != PI05_DROID_JOINTPOS_VULKAN_ICD_SIZE
        or sha256 != PI05_DROID_JOINTPOS_VULKAN_ICD_SHA256
    ):
        raise ValueError(
            "Vulkan ICD bytes mismatch: "
            f"expected size={PI05_DROID_JOINTPOS_VULKAN_ICD_SIZE} "
            f"sha256={PI05_DROID_JOINTPOS_VULKAN_ICD_SHA256}, "
            f"got size={size} sha256={sha256}"
        )
    return {"path": path, "size": size, "sha256": sha256}


def _validate_nvidia_smi_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "query",
        "uuid",
        "name",
        "driver_version",
    }:
        raise ValueError("simulator NVIDIA identity schema mismatch")
    uuid = value["uuid"]
    if (
        value["query"] != list(PI05_DROID_JOINTPOS_NVIDIA_SMI_QUERY)
        or not isinstance(uuid, str)
        or re.fullmatch(
            r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
            r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}",
            uuid,
        )
        is None
        or value["name"] != PI05_DROID_JOINTPOS_NVIDIA_GPU_NAME
        or value["driver_version"] != PI05_DROID_JOINTPOS_NVIDIA_DRIVER_VERSION
    ):
        raise ValueError("simulator NVIDIA identity mismatch")
    return copy.deepcopy(value)


def _capture_nvidia_smi_identity() -> dict[str, Any]:
    try:
        result = subprocess.run(
            list(PI05_DROID_JOINTPOS_NVIDIA_SMI_QUERY),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("Cannot capture the simulator NVIDIA identity") from error
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("simulator runtime requires exactly one NVIDIA GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3 or any(not field for field in fields):
        raise ValueError("Cannot parse the simulator NVIDIA identity")
    return _validate_nvidia_smi_identity(
        {
            "query": list(PI05_DROID_JOINTPOS_NVIDIA_SMI_QUERY),
            "uuid": fields[0],
            "name": fields[1],
            "driver_version": fields[2],
        }
    )


def _is_graphics_library_path(path: str) -> bool:
    basename = os.path.basename(path)
    return any(
        basename.startswith(prefix)
        for prefix in PI05_DROID_JOINTPOS_GRAPHICS_CANDIDATE_BASENAME_PREFIXES
    )


def _read_exact_at(descriptor: int, size: int, offset: int, *, field: str) -> bytes:
    try:
        value = os.pread(descriptor, size, offset)
    except OSError as error:
        raise ValueError(f"Cannot read {field}") from error
    if len(value) != size:
        raise ValueError(f"Truncated {field}")
    return value


def _elf_gnu_build_id(descriptor: int, file_size: int) -> str:
    """Return the unique GNU build-id from a little-endian ELF64 PT_NOTE."""

    if file_size < 64:
        raise ValueError("Mapped graphics library is too small to be ELF64")
    header = _read_exact_at(descriptor, 64, 0, field="ELF64 header")
    if header[:4] != b"\x7fELF" or header[4] != 2 or header[5] != 1 or header[6] != 1:
        raise ValueError("Mapped graphics library is not little-endian ELF64")
    try:
        fields = struct.unpack("<HHIQQQIHHHHHH", header[16:64])
    except struct.error as error:
        raise ValueError("Cannot parse mapped graphics ELF64 header") from error
    program_offset = fields[4]
    program_entry_size = fields[8]
    program_count = fields[9]
    if (
        program_entry_size < 56
        or program_count <= 0
        or program_count > 1024
        or program_offset < 64
        or program_offset + program_entry_size * program_count > file_size
    ):
        raise ValueError("Mapped graphics ELF64 program-header table is invalid")

    build_ids: set[str] = set()
    for index in range(program_count):
        entry = _read_exact_at(
            descriptor,
            56,
            program_offset + index * program_entry_size,
            field="ELF64 program header",
        )
        try:
            program = struct.unpack("<IIQQQQQQ", entry)
        except struct.error as error:
            raise ValueError(
                "Cannot parse mapped graphics ELF64 program header"
            ) from error
        if program[0] != 4:  # PT_NOTE
            continue
        note_offset = program[2]
        note_size = program[5]
        if (
            note_size <= 0
            or note_size > 16 * 1024 * 1024
            or note_offset + note_size > file_size
        ):
            raise ValueError("Mapped graphics ELF64 PT_NOTE range is invalid")
        notes = _read_exact_at(
            descriptor, note_size, note_offset, field="ELF64 PT_NOTE segment"
        )
        cursor = 0
        while cursor + 12 <= len(notes):
            name_size, description_size, note_type = struct.unpack(
                "<III", notes[cursor : cursor + 12]
            )
            cursor += 12
            name_end = cursor + name_size
            description_start = (name_end + 3) & ~3
            description_end = description_start + description_size
            next_note = (description_end + 3) & ~3
            if (
                name_end > len(notes)
                or description_end > len(notes)
                or next_note > len(notes)
            ):
                raise ValueError("Mapped graphics ELF64 note is truncated")
            name = notes[cursor:name_end]
            description = notes[description_start:description_end]
            if note_type == 3 and name.rstrip(b"\0") == b"GNU" and description:
                build_ids.add(description.hex())
            cursor = next_note
        if any(notes[cursor:]):
            raise ValueError("Mapped graphics ELF64 PT_NOTE has nonzero trailing bytes")
    if len(build_ids) != 1:
        raise ValueError(
            "Mapped graphics library must contain one unique ELF GNU build-id"
        )
    return next(iter(build_ids))


def _graphics_environment() -> dict[str, str | None]:
    names = (
        "LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "NVIDIA_DRIVER_CAPABILITIES",
        *PI05_DROID_JOINTPOS_GRAPHICS_FORBIDDEN_ENVIRONMENT,
    )
    return {name: os.environ.get(name) for name in names}


def _expected_graphics_library_records() -> list[dict[str, Any]]:
    records = [
        {
            "path": path,
            "size": size,
            "sha256": sha256,
            "elf_gnu_build_id": build_id,
        }
        for path, size, sha256, build_id in (
            PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES
        )
    ]
    if (
        not records
        or records != sorted(records, key=lambda item: item["path"])
        or len({item["path"] for item in records}) != len(records)
        or any(
            not isinstance(item["path"], str)
            or not os.path.isabs(item["path"])
            or not _is_graphics_library_path(item["path"])
            or type(item["size"]) is not int
            or item["size"] <= 0
            or not isinstance(item["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
            or not isinstance(item["elf_gnu_build_id"], str)
            or re.fullmatch(r"[0-9a-f]{16,128}", item["elf_gnu_build_id"]) is None
            for item in records
        )
    ):
        raise ValueError("Pinned mapped graphics-library table is invalid")
    return records


def _parse_graphics_proc_maps() -> dict[str, tuple[str, int]]:
    try:
        lines = open(
            PI05_DROID_JOINTPOS_GRAPHICS_PROC_MAPS_PATH,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, UnicodeError) as error:
        raise ValueError("Cannot read the simulator process maps") from error
    observed: dict[str, tuple[str, int]] = {}
    try:
        for line in lines:
            fields = line.rstrip("\n").split(maxsplit=5)
            if len(fields) != 6:
                continue
            raw_path = fields[5]
            deleted = raw_path.endswith(" (deleted)")
            candidate_path = raw_path[: -len(" (deleted)")] if deleted else raw_path
            if not _is_graphics_library_path(candidate_path):
                continue
            if deleted:
                raise ValueError("Mapped graphics library has been deleted")
            if not os.path.isabs(candidate_path):
                raise ValueError("Mapped graphics-library path must be absolute")
            device = fields[3].lower()
            if re.fullmatch(r"[0-9a-f]+:[0-9a-f]+", device) is None:
                raise ValueError("Mapped graphics-library device is invalid")
            device = ":".join(
                f"{int(component, 16):x}" for component in device.split(":")
            )
            try:
                inode = int(fields[4], 10)
            except ValueError as error:
                raise ValueError("Mapped graphics-library inode is invalid") from error
            if inode <= 0:
                raise ValueError("Mapped graphics-library inode must be positive")
            identity = (device, inode)
            previous = observed.setdefault(candidate_path, identity)
            if previous != identity:
                raise ValueError("Mapped graphics-library identity is inconsistent")
    except (OSError, UnicodeError) as error:
        raise ValueError("Cannot parse the simulator process maps") from error
    finally:
        lines.close()
    expected_paths = {item["path"] for item in _expected_graphics_library_records()}
    if set(observed) != expected_paths:
        raise ValueError(
            "Mapped graphics-library set mismatch: "
            f"missing={sorted(expected_paths - set(observed))} "
            f"extra={sorted(set(observed) - expected_paths)}"
        )
    return observed


def _capture_mapped_graphics_library(
    path: str, maps_identity: tuple[str, int]
) -> dict[str, Any]:
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ValueError(f"Cannot inspect mapped graphics library: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError("Mapped graphics library must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            "Cannot open mapped graphics library without symlinks"
        ) from error
    try:
        opened = os.fstat(descriptor)
        device = f"{os.major(opened.st_dev):x}:{os.minor(opened.st_dev):x}"
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or (device, opened.st_ino) != maps_identity
        ):
            raise ValueError("Mapped graphics library differs from process maps")
        digest = hashlib.sha256()
        size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        build_id = _elf_gnu_build_id(descriptor, opened.st_size)
    finally:
        os.close(descriptor)
    try:
        after = os.lstat(path)
    except OSError as error:
        raise ValueError(
            "Mapped graphics library disappeared while captured"
        ) from error
    if (
        stat.S_ISLNK(after.st_mode)
        or not stat.S_ISREG(after.st_mode)
        or (after.st_dev, after.st_ino, after.st_size)
        != (before.st_dev, before.st_ino, before.st_size)
        or size != opened.st_size
    ):
        raise ValueError("Mapped graphics library changed while captured")
    return {
        "path": path,
        "size": size,
        "sha256": digest.hexdigest(),
        "elf_gnu_build_id": build_id,
        "maps_device": maps_identity[0],
        "maps_inode": maps_identity[1],
    }


def _graphics_runtime_sha256(value: dict[str, Any]) -> str:
    stable_environment = {
        name: item
        for name, item in value["environment"].items()
        if name != "NVIDIA_VISIBLE_DEVICES"
    }
    stable = {
        "profile": value["profile"],
        "proc_maps_path": value["proc_maps_path"],
        "environment": stable_environment,
        "nvidia_visible_devices_binding": "equals_execution_environment.nvidia_smi.uuid",
        "libraries": [
            {
                name: item[name]
                for name in ("path", "size", "sha256", "elf_gnu_build_id")
            }
            for item in value["libraries"]
        ],
    }
    return canonical_sha256(stable)


def _validate_graphics_runtime(value: Any, *, expected_gpu_uuid: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "profile",
        "proc_maps_path",
        "environment",
        "libraries",
        "graphics_runtime_sha256",
    }:
        raise ValueError("simulator mapped graphics-runtime schema mismatch")
    if (
        value["profile"] != PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE
        or value["proc_maps_path"] != PI05_DROID_JOINTPOS_GRAPHICS_PROC_MAPS_PATH
    ):
        raise ValueError("simulator mapped graphics-runtime identity mismatch")
    expected_environment = {
        "LD_LIBRARY_PATH": PI05_DROID_JOINTPOS_GRAPHICS_EXPECTED_LD_LIBRARY_PATH,
        "NVIDIA_VISIBLE_DEVICES": expected_gpu_uuid,
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        **{name: None for name in PI05_DROID_JOINTPOS_GRAPHICS_FORBIDDEN_ENVIRONMENT},
    }
    if value["environment"] != expected_environment:
        raise ValueError("simulator graphics override environment mismatch")
    libraries = value["libraries"]
    expected = _expected_graphics_library_records()
    if (
        not isinstance(libraries, list)
        or libraries != sorted(libraries, key=lambda item: item.get("path", ""))
        or len(libraries) != len(expected)
    ):
        raise ValueError("simulator mapped graphics-library inventory mismatch")
    stable = []
    for item in libraries:
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "path",
                "size",
                "sha256",
                "elf_gnu_build_id",
                "maps_device",
                "maps_inode",
            }
            or not isinstance(item["maps_device"], str)
            or re.fullmatch(r"[0-9a-f]+:[0-9a-f]+", item["maps_device"]) is None
            or type(item["maps_inode"]) is not int
            or item["maps_inode"] <= 0
        ):
            raise ValueError("simulator mapped graphics-library schema mismatch")
        stable.append({name: item[name] for name in expected[0]})
    if stable != expected:
        raise ValueError("simulator mapped graphics-library identity mismatch")
    computed_sha256 = _graphics_runtime_sha256(value)
    if (
        value["graphics_runtime_sha256"] != computed_sha256
        or computed_sha256 != PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_SHA256
    ):
        raise ValueError("simulator mapped graphics-runtime SHA-256 mismatch")
    return copy.deepcopy(value)


def _capture_graphics_runtime(*, expected_gpu_uuid: str) -> dict[str, Any]:
    maps = _parse_graphics_proc_maps()
    report = {
        "profile": PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE,
        "proc_maps_path": PI05_DROID_JOINTPOS_GRAPHICS_PROC_MAPS_PATH,
        "environment": _graphics_environment(),
        "libraries": [
            _capture_mapped_graphics_library(path, maps[path]) for path in sorted(maps)
        ],
    }
    report["graphics_runtime_sha256"] = _graphics_runtime_sha256(report)
    return _validate_graphics_runtime(report, expected_gpu_uuid=expected_gpu_uuid)


def validate_jointpos_execution_environment(value: Any) -> dict[str, Any]:
    """Pure validation for the simulator GPU and Vulkan execution surface."""

    if not isinstance(value, dict) or set(value) != {
        "nvidia_smi",
        "vulkan",
        "graphics_runtime",
    }:
        raise ValueError("simulator execution-environment schema mismatch")
    nvidia_smi = _validate_nvidia_smi_identity(value["nvidia_smi"])
    vulkan = value["vulkan"]
    if not isinstance(vulkan, dict) or set(vulkan) != {"vk_driver_files", "icd"}:
        raise ValueError("simulator Vulkan runtime schema mismatch")
    icd = vulkan["icd"]
    expected_icd = {
        "path": PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH,
        "size": PI05_DROID_JOINTPOS_VULKAN_ICD_SIZE,
        "sha256": PI05_DROID_JOINTPOS_VULKAN_ICD_SHA256,
    }
    if (
        vulkan["vk_driver_files"] != PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH
        or icd != expected_icd
    ):
        raise ValueError("simulator Vulkan runtime identity mismatch")
    graphics_runtime = _validate_graphics_runtime(
        value["graphics_runtime"], expected_gpu_uuid=nvidia_smi["uuid"]
    )
    return {
        "nvidia_smi": nvidia_smi,
        "vulkan": copy.deepcopy(vulkan),
        "graphics_runtime": graphics_runtime,
    }


def capture_jointpos_execution_environment() -> dict[str, Any]:
    """Capture the live simulator GPU and Vulkan identities, failing closed."""

    vk_driver_files = os.environ.get("VK_DRIVER_FILES")
    if vk_driver_files != PI05_DROID_JOINTPOS_VULKAN_ICD_CONTAINER_PATH:
        raise ValueError(
            "VK_DRIVER_FILES must select only the canonical mounted NVIDIA ICD"
        )
    icd = _capture_vulkan_icd()
    nvidia_smi = _capture_nvidia_smi_identity()
    report = {
        "nvidia_smi": nvidia_smi,
        "vulkan": {
            "vk_driver_files": vk_driver_files,
            "icd": icd,
        },
        "graphics_runtime": _capture_graphics_runtime(
            expected_gpu_uuid=nvidia_smi["uuid"]
        ),
    }
    return validate_jointpos_execution_environment(report)


class JointPositionExecutionRecorder:
    """Observe upstream processing and all eight target-buffer setter calls."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._raw: np.ndarray | None = None
        self._processed: np.ndarray | None = None
        self._holds: list[np.ndarray] = []

    def begin_policy_step(self, raw: Any, processed: Any) -> None:
        if self._raw is not None:
            raise RuntimeError("prior joint-position execution report was not consumed")
        raw_array = _numpy(raw, field="raw joint-position action")
        processed_array = _numpy(processed, field="processed joint-position action")
        if (
            raw_array.shape != (1, 7)
            or processed_array.shape != (1, 7)
            or raw_array.dtype != np.float32
            or processed_array.dtype != np.float32
        ):
            raise ValueError("joint-position action buffers must be float32 [1,7]")
        self._raw = raw_array.copy()
        self._processed = processed_array.copy()
        self._holds = []

    def record_apply_target(self, target: Any) -> None:
        if self._raw is None or self._processed is None:
            raise RuntimeError("joint-position setter ran without process_actions")
        target_array = _numpy(target, field="joint-position apply target")
        if target_array.shape != (1, 7) or target_array.dtype != np.float32:
            raise ValueError("joint-position apply target must be float32 [1,7]")
        if len(self._holds) >= PI05_DROID_JOINTPOS_DECIMATION:
            raise ValueError("more than eight joint-position setter calls in one step")
        self._holds.append(target_array.copy())

    def finish_policy_step(self, post_step_target: Any) -> dict[str, Any]:
        if self._raw is None or self._processed is None:
            raise RuntimeError("no pending joint-position execution report")
        post = _numpy(post_step_target, field="post-step articulation target")
        if post.shape != (1, 7) or post.dtype != np.float32:
            raise ValueError("post-step articulation target must be float32 [1,7]")
        if len(self._holds) != PI05_DROID_JOINTPOS_DECIMATION:
            raise ValueError(
                "joint-position action was not held for exactly eight physics substeps"
            )
        if not np.array_equal(self._raw, self._processed):
            raise ValueError("scale-one offset-zero processing changed the raw action")
        if any(not np.array_equal(hold, self._processed) for hold in self._holds):
            raise ValueError(
                "articulation target buffer drifted during the eight holds"
            )
        if not np.array_equal(post, self._processed):
            raise ValueError(
                "post-step articulation target differs from processed target"
            )
        result = {
            "schema_version": 1,
            "processing": "upstream_joint_position_action_scale1_offset0_no_clip",
            "raw_action_buffer": self._raw[0].tolist(),
            "processed_action_buffer": self._processed[0].tolist(),
            "apply_target_holds": [hold[0].tolist() for hold in self._holds],
            "apply_target_hold_count": len(self._holds),
            "post_step_articulation_target": post[0].tolist(),
        }
        self.reset()
        return result


def configure_jointpos_timeout(env_cfg: Any) -> float:
    """Leave one internal step beyond the explicit 450-step scoring horizon."""

    seconds = (
        PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS / PI05_DROID_JOINTPOS_POLICY_HZ
    )
    env_cfg.episode_length_s = seconds
    return seconds


def capture_jointpos_environment_state(env: Any) -> dict[str, Any]:
    root = getattr(env, "unwrapped", env)
    if root.max_episode_length != PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS:
        raise ValueError("live joint-position max_episode_length must be 451")
    if type(root._sim_step_counter) is not int or root._sim_step_counter < 0:
        raise ValueError("live simulation step counter is invalid")
    if type(root.common_step_counter) is not int or root.common_step_counter < 0:
        raise ValueError("live common step counter is invalid")
    sensors = getattr(root.scene, "sensors", None)
    if not isinstance(sensors, dict):
        raise ValueError("live environment has no closed camera mapping")
    counters = {}
    for name in PI05_DROID_JOINTPOS_SENSOR_NAMES:
        if name not in sensors:
            raise ValueError(f"missing live camera sensor {name}")
        counters[name] = _one_nonnegative_integer(
            sensors[name].frame, field=f"{name} camera frame counter"
        )
    return {
        "boundary_profile": PI05_DROID_JOINTPOS_BOUNDARY_PROFILE,
        "live_max_episode_length": root.max_episode_length,
        "episode_length": _one_nonnegative_integer(
            root.episode_length_buf, field="episode length buffer"
        ),
        "sim_step_counter": root._sim_step_counter,
        "common_step_counter": root.common_step_counter,
        "sensor_frame_counters": counters,
    }


def _validate_native_observation(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict) or not isinstance(obs.get("splat"), dict):
        raise ValueError("joint-position observation has no splat camera mapping")
    report = {}
    for name in PI05_DROID_JOINTPOS_SENSOR_NAMES:
        image = np.asarray(obs["splat"].get(name))
        if image.shape != PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE:
            raise ValueError(f"{name} native image shape mismatch: {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"{name} native image dtype must be uint8")
        report[name] = {"shape": list(image.shape), "dtype": str(image.dtype)}
    return report


def capture_jointpos_runtime(env: Any, obs: Any) -> dict[str, Any]:
    """Fail closed over the live native position-control execution surface."""

    root = getattr(env, "unwrapped", env)
    execution_environment = capture_jointpos_execution_environment()
    if root.cfg.sim.dt != 1.0 / PI05_DROID_JOINTPOS_PHYSICS_HZ:
        raise ValueError("native joint-position physics dt must be 1/120")
    if root.cfg.decimation != PI05_DROID_JOINTPOS_DECIMATION:
        raise ValueError("native joint-position decimation must be 8")
    capture_jointpos_environment_state(root)
    if list(root.action_manager._terms) != ["arm", "finger_joint"]:
        raise ValueError("joint-position action order must be arm then finger_joint")
    arm = root.action_manager._terms["arm"]
    finger = root.action_manager._terms["finger_joint"]
    if _class_path(arm) != _ACTION_TERM_CLASS:
        raise ValueError(f"joint-position action class mismatch: {_class_path(arm)}")
    if _class_path(arm.cfg) != _ACTION_CFG_CLASS:
        raise ValueError(
            f"joint-position action config mismatch: {_class_path(arm.cfg)}"
        )
    if not any(
        f"{base.__module__}.{base.__qualname__}" == _ACTION_BASE_CLASS
        for base in type(arm).__mro__
    ):
        raise ValueError("audited action does not preserve JointPositionAction")
    if not any(
        f"{base.__module__}.{base.__qualname__}" == _ACTION_CFG_BASE_CLASS
        for base in type(arm.cfg).__mro__
    ):
        raise ValueError("audited config does not preserve JointPositionActionCfg")
    if (
        tuple(arm._joint_names) != PANDA_ARM_JOINT_NAMES
        or arm.cfg.preserve_order is not True
        or arm.cfg.use_default_offset is not False
        or arm.cfg.clip is not None
        or type(arm._scale) is not float
        or arm._scale != 1.0
        or type(arm._offset) is not float
        or arm._offset != 0.0
    ):
        raise ValueError("live joint-position action affine/order contract mismatch")
    if _class_path(finger) != _GRIPPER_ACTION_CLASS:
        raise ValueError("live gripper action class does not preserve closed-positive")
    if tuple(finger._joint_names) != ("finger_joint",):
        raise ValueError("live gripper action joint mismatch")
    open_command = _float32_values(
        finger._open_command,
        np.zeros((1,), dtype=np.float32),
        field="gripper open command",
    )
    closed_command = _float32_values(
        finger._close_command,
        np.full((1,), np.pi / 4.0, dtype=np.float32),
        field="gripper closed command",
    )

    robot = root.scene["robot"]
    joint_ids, joint_names = robot.find_joints(
        list(PANDA_ARM_JOINT_NAMES), preserve_order=True
    )
    if tuple(joint_names) != PANDA_ARM_JOINT_NAMES or list(joint_ids) != list(range(7)):
        raise ValueError(f"live articulation joint order mismatch: {joint_names}")
    finger_ids, finger_names = robot.find_joints(["finger_joint"], preserve_order=True)
    if finger_names != ["finger_joint"] or list(finger_ids) != [7]:
        raise ValueError("live historical gripper observation index mismatch")
    expected_arrays = {
        "joint_stiffness": _EXPECTED_STIFFNESS,
        "joint_damping": _EXPECTED_DAMPING,
        "joint_effort_limits": _EXPECTED_EFFORT,
        "joint_velocity_limits": _EXPECTED_VELOCITY,
        "hard_joint_position_limits": _EXPECTED_HARD_LIMITS,
        "soft_joint_position_limits": _EXPECTED_SOFT_LIMITS,
    }
    live_values = {
        "joint_stiffness": robot.data.joint_stiffness[:, joint_ids],
        "joint_damping": robot.data.joint_damping[:, joint_ids],
        "joint_effort_limits": robot.data.joint_effort_limits[:, joint_ids],
        "joint_velocity_limits": robot.data.joint_vel_limits[:, joint_ids],
        "hard_joint_position_limits": robot.data.joint_pos_limits[:, joint_ids],
        "soft_joint_position_limits": robot.data.soft_joint_pos_limits[:, joint_ids],
    }
    live_actuator = {
        name: _float32_values(live_values[name], expected, field=name)
        for name, expected in expected_arrays.items()
    }
    direct_values = {
        "joint_stiffness": robot.root_physx_view.get_dof_stiffnesses()[:, joint_ids],
        "joint_damping": robot.root_physx_view.get_dof_dampings()[:, joint_ids],
        "joint_effort_limits": robot.root_physx_view.get_dof_max_forces()[:, joint_ids],
        "joint_velocity_limits": robot.root_physx_view.get_dof_max_velocities()[
            :, joint_ids
        ],
        "hard_joint_position_limits": robot.root_physx_view.get_dof_limits()[
            :, joint_ids
        ],
    }
    direct_physx = {
        name: _float32_values(
            direct_values[name], expected_arrays[name], field=f"direct PhysX {name}"
        )
        for name in direct_values
    }
    configured_actuators = {}
    for name, expected in (
        (
            "panda_shoulder",
            {
                "joint_names_expr": ["panda_joint[1-4]"],
                "stiffness": 400.0,
                "damping": 80.0,
                "effort_limit": 87.0,
                "velocity_limit": 2.175,
            },
        ),
        (
            "panda_forearm",
            {
                "joint_names_expr": ["panda_joint[5-7]"],
                "stiffness": 400.0,
                "damping": 80.0,
                "effort_limit": 12.0,
                "velocity_limit": 2.61,
            },
        ),
    ):
        cfg = robot.cfg.actuators[name]
        actual = {
            "joint_names_expr": list(cfg.joint_names_expr),
            "stiffness": cfg.stiffness,
            "damping": cfg.damping,
            "effort_limit": cfg.effort_limit,
            "velocity_limit": cfg.velocity_limit,
        }
        if actual != expected:
            raise ValueError(f"configured {name} actuator mismatch: {actual}")
        configured_actuators[name] = actual

    cameras = _validate_native_observation(obs)
    for name in PI05_DROID_JOINTPOS_SENSOR_NAMES:
        image_shape = tuple(root.scene.sensors[name].image_shape)
        if image_shape != PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE[:2]:
            raise ValueError(f"live {name} camera shape mismatch: {image_shape}")

    policy_cfg = root.cfg.observations.policy
    arm_function = getattr(policy_cfg.arm_joint_pos, "func", None)
    gripper_function = getattr(policy_cfg.gripper_pos, "func", None)
    eef_pos_function = getattr(policy_cfg.eef_pos, "func", None)
    eef_quat_function = getattr(policy_cfg.eef_quat, "func", None)
    gripper_noise = policy_cfg.gripper_pos.noise
    gripper_noise_class = _class_path(gripper_noise)
    if (
        gripper_noise_class != "isaaclab.utils.noise.noise_cfg.GaussianNoiseCfg"
        or type(gripper_noise.mean) is not float
        or gripper_noise.mean != 0.0
        or type(gripper_noise.std) is not float
        or gripper_noise.std != 0.05
    ):
        raise ValueError("historical gripper noise configuration drifted")
    observation = {
        "term_order": ["arm_joint_pos", "gripper_pos", "eef_pos", "eef_quat"],
        "enable_corruption": policy_cfg.enable_corruption,
        "concatenate_terms": policy_cfg.concatenate_terms,
        "state_layout": {
            "arm_joint_indices": list(joint_ids),
            "gripper_joint_index": finger_ids[0],
            "historical_filter_order_equivalent": True,
        },
        "terms": {
            "arm_joint_pos": {
                "function": (
                    f"{getattr(arm_function, '__module__', '')}."
                    f"{getattr(arm_function, '__name__', '')}"
                ),
                "noise": None,
                "clip": None,
            },
            "gripper_pos": {
                "function": (
                    f"{getattr(gripper_function, '__module__', '')}."
                    f"{getattr(gripper_function, '__name__', '')}"
                ),
                "noise": {
                    "class": gripper_noise_class,
                    "mean": gripper_noise.mean,
                    "std": gripper_noise.std,
                    "active": False,
                },
                "clip": list(policy_cfg.gripper_pos.clip),
            },
            "eef_pos": {
                "function": (
                    f"{getattr(eef_pos_function, '__module__', '')}."
                    f"{getattr(eef_pos_function, '__name__', '')}"
                ),
                "noise": None,
                "clip": None,
            },
            "eef_quat": {
                "function": (
                    f"{getattr(eef_quat_function, '__module__', '')}."
                    f"{getattr(eef_quat_function, '__name__', '')}"
                ),
                "noise": None,
                "clip": None,
            },
        },
    }
    expected_observation = {
        "term_order": ["arm_joint_pos", "gripper_pos", "eef_pos", "eef_quat"],
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
                    "class": "isaaclab.utils.noise.noise_cfg.GaussianNoiseCfg",
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
    }
    if observation != expected_observation:
        raise ValueError(f"joint-position observation config mismatch: {observation}")

    report = {
        "schema_version": PI05_DROID_JOINTPOS_RUNTIME_SCHEMA_VERSION,
        "profile": PI05_DROID_JOINTPOS_PROFILE,
        "status": "pass",
        "execution_environment": execution_environment,
        "boundary": {
            "profile": PI05_DROID_JOINTPOS_BOUNDARY_PROFILE,
            "outer_steps": PI05_DROID_JOINTPOS_OUTER_STEPS,
            "internal_max_episode_steps": (
                PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS
            ),
            "returned_terminal_flags": "all_false",
            "terminal_rubric_source": "post_action_450_pre_autoreset_info",
        },
        "timing": {
            "physics_dt_seconds": 1.0 / PI05_DROID_JOINTPOS_PHYSICS_HZ,
            "physics_frequency_hz": PI05_DROID_JOINTPOS_PHYSICS_HZ,
            "decimation": PI05_DROID_JOINTPOS_DECIMATION,
            "policy_frequency_hz": PI05_DROID_JOINTPOS_POLICY_HZ,
        },
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
        "action": {
            "term_class": _class_path(arm),
            "cfg_class": _class_path(arm.cfg),
            "base_class": _ACTION_BASE_CLASS,
            "cfg_base_class": _ACTION_CFG_BASE_CLASS,
            "preserve_order": arm.cfg.preserve_order,
            "scale": arm._scale,
            "offset": arm._offset,
            "use_default_offset": arm.cfg.use_default_offset,
            "clip": arm.cfg.clip,
            "semantic": "absolute_joint_position_observation_only_no_guard",
            "setter_calls_per_outer_step": PI05_DROID_JOINTPOS_DECIMATION,
        },
        "observation": observation,
        "configured_actuators": configured_actuators,
        "live_actuator_and_limits": live_actuator,
        "direct_physx_actuator_and_limits": direct_physx,
        "cameras": cameras,
        "gripper": {
            "action_class": _class_path(finger),
            "joint_name": "finger_joint",
            "threshold": "closed_if_gt_0p5_else_open",
            "open_target_rad": open_command[0],
            "closed_target_rad": closed_command[0],
            "observation": "finger_joint_position_divided_by_pi_over_4_closed_positive",
        },
    }
    report["runtime_sha256"] = canonical_sha256(report)
    return validate_jointpos_runtime_report(report)


def validate_jointpos_runtime_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("joint-position runtime report must be an object")
    required = {
        "schema_version",
        "profile",
        "status",
        "execution_environment",
        "boundary",
        "timing",
        "joint_names",
        "action",
        "observation",
        "configured_actuators",
        "live_actuator_and_limits",
        "direct_physx_actuator_and_limits",
        "cameras",
        "gripper",
        "runtime_sha256",
    }
    if set(value) != required:
        raise ValueError("joint-position runtime report schema mismatch")
    payload = copy.deepcopy(value)
    digest = payload.pop("runtime_sha256")
    if digest != canonical_sha256(payload):
        raise ValueError("joint-position runtime report SHA-256 mismatch")
    expected_scalars = {
        "schema_version": PI05_DROID_JOINTPOS_RUNTIME_SCHEMA_VERSION,
        "profile": PI05_DROID_JOINTPOS_PROFILE,
        "status": "pass",
        "joint_names": list(PANDA_ARM_JOINT_NAMES),
    }
    for name, expected in expected_scalars.items():
        if value[name] != expected:
            raise ValueError(f"joint-position runtime {name} mismatch")
    validate_jointpos_execution_environment(value["execution_environment"])
    if value["boundary"] != {
        "profile": PI05_DROID_JOINTPOS_BOUNDARY_PROFILE,
        "outer_steps": PI05_DROID_JOINTPOS_OUTER_STEPS,
        "internal_max_episode_steps": PI05_DROID_JOINTPOS_INTERNAL_MAX_EPISODE_STEPS,
        "returned_terminal_flags": "all_false",
        "terminal_rubric_source": "post_action_450_pre_autoreset_info",
    }:
        raise ValueError("joint-position 450/451 boundary contract mismatch")
    if value["timing"] != {
        "physics_dt_seconds": 1.0 / PI05_DROID_JOINTPOS_PHYSICS_HZ,
        "physics_frequency_hz": PI05_DROID_JOINTPOS_PHYSICS_HZ,
        "decimation": PI05_DROID_JOINTPOS_DECIMATION,
        "policy_frequency_hz": PI05_DROID_JOINTPOS_POLICY_HZ,
    }:
        raise ValueError("joint-position runtime timing mismatch")
    if value["action"] != {
        "term_class": _ACTION_TERM_CLASS,
        "cfg_class": _ACTION_CFG_CLASS,
        "base_class": _ACTION_BASE_CLASS,
        "cfg_base_class": _ACTION_CFG_BASE_CLASS,
        "preserve_order": True,
        "scale": 1.0,
        "offset": 0.0,
        "use_default_offset": False,
        "clip": None,
        "semantic": "absolute_joint_position_observation_only_no_guard",
        "setter_calls_per_outer_step": PI05_DROID_JOINTPOS_DECIMATION,
    }:
        raise ValueError("joint-position runtime action mismatch")
    if value["observation"] != {
        "term_order": ["arm_joint_pos", "gripper_pos", "eef_pos", "eef_quat"],
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
                    "class": "isaaclab.utils.noise.noise_cfg.GaussianNoiseCfg",
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
    }:
        raise ValueError("joint-position runtime observation mismatch")
    if value["configured_actuators"] != {
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
    }:
        raise ValueError("joint-position configured actuator mismatch")
    if value["gripper"] != {
        "action_class": _GRIPPER_ACTION_CLASS,
        "joint_name": "finger_joint",
        "threshold": "closed_if_gt_0p5_else_open",
        "open_target_rad": 0.0,
        "closed_target_rad": float(np.float32(np.pi / 4.0)),
        "observation": ("finger_joint_position_divided_by_pi_over_4_closed_positive"),
    }:
        raise ValueError("joint-position gripper runtime mismatch")
    expected_live = {
        "joint_stiffness": _EXPECTED_STIFFNESS,
        "joint_damping": _EXPECTED_DAMPING,
        "joint_effort_limits": _EXPECTED_EFFORT,
        "joint_velocity_limits": _EXPECTED_VELOCITY,
        "hard_joint_position_limits": _EXPECTED_HARD_LIMITS,
        "soft_joint_position_limits": _EXPECTED_SOFT_LIMITS,
    }
    live = value["live_actuator_and_limits"]
    if not isinstance(live, dict) or set(live) != set(expected_live):
        raise ValueError("joint-position live actuator report schema mismatch")
    for name, expected in expected_live.items():
        if not np.array_equal(np.asarray(live[name], dtype=np.float32), expected):
            raise ValueError(f"joint-position live {name} mismatch")
    direct_expected = {
        name: expected
        for name, expected in expected_live.items()
        if name != "soft_joint_position_limits"
    }
    direct = value["direct_physx_actuator_and_limits"]
    if not isinstance(direct, dict) or set(direct) != set(direct_expected):
        raise ValueError("joint-position direct PhysX report schema mismatch")
    for name, expected in direct_expected.items():
        if not np.array_equal(np.asarray(direct[name], dtype=np.float32), expected):
            raise ValueError(f"joint-position direct PhysX {name} mismatch")
    cameras = value["cameras"]
    expected_camera = {
        "shape": list(PI05_DROID_JOINTPOS_NATIVE_IMAGE_SHAPE),
        "dtype": "uint8",
    }
    if cameras != {name: expected_camera for name in PI05_DROID_JOINTPOS_SENSOR_NAMES}:
        raise ValueError("joint-position native camera contract mismatch")
    return copy.deepcopy(value)


def format_jointpos_runtime(report: dict[str, Any]) -> str:
    canonical = validate_jointpos_runtime_report(report)
    return PI05_DROID_JOINTPOS_RUNTIME_MARKER + canonical_json_bytes(canonical).decode(
        "ascii"
    )


def publish_jointpos_runtime(path: Any, report: dict[str, Any]) -> dict[str, Any]:
    """Publish one immutable runtime document without a writable final state."""

    from pathlib import Path  # local keeps the portable import surface minimal

    canonical = validate_jointpos_runtime_report(report)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"joint-position runtime artifact exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"joint-position runtime temporary exists: {temporary}")
    payload = (
        json.dumps(
            canonical,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o444)
        # link(2) is the no-replace publication primitive: a competing final
        # path makes this fail instead of silently replacing evidence.
        os.link(temporary, destination, follow_symlinks=False)
        temporary.unlink()
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()
    return validate_jointpos_runtime_artifact(
        destination, expected_runtime_sha256=canonical["runtime_sha256"]
    )


def validate_jointpos_runtime_artifact(
    path: Any, *, expected_runtime_sha256: str | None = None
) -> dict[str, Any]:
    from pathlib import Path

    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("joint-position runtime artifact must be a regular file")
    metadata = artifact.stat()
    if metadata.st_mode & 0o777 != 0o444 or metadata.st_nlink != 1:
        raise ValueError("joint-position runtime artifact must be immutable mode 0444")
    raw = artifact.read_bytes()
    report = validate_jointpos_runtime_report(json.loads(raw))
    if (
        expected_runtime_sha256 is not None
        and report["runtime_sha256"] != expected_runtime_sha256
    ):
        raise ValueError("joint-position runtime artifact identity mismatch")
    return {
        "path": str(artifact.resolve()),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_sha256": report["runtime_sha256"],
        "execution_environment": copy.deepcopy(report["execution_environment"]),
        "mode": "0444",
        "nlink": metadata.st_nlink,
    }
