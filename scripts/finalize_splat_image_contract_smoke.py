#!/usr/bin/env python3
"""Independently finalize one immutable PolaRiS image-contract smoke bundle.

This host-side evidence verifier does not import the smoke producer, Isaac,
PolaRiS, NumPy, Pillow, OpenCV, or the OpenPI client.  It changes no runtime
behavior and makes no checkpoint, policy, task-success, benchmark, canary, or
promotion claim.  ``finalize`` may publish one new immutable evidence
attestation; ``verify`` only reconstructs and verifies the same bytes.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import struct
import subprocess
import sys
from typing import Any, Mapping
import zlib


class VerificationError(RuntimeError):
    """Raised when any image-evidence invariant drifts."""


# Producer identity.  The evidence commit must be a direct child and is
# supplied explicitly after it exists, avoiding a self-referential source hash.
PRODUCER_COMMIT = "9d296361bb323b2e309a3b92a204c102908c61a6"
PRODUCER_TREE = "2e868fd4a31a55c9cedfb3221e4c2bc1fbbb9310"
PRODUCER_PARENT = "42e266353df71d5906e98975165f8aa021020dad"
EVIDENCE_CHANGED_PATHS = {
    "WORKLOG.v6.md",
    "scripts/finalize_splat_image_contract_smoke.py",
    "tests/test_finalize_splat_image_contract_smoke.py",
}
PRODUCER_SOURCE_SHA256 = {
    "scripts/smoke_splat_image_contract.py": "29db9e302179bb3ca4b05c14cae92e697376bb26066e4583b752bf9f6ce8d202",
    "src/polaris/environments/manager_based_rl_splat_environment.py": "175bf8a8fcf02418827d9c1913b1b3c0b2373967788fd5a2ff6250a49c3ac592",
    "src/polaris/splat_image_contract.py": "a1bb3f3a9235cc08228c3ac1fc15f0c472c7cdfa1222425cdc40273a3bb64472",
    "src/polaris/policy/lap_eef_pose_client.py": "b853182c2cac34e5a25edc09113c49f49537f2b43dacc784104ee7167473aa5e",
    "src/polaris/policy/ego_lap_contract.py": "e3628052062d40ede220b596266617d806e2f1de10eee0873d28481952a00702",
    "src/polaris/environments/droid_cfg.py": "91f54fdf2dd487294e514534f8fd2513148596caf86b31622d680885be0f7b0d",
    "src/polaris/splat_renderer/splat_renderer.py": "b9104be8620738b6fe5bea88950e0a3b721d8d23455402071745d798c662a8aa",
}

# Stable runtime protocol identities.  Volatile job/result/log identities are
# supplied explicitly and cryptographically bound by the reviewer.
PROFILE = "polaris_foodbussing_splat_image_contract_smoke_v1"
SCOPE = "standalone_image_boundary_evidence_only"
CAMERAS = ("external_cam", "wrist_cam")
NATIVE_SHAPE = (720, 1280, 3)
PREPROCESSED_SHAPE = (224, 224, 3)
CONTENT_SHAPE = (126, 224, 3)
PAD_ROWS = 49
HUB_REVISION = "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b"
IMAGE_SHA256 = "ad566a3a0bbb300cafb4a63e0f4c0056f501e4490a136881b0b1ae2d556b324a"
SCENE_SHA256 = "82cd641e422935b394ce7ea7b6be55214c9952a2544000222921e544c409b489"
INITIAL_CONDITIONS_SHA256 = (
    "40091faee14f692350220871d30705294f21f17ae3d2974cd3c09a34d560f5de"
)
INITIAL_METADATA_SHA256 = (
    "852dd0345afb7e4d0c7526b5c327086b5132c40624ed97ff6942962126e90534"
)
SCENE_METADATA_SHA256 = (
    "accd9b67e90e510eb4ed44a789b9169df058e71ce557164f960de2d62a840e63"
)

LITERAL_USER_ROOT = Path("/lustre/fsw/portfolios/nvr/users/lzha")
CANONICAL_USER_ROOT = Path(
    "/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha"
)
CONTAINER_IMAGE_PATH = (
    LITERAL_USER_ROOT / "cache/polaris/polaris-eval-cuda13-fd00a51.sqsh"
)
DATA_ROOT = LITERAL_USER_ROOT / "data/PolaRiS-Hub"
SCENE_PATH = DATA_ROOT / "food_bussing/scene.usda"
INITIAL_CONDITIONS_PATH = DATA_ROOT / "food_bussing/initial_conditions.json"
INITIAL_METADATA_PATH = (
    DATA_ROOT
    / ".cache/huggingface/download/food_bussing/initial_conditions.json.metadata"
)
SCENE_METADATA_PATH = (
    DATA_ROOT / ".cache/huggingface/download/food_bussing/scene.usda.metadata"
)

# Closed leaf-name/kind contract.  Their exact 28 path/size/hash identities are
# loaded from raw, rehashed securely, and cross-bound to a reviewer-supplied
# canonical manifest SHA-256.
ARTIFACT_KINDS: dict[str, str] = {
    "contact_sheet": "png",
    "ego_lap_request_msgpack": "msgpack",
    **{
        f"{camera}_{suffix}": kind
        for camera in CAMERAS
        for suffix, kind in (
            ("composited_png", "png"),
            ("composited_uint8", "npy"),
            ("native_png", "png"),
            ("native_uint8", "npy"),
            ("old_abs_diff_uint8", "npy"),
            ("old_abs_diff_x4_png", "png"),
            ("old_half_down_up_png", "png"),
            ("old_half_down_up_uint8", "npy"),
            ("preprocessed_png", "png"),
            ("preprocessed_uint8", "npy"),
            ("renderer_raw_float", "npy"),
            ("robot_mask", "npy"),
            ("robot_rgb", "npy"),
        )
    },
}

TOP_LEVEL_FIELDS = {
    "schema_version",
    "profile",
    "scope",
    "stage",
    "status",
    "promotion_authorized",
    "host_finalization_required",
    "source",
    "launch_provenance",
    "result",
    "failure",
    "close_failures",
    "persistence_failures",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _typed_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _strict_json(data: bytes, field: str) -> Any:
    try:
        value = json.loads(
            data,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise VerificationError(f"{field} is not strict JSON: {error}") from error
    return value


def _serialized(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    sha256: str
    mode: str
    nlink: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class ExpectedLeaf:
    size_bytes: int
    sha256: str
    mode: int


@dataclass(frozen=True)
class RuntimeContext:
    job_id: int
    job_name: str
    result_root: Path
    raw_path: Path
    ready_path: Path
    source_identity_path: Path
    post_srun_path: Path
    attestation_path: Path
    producer_repo: Path
    evidence_repo: Path
    saved_job_script: Path
    slurm_log: Path
    sacct_snapshot: Path
    raw_spec: ExpectedLeaf
    ready_spec: ExpectedLeaf
    source_identity_spec: ExpectedLeaf
    post_srun_spec: ExpectedLeaf
    saved_job_script_spec: ExpectedLeaf
    slurm_log_spec: ExpectedLeaf
    sacct_snapshot_spec: ExpectedLeaf
    artifact_manifest_sha256: str
    srun_start_epoch_ns: int


_STABLE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
    "st_mode",
    "st_nlink",
)


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, name) == getattr(right, name) for name in _STABLE_STAT_FIELDS
    )


def _secure_read(
    path: Path,
    field: str,
    *,
    required_mode: int,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> tuple[bytes, FileRecord]:
    """Read a single-link regular file with open/fstat/lstat TOCTOU binding."""

    try:
        before = os.lstat(path)
    except OSError as error:
        raise VerificationError(f"cannot inspect {field}: {error}") from error
    _require(stat.S_ISREG(before.st_mode), f"{field} is not a regular file")
    _require(before.st_nlink == 1, f"{field} hard-link count drift")
    _require(stat.S_IMODE(before.st_mode) == required_mode, f"{field} mode drift")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise VerificationError(f"cannot securely open {field}: {error}") from error
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        _require(_same_stat(before, opened), f"{field} changed during open")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        _require(_same_stat(opened, after), f"{field} changed during read")
    finally:
        os.close(descriptor)
    try:
        linked = os.lstat(path)
    except OSError as error:
        raise VerificationError(f"cannot re-inspect {field}: {error}") from error
    _require(_same_stat(after, linked), f"{field} path changed during read")
    data = b"".join(chunks)
    digest = hashlib.sha256(data).hexdigest()
    _require(len(data) == after.st_size, f"{field} size changed during read")
    if expected_size is not None:
        _require(len(data) == expected_size, f"{field} expected size drift")
    if expected_sha256 is not None:
        _require(digest == expected_sha256, f"{field} digest drift")
    return data, FileRecord(
        path=str(path),
        size_bytes=len(data),
        sha256=digest,
        mode=f"{stat.S_IMODE(after.st_mode):04o}",
        nlink=after.st_nlink,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


def _secure_hash(
    path: Path,
    field: str,
    *,
    required_mode: int,
    expected_size: int,
    expected_sha256: str,
    must_predate_ns: int,
) -> FileRecord:
    """Streaming variant for the multi-gigabyte container image."""

    before = os.lstat(path)
    _require(stat.S_ISREG(before.st_mode), f"{field} is not a regular file")
    _require(before.st_nlink == 1, f"{field} hard-link count drift")
    _require(stat.S_IMODE(before.st_mode) == required_mode, f"{field} mode drift")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    count = 0
    try:
        opened = os.fstat(descriptor)
        _require(_same_stat(before, opened), f"{field} changed during open")
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            count += len(block)
        after = os.fstat(descriptor)
        _require(_same_stat(opened, after), f"{field} changed during hash")
    finally:
        os.close(descriptor)
    linked = os.lstat(path)
    _require(_same_stat(after, linked), f"{field} path changed during hash")
    _require(count == expected_size, f"{field} size drift")
    _require(digest.hexdigest() == expected_sha256, f"{field} digest drift")
    _require(
        max(after.st_mtime_ns, after.st_ctime_ns) <= must_predate_ns,
        f"{field} temporal provenance drift",
    )
    return FileRecord(
        path=str(path),
        size_bytes=count,
        sha256=digest.hexdigest(),
        mode=f"{stat.S_IMODE(after.st_mode):04o}",
        nlink=1,
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )


@dataclass(frozen=True)
class NpyArray:
    descr: str
    dtype: str
    shape: tuple[int, ...]
    itemsize: int
    data: bytes


_NPY_DTYPES = {
    "|u1": ("uint8", 1),
    "<u1": ("uint8", 1),
    "<i8": ("int64", 8),
    "<f4": ("float32", 4),
}


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def parse_npy(data: bytes, field: str) -> NpyArray:
    _require(data.startswith(b"\x93NUMPY"), f"{field} NPY magic drift")
    _require(len(data) >= 10, f"{field} NPY truncated")
    major, minor = data[6], data[7]
    _require((major, minor) in {(1, 0), (2, 0), (3, 0)}, f"{field} NPY version")
    if major == 1:
        header_size = struct.unpack_from("<H", data, 8)[0]
        header_start = 10
    else:
        _require(len(data) >= 12, f"{field} NPY truncated header")
        header_size = struct.unpack_from("<I", data, 8)[0]
        header_start = 12
    header_end = header_start + header_size
    _require(header_end <= len(data), f"{field} NPY header overflow")
    try:
        header = ast.literal_eval(
            data[header_start:header_end].decode("latin1").strip()
        )
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise VerificationError(f"{field} NPY header invalid: {error}") from error
    _require(
        type(header) is dict and set(header) == {"descr", "fortran_order", "shape"},
        f"{field} NPY header schema",
    )
    _require(header["fortran_order"] is False, f"{field} NPY Fortran order")
    descr = header["descr"]
    _require(descr in _NPY_DTYPES, f"{field} NPY dtype {descr!r}")
    dtype, itemsize = _NPY_DTYPES[descr]
    shape = header["shape"]
    _require(
        type(shape) is tuple
        and shape
        and all(type(item) is int and item > 0 for item in shape),
        f"{field} NPY shape",
    )
    payload = data[header_end:]
    _require(
        len(payload) == _product(shape) * itemsize,
        f"{field} NPY payload size",
    )
    return NpyArray(
        descr=descr, dtype=dtype, shape=shape, itemsize=itemsize, data=payload
    )


@dataclass(frozen=True)
class PngImage:
    width: int
    height: int
    pixels: bytes


def _paeth(left: int, above: int, upper_left: int) -> int:
    estimate = left + above - upper_left
    left_distance = abs(estimate - left)
    above_distance = abs(estimate - above)
    diagonal_distance = abs(estimate - upper_left)
    if left_distance <= above_distance and left_distance <= diagonal_distance:
        return left
    if above_distance <= diagonal_distance:
        return above
    return upper_left


def parse_png(data: bytes, field: str) -> PngImage:
    _require(data.startswith(b"\x89PNG\r\n\x1a\n"), f"{field} PNG signature")
    offset = 8
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(data):
        _require(offset + 12 <= len(data), f"{field} PNG truncated chunk")
        size = struct.unpack_from(">I", data, offset)[0]
        chunk_type = data[offset + 4 : offset + 8]
        end = offset + 12 + size
        _require(end <= len(data), f"{field} PNG chunk overflow")
        payload = data[offset + 8 : offset + 8 + size]
        expected_crc = struct.unpack_from(">I", data, offset + 8 + size)[0]
        actual_crc = zlib.crc32(chunk_type)
        actual_crc = zlib.crc32(payload, actual_crc) & 0xFFFFFFFF
        _require(actual_crc == expected_crc, f"{field} PNG CRC drift")
        chunks.append((chunk_type, payload))
        offset = end
        if chunk_type == b"IEND":
            break
    _require(offset == len(data), f"{field} PNG trailing bytes")
    _require(chunks and chunks[0][0] == b"IHDR", f"{field} PNG IHDR order")
    _require(chunks[-1] == (b"IEND", b""), f"{field} PNG IEND")
    ihdr = chunks[0][1]
    _require(len(ihdr) == 13, f"{field} PNG IHDR size")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    _require(width > 0 and height > 0, f"{field} PNG dimensions")
    _require(
        (depth, color, compression, filtering, interlace) == (8, 2, 0, 0, 0),
        f"{field} PNG must be noninterlaced RGB8",
    )
    compressed = b"".join(payload for kind, payload in chunks if kind == b"IDAT")
    _require(compressed, f"{field} PNG has no IDAT")
    try:
        filtered = zlib.decompress(compressed)
    except zlib.error as error:
        raise VerificationError(f"{field} PNG zlib failure: {error}") from error
    stride = width * 3
    _require(
        len(filtered) == height * (stride + 1),
        f"{field} PNG decompressed size",
    )
    pixels = bytearray(height * stride)
    source_offset = 0
    for row in range(height):
        filter_type = filtered[source_offset]
        _require(filter_type in {0, 1, 2, 3, 4}, f"{field} PNG filter")
        source_offset += 1
        row_start = row * stride
        for column in range(stride):
            encoded = filtered[source_offset + column]
            left = pixels[row_start + column - 3] if column >= 3 else 0
            above = pixels[row_start - stride + column] if row > 0 else 0
            upper_left = (
                pixels[row_start - stride + column - 3]
                if row > 0 and column >= 3
                else 0
            )
            if filter_type == 0:
                value = encoded
            elif filter_type == 1:
                value = encoded + left
            elif filter_type == 2:
                value = encoded + above
            elif filter_type == 3:
                value = encoded + ((left + above) // 2)
            else:
                value = encoded + _paeth(left, above, upper_left)
            pixels[row_start + column] = value & 0xFF
        source_offset += stride
    return PngImage(width=width, height=height, pixels=bytes(pixels))


class _MsgpackDecoder:
    """Restricted, non-executable MessagePack decoder for the saved request."""

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def _take(self, count: int) -> bytes:
        end = self.offset + count
        _require(end <= len(self.data), "MessagePack truncated")
        result = self.data[self.offset : end]
        self.offset = end
        return result

    def _uint(self, count: int) -> int:
        return int.from_bytes(self._take(count), "big", signed=False)

    def _int(self, count: int) -> int:
        return int.from_bytes(self._take(count), "big", signed=True)

    def _text(self, count: int) -> str:
        try:
            return self._take(count).decode("utf-8")
        except UnicodeDecodeError as error:
            raise VerificationError(f"MessagePack invalid UTF-8: {error}") from error

    def value(self) -> Any:  # noqa: C901, PLR0911, PLR0912
        tag = self._uint(1)
        if tag <= 0x7F:
            return tag
        if tag >= 0xE0:
            return tag - 256
        if 0x80 <= tag <= 0x8F:
            return self.mapping(tag & 0x0F)
        if 0x90 <= tag <= 0x9F:
            return [self.value() for _ in range(tag & 0x0F)]
        if 0xA0 <= tag <= 0xBF:
            return self._text(tag & 0x1F)
        if tag == 0xC0:
            return None
        if tag == 0xC2:
            return False
        if tag == 0xC3:
            return True
        if tag == 0xC4:
            return self._take(self._uint(1))
        if tag == 0xC5:
            return self._take(self._uint(2))
        if tag == 0xC6:
            return self._take(self._uint(4))
        if tag == 0xCA:
            return struct.unpack(">f", self._take(4))[0]
        if tag == 0xCB:
            return struct.unpack(">d", self._take(8))[0]
        if tag == 0xCC:
            return self._uint(1)
        if tag == 0xCD:
            return self._uint(2)
        if tag == 0xCE:
            return self._uint(4)
        if tag == 0xCF:
            return self._uint(8)
        if tag == 0xD0:
            return self._int(1)
        if tag == 0xD1:
            return self._int(2)
        if tag == 0xD2:
            return self._int(4)
        if tag == 0xD3:
            return self._int(8)
        if tag == 0xD9:
            return self._text(self._uint(1))
        if tag == 0xDA:
            return self._text(self._uint(2))
        if tag == 0xDB:
            return self._text(self._uint(4))
        if tag == 0xDC:
            return [self.value() for _ in range(self._uint(2))]
        if tag == 0xDD:
            return [self.value() for _ in range(self._uint(4))]
        if tag == 0xDE:
            return self.mapping(self._uint(2))
        if tag == 0xDF:
            return self.mapping(self._uint(4))
        raise VerificationError(f"unsupported MessagePack tag 0x{tag:02x}")

    def mapping(self, count: int) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        for _ in range(count):
            key = self.value()
            _require(type(key) in {str, bytes, int}, "MessagePack map key type")
            _require(key not in result, "MessagePack duplicate map key")
            result[key] = self.value()
        return result


def parse_msgpack(data: bytes) -> Any:
    decoder = _MsgpackDecoder(data)
    result = decoder.value()
    _require(decoder.offset == len(data), "MessagePack trailing bytes")
    return result


def _ndarray_from_msgpack(value: Any, field: str) -> tuple[str, tuple[int, ...], bytes]:
    _require(type(value) is dict, f"{field} MessagePack ndarray object")
    _require(
        set(value) == {b"__ndarray__", b"data", b"dtype", b"shape"},
        f"{field} MessagePack ndarray schema",
    )
    _require(value[b"__ndarray__"] is True, f"{field} ndarray marker")
    dtype = value[b"dtype"]
    if type(dtype) is bytes:
        dtype = dtype.decode("ascii")
    _require(type(dtype) is str, f"{field} ndarray dtype")
    shape = value[b"shape"]
    _require(
        type(shape) is list
        and shape
        and all(type(item) is int and item > 0 for item in shape),
        f"{field} ndarray shape",
    )
    payload = value[b"data"]
    _require(type(payload) is bytes, f"{field} ndarray bytes")
    return dtype, tuple(shape), payload


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _half_pixel_axis(input_size: int, output_size: int) -> list[tuple[int, int, float]]:
    scale = _f32(input_size / output_size)
    result = []
    for output_index in range(output_size):
        position = _f32(_f32(_f32(output_index) + _f32(0.5)) * scale - _f32(0.5))
        lower_unclipped = math.floor(position)
        fraction = _f32(position - _f32(lower_unclipped))
        lower = min(max(lower_unclipped, 0), input_size - 1)
        upper = min(max(lower_unclipped + 1, 0), input_size - 1)
        result.append((lower, upper, fraction))
    return result


def resize_with_pad_rgb8(
    pixels: bytes,
    input_height: int,
    input_width: int,
    target_height: int,
    target_width: int,
) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Independent float32 half-pixel bilinear resize and symmetric zero pad."""

    _require(
        len(pixels) == input_height * input_width * 3,
        "resize input byte count",
    )
    height_f = _f32(input_height)
    width_f = _f32(input_width)
    ratio = max(
        _f32(width_f / _f32(target_width)),
        _f32(height_f / _f32(target_height)),
    )
    resized_height = math.floor(_f32(height_f / ratio))
    resized_width = math.floor(_f32(width_f / ratio))
    _require(
        resized_height > 0
        and resized_width > 0
        and resized_height <= target_height
        and resized_width <= target_width,
        "resize geometry",
    )
    y_axis = _half_pixel_axis(input_height, resized_height)
    x_axis = _half_pixel_axis(input_width, resized_width)
    pad_top = (target_height - resized_height) // 2
    pad_left = (target_width - resized_width) // 2
    output = bytearray(target_height * target_width * 3)
    for output_y, (y0, y1, y_fraction) in enumerate(y_axis):
        for output_x, (x0, x1, x_fraction) in enumerate(x_axis):
            output_base = (
                (output_y + pad_top) * target_width + output_x + pad_left
            ) * 3
            for channel in range(3):
                top_left = _f32(pixels[(y0 * input_width + x0) * 3 + channel])
                top_right = _f32(pixels[(y0 * input_width + x1) * 3 + channel])
                bottom_left = _f32(pixels[(y1 * input_width + x0) * 3 + channel])
                bottom_right = _f32(pixels[(y1 * input_width + x1) * 3 + channel])
                top = _f32(top_left + _f32(_f32(top_right - top_left) * x_fraction))
                bottom = _f32(
                    bottom_left + _f32(_f32(bottom_right - bottom_left) * x_fraction)
                )
                value = _f32(top + _f32(_f32(bottom - top) * y_fraction))
                output[output_base + channel] = min(max(round(value), 0), 255)
    return (
        bytes(output),
        (resized_height, resized_width),
        (
            target_height - resized_height - pad_top,
            target_width - resized_width - pad_left,
        ),
    )


def rotate_rgb8_180(pixels: bytes) -> bytes:
    _require(len(pixels) % 3 == 0, "RGB byte count")
    output = bytearray(len(pixels))
    pixel_count = len(pixels) // 3
    for index in range(pixel_count):
        source = index * 3
        target = (pixel_count - index - 1) * 3
        output[target : target + 3] = pixels[source : source + 3]
    return bytes(output)


def _array_values(array: NpyArray) -> list[int] | list[float]:
    if array.dtype == "uint8":
        return list(array.data)
    if array.dtype == "int64":
        return [item[0] for item in struct.iter_unpack("<q", array.data)]
    if array.dtype == "float32":
        values = [item[0] for item in struct.iter_unpack("<f", array.data)]
        _require(all(math.isfinite(item) for item in values), "float32 array nonfinite")
        return values
    raise VerificationError(f"unsupported array dtype {array.dtype}")


def _array_summary(array: NpyArray) -> dict[str, Any]:
    values = _array_values(array)
    return {
        "shape": list(array.shape),
        "dtype": array.dtype,
        "min": min(values),
        "max": max(values),
    }


def _png_summary(image: PngImage) -> dict[str, Any]:
    return {
        "shape": [image.height, image.width, 3],
        "dtype": "uint8",
        "min": min(image.pixels),
        "max": max(image.pixels),
    }


def _artifact_filename(name: str, kind: str) -> str:
    if kind == "msgpack":
        return f"{name}.msgpack"
    return f"{name}.{kind}"


def _validate_artifact_identity(
    name: str,
    raw_identity: Any,
    file_record: FileRecord,
    kind: str,
    summary: Mapping[str, Any] | None,
    result_root: Path,
) -> None:
    _require(
        type(raw_identity) is dict
        and set(raw_identity)
        == {"path", "size_bytes", "sha256", "mode", "kind", "array"},
        f"artifact {name} identity schema",
    )
    expected = {
        "path": str(result_root / _artifact_filename(name, kind)),
        "size_bytes": file_record.size_bytes,
        "sha256": file_record.sha256,
        "mode": "0444",
        "kind": kind,
        "array": None if summary is None else dict(summary),
    }
    _require(_typed_equal(raw_identity, expected), f"artifact {name} identity drift")


def _validate_renderer_conversion(raw: NpyArray, native: NpyArray, camera: str) -> int:
    _require(raw.shape == NATIVE_SHAPE and raw.dtype == "float32", f"{camera} raw")
    _require(
        native.shape == NATIVE_SHAPE and native.dtype == "uint8",
        f"{camera} native",
    )
    differing_red_blue = 0
    native_data = native.data
    for index, (value,) in enumerate(struct.iter_unpack("<f", raw.data)):
        _require(math.isfinite(value), f"{camera} raw nonfinite")
        expected = int(min(max(value, 0.0), 1.0) * 255.0)
        _require(native_data[index] == expected, f"{camera} renderer conversion pixel")
    for offset in range(0, len(native_data), 3):
        differing_red_blue += native_data[offset] != native_data[offset + 2]
    _require(differing_red_blue > 0, f"{camera} RGB discriminator")
    return int(differing_red_blue)


def _validate_compositing(
    native: NpyArray,
    robot_rgb: NpyArray,
    mask: NpyArray,
    composited: NpyArray,
    camera: str,
) -> tuple[int, int]:
    for field, array in (
        ("native", native),
        ("robot RGB", robot_rgb),
        ("composited", composited),
    ):
        _require(
            array.dtype == "uint8" and array.shape == NATIVE_SHAPE,
            f"{camera} {field} shape/dtype",
        )
    _require(
        mask.dtype == "int64" and mask.shape == (720, 1280, 1),
        f"{camera} mask shape/dtype",
    )
    mask_values = _array_values(mask)
    _require(set(mask_values) == {0, 1}, f"{camera} mask values")
    true_count = sum(mask_values)
    false_count = len(mask_values) - true_count
    _require(true_count > 0 and false_count > 0, f"{camera} degenerate mask")
    for pixel, selected in enumerate(mask_values):
        source = robot_rgb.data if selected else native.data
        start = pixel * 3
        _require(
            composited.data[start : start + 3] == source[start : start + 3],
            f"{camera} compositing pixel",
        )
    return int(true_count), int(false_count)


def _validate_counterfactual(
    native: NpyArray,
    old: NpyArray,
    difference: NpyArray,
    camera: str,
) -> dict[str, Any]:
    for field, array in (("old", old), ("difference", difference)):
        _require(
            array.dtype == "uint8" and array.shape == NATIVE_SHAPE,
            f"{camera} counterfactual {field}",
        )
    changed_values = 0
    changed_pixels = 0
    total = 0
    maximum = 0
    for offset in range(0, len(native.data), 3):
        pixel_changed = False
        for channel in range(3):
            index = offset + channel
            expected = abs(native.data[index] - old.data[index])
            _require(difference.data[index] == expected, f"{camera} difference pixel")
            total += expected
            maximum = max(maximum, expected)
            if expected:
                changed_values += 1
                pixel_changed = True
        changed_pixels += pixel_changed
    _require(changed_values > 0 and changed_pixels > 0, f"{camera} old path unchanged")
    return {
        "changed_values": changed_values,
        "changed_pixels": changed_pixels,
        "mean_abs_diff": total / len(native.data),
        "max_abs_diff": maximum,
    }


def _validate_order_probe(contract: Any) -> dict[str, Any]:
    _require(
        type(contract) is dict
        and set(contract)
        == {
            "profile",
            "input_shape",
            "target_shape",
            "resize_then_rotate_sha256",
            "rotate_then_resize_sha256",
            "differing_values",
            "production_matches_resize_then_rotate",
        },
        "operation-order probe schema",
    )
    sentinel = bytes((index % 251) + 1 for index in range(5 * 8 * 3))
    resized, resized_shape, _ = resize_with_pad_rgb8(sentinel, 5, 8, 7, 7)
    resize_then_rotate = rotate_rgb8_180(resized)
    rotated_first = rotate_rgb8_180(sentinel)
    rotate_then_resize, rotate_first_shape, _ = resize_with_pad_rgb8(
        rotated_first, 5, 8, 7, 7
    )
    _require(resized_shape == (4, 6), "operation-order resized geometry")
    _require(rotate_first_shape == (4, 6), "operation-order alternate geometry")
    differing = sum(
        left != right
        for left, right in zip(resize_then_rotate, rotate_then_resize, strict=True)
    )
    expected = {
        "profile": "odd_5x8_to_7x7_asymmetric_padding_v1",
        "input_shape": [5, 8, 3],
        "target_shape": [7, 7, 3],
        "resize_then_rotate_sha256": hashlib.sha256(resize_then_rotate).hexdigest(),
        "rotate_then_resize_sha256": hashlib.sha256(rotate_then_resize).hexdigest(),
        "differing_values": differing,
        "production_matches_resize_then_rotate": True,
    }
    _require(differing > 0, "operation-order probe unexpectedly commutes")
    _require(_typed_equal(contract, expected), "operation-order probe drift")
    return expected


def _validate_msgpack_request(
    data: bytes,
    external_preprocessed: bytes,
    wrist_preprocessed: bytes,
) -> dict[str, Any]:
    request = parse_msgpack(data)
    _require(type(request) is dict, "request MessagePack root")
    _require(
        set(request)
        == {
            "observation",
            "prompt",
            "frame_description",
            "eef_frame",
            "dataset_name",
            "state_type",
            "has_wrist_image",
            "is_bimanual",
            "rotation_applied",
        },
        "request MessagePack schema",
    )
    _require(request["prompt"] == "Put all the foods in the bowl", "request prompt")
    _require(
        request["frame_description"]
        == "image contract smoke; no checkpoint or policy server",
        "request frame description",
    )
    _require(request["eef_frame"] == "panda_link8", "request EEF frame")
    _require(request["dataset_name"] == "droid", "request dataset")
    _require(request["state_type"] == "eef_pose", "request state type")
    _require(request["has_wrist_image"] is True, "request wrist marker")
    _require(request["is_bimanual"] is False, "request bimanual marker")
    _require(request["rotation_applied"] is True, "request rotation marker")
    observation = request["observation"]
    _require(
        type(observation) is dict
        and set(observation)
        == {
            "base_0_rgb",
            "left_wrist_0_rgb",
            "cartesian_position",
            "gripper_position",
            "state",
        },
        "request observation schema",
    )
    for key, expected in (
        ("base_0_rgb", external_preprocessed),
        ("left_wrist_0_rgb", wrist_preprocessed),
    ):
        dtype, shape, payload = _ndarray_from_msgpack(observation[key], key)
        _require(dtype in {"|u1", "<u1"}, f"request {key} dtype")
        _require(shape == PREPROCESSED_SHAPE, f"request {key} shape")
        _require(payload == expected, f"request {key} pixels")
    for key, shape in (
        ("cartesian_position", (9,)),
        ("gripper_position", (1,)),
        ("state", (10,)),
    ):
        dtype, actual_shape, payload = _ndarray_from_msgpack(observation[key], key)
        _require(dtype == "<f4", f"request {key} dtype")
        _require(actual_shape == shape, f"request {key} shape")
        _require(
            all(math.isfinite(item[0]) for item in struct.iter_unpack("<f", payload)),
            f"request {key} nonfinite",
        )
    return {
        "request_keys": sorted(request),
        "observation_keys": sorted(observation),
        "external_image_sha256": hashlib.sha256(external_preprocessed).hexdigest(),
        "wrist_image_sha256": hashlib.sha256(wrist_preprocessed).hexdigest(),
        "serialized_request_sha256": hashlib.sha256(data).hexdigest(),
    }


def _validate_camera_semantics(
    camera: str,
    arrays: Mapping[str, NpyArray],
    raw_camera: Any,
) -> dict[str, Any]:
    _require(
        type(raw_camera) is dict
        and set(raw_camera)
        == {
            "raw_renderer",
            "native_uint8",
            "robot_rgb",
            "robot_mask",
            "composited_uint8",
            "preprocessed_uint8",
            "conversion",
            "compositing",
            "preprocessing",
            "counterfactual",
        },
        f"{camera} raw schema",
    )
    summary_mapping = {
        "raw_renderer": arrays["raw"],
        "native_uint8": arrays["native"],
        "robot_rgb": arrays["robot_rgb"],
        "robot_mask": arrays["mask"],
        "composited_uint8": arrays["composited"],
        "preprocessed_uint8": arrays["preprocessed"],
    }
    for key, array in summary_mapping.items():
        _require(
            _typed_equal(raw_camera[key], _array_summary(array)),
            f"{camera} {key} summary drift",
        )
    red_blue = _validate_renderer_conversion(arrays["raw"], arrays["native"], camera)
    mask_true, mask_false = _validate_compositing(
        arrays["native"],
        arrays["robot_rgb"],
        arrays["mask"],
        arrays["composited"],
        camera,
    )
    counterfactual = _validate_counterfactual(
        arrays["native"], arrays["old"], arrays["difference"], camera
    )
    expected_preprocessed, resized_shape, remainder = resize_with_pad_rgb8(
        arrays["composited"].data, 720, 1280, 224, 224
    )
    _require(resized_shape == (126, 224), f"{camera} resized content shape")
    _require(remainder == (49, 0), f"{camera} resize remainder")
    if camera == "wrist_cam":
        expected_preprocessed = rotate_rgb8_180(expected_preprocessed)
    _require(
        arrays["preprocessed"].data == expected_preprocessed,
        f"{camera} independent preprocessing pixels",
    )
    row_size = 224 * 3
    preprocessed = arrays["preprocessed"].data
    _require(
        not any(preprocessed[: PAD_ROWS * row_size])
        and not any(preprocessed[-PAD_ROWS * row_size :]),
        f"{camera} zero-padding drift",
    )
    expected_conversion = {
        "pixel_exact": True,
        "same_shape": True,
        "finite_raw": True,
        "rgb_red_blue_differing_values": red_blue,
    }
    expected_compositing = {
        "pixel_exact": True,
        "mask_true_values": mask_true,
        "mask_false_values": mask_false,
    }
    expected_preprocessing = {
        "request_key": (
            "base_0_rgb" if camera == "external_cam" else "left_wrist_0_rgb"
        ),
        "request_pixel_exact": True,
        "top_pad_zero": True,
        "bottom_pad_zero": True,
    }
    _require(
        _typed_equal(raw_camera["conversion"], expected_conversion),
        f"{camera} conversion evidence drift",
    )
    _require(
        _typed_equal(raw_camera["compositing"], expected_compositing),
        f"{camera} compositing evidence drift",
    )
    _require(
        _typed_equal(raw_camera["preprocessing"], expected_preprocessing),
        f"{camera} preprocessing evidence drift",
    )
    _require(
        _typed_equal(raw_camera["counterfactual"], counterfactual),
        f"{camera} counterfactual evidence drift",
    )
    return {
        "renderer_raw_sha256": hashlib.sha256(arrays["raw"].data).hexdigest(),
        "native_rgb_sha256": hashlib.sha256(arrays["native"].data).hexdigest(),
        "composited_rgb_sha256": hashlib.sha256(arrays["composited"].data).hexdigest(),
        "preprocessed_rgb_sha256": hashlib.sha256(
            arrays["preprocessed"].data
        ).hexdigest(),
        "rgb_red_blue_differing_values": red_blue,
        "mask_true_values": mask_true,
        "mask_false_values": mask_false,
        "counterfactual": counterfactual,
    }


def _load_and_validate_artifacts(
    raw_artifacts: Any,
    raw_cameras: Any,
    contracts: Any,
    context: RuntimeContext,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    _require(
        type(raw_artifacts) is dict and set(raw_artifacts) == set(ARTIFACT_KINDS),
        "artifact set drift",
    )
    _require(type(raw_cameras) is dict and tuple(raw_cameras) == CAMERAS, "camera set")
    identities: dict[str, dict[str, Any]] = {}
    parsed_npy: dict[str, NpyArray] = {}
    parsed_png: dict[str, PngImage] = {}
    msgpack_bytes: bytes | None = None
    manifest: dict[str, dict[str, Any]] = {}
    for name, kind in ARTIFACT_KINDS.items():
        raw_identity = raw_artifacts[name]
        _require(
            type(raw_identity) is dict
            and type(raw_identity.get("size_bytes")) is int
            and raw_identity["size_bytes"] > 0
            and type(raw_identity.get("sha256")) is str
            and len(raw_identity["sha256"]) == 64,
            f"artifact {name} expected identity",
        )
        path = context.result_root / _artifact_filename(name, kind)
        payload, record = _secure_read(
            path,
            f"artifact {name}",
            required_mode=0o444,
            expected_size=raw_identity["size_bytes"],
            expected_sha256=raw_identity["sha256"],
        )
        if kind == "npy":
            array = parse_npy(payload, name)
            summary: Mapping[str, Any] | None = _array_summary(array)
            parsed_npy[name] = array
        elif kind == "png":
            image = parse_png(payload, name)
            summary = _png_summary(image)
            parsed_png[name] = image
        else:
            summary = None
            msgpack_bytes = payload
        _validate_artifact_identity(
            name,
            raw_identity,
            record,
            kind,
            summary,
            context.result_root,
        )
        identities[name] = record.__dict__
        manifest[name] = {
            "path": record.path,
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
            "mode": record.mode,
            "kind": kind,
        }

    manifest_sha256 = hashlib.sha256(_serialized(manifest)).hexdigest()
    _require(
        manifest_sha256 == context.artifact_manifest_sha256,
        "artifact manifest digest drift",
    )

    camera_summaries: dict[str, Any] = {}
    preprocessed: dict[str, bytes] = {}
    for camera in CAMERAS:
        arrays = {
            "raw": parsed_npy[f"{camera}_renderer_raw_float"],
            "native": parsed_npy[f"{camera}_native_uint8"],
            "robot_rgb": parsed_npy[f"{camera}_robot_rgb"],
            "mask": parsed_npy[f"{camera}_robot_mask"],
            "composited": parsed_npy[f"{camera}_composited_uint8"],
            "preprocessed": parsed_npy[f"{camera}_preprocessed_uint8"],
            "old": parsed_npy[f"{camera}_old_half_down_up_uint8"],
            "difference": parsed_npy[f"{camera}_old_abs_diff_uint8"],
        }
        camera_summaries[camera] = _validate_camera_semantics(
            camera, arrays, raw_cameras[camera]
        )
        preprocessed[camera] = arrays["preprocessed"].data
        for suffix, array_key in (
            ("native_png", "native"),
            ("composited_png", "composited"),
            ("preprocessed_png", "preprocessed"),
            ("old_half_down_up_png", "old"),
        ):
            image = parsed_png[f"{camera}_{suffix}"]
            _require(
                image.pixels == arrays[array_key].data,
                f"{camera} {suffix} lossless pixel drift",
            )
        diff_png = parsed_png[f"{camera}_old_abs_diff_x4_png"]
        expected_diff_preview = bytes(
            min(value * 4, 255) for value in arrays["difference"].data
        )
        _require(
            diff_png.pixels == expected_diff_preview,
            f"{camera} difference preview pixel drift",
        )
    contact = parsed_png["contact_sheet"]
    _require(
        (contact.height, contact.width, 3) == (720, 3200, 3),
        "contact-sheet dimensions",
    )
    _require(msgpack_bytes is not None, "MessagePack artifact missing")
    msgpack_summary = _validate_msgpack_request(
        msgpack_bytes,
        preprocessed["external_cam"],
        preprocessed["wrist_cam"],
    )
    _require(
        contracts["msgpack_roundtrip"]
        == {
            "implementation": "openpi_client.msgpack_numpy",
            "exact_arrays": True,
            "exact_image_bytes": True,
            "packed_sha256": hashlib.sha256(msgpack_bytes).hexdigest(),
        },
        "MessagePack contract drift",
    )
    return identities, {
        "artifact_manifest_sha256": manifest_sha256,
        "camera_evidence": camera_summaries,
        "msgpack": msgpack_summary,
        "contact_sheet_rgb_sha256": hashlib.sha256(contact.pixels).hexdigest(),
    }


def _validate_environment(value: Any) -> dict[str, Any]:
    _require(
        type(value) is dict
        and set(value)
        == {
            "id",
            "runtime_class",
            "scene_file",
            "initial_conditions_file",
            "initial_condition_index",
            "instruction",
            "hub_revision",
            "hub_metadata",
            "camera_sensor_keys",
            "renderer_camera_keys",
        },
        "environment schema",
    )
    expected = {
        "id": "DROID-FoodBussing",
        "runtime_class": (
            "polaris.environments.manager_based_rl_splat_environment."
            "ManagerBasedRLSplatEnv"
        ),
        "scene_file": {
            "path": str(SCENE_PATH),
            "size_bytes": 14914,
            "sha256": SCENE_SHA256,
            "mode": "0640",
        },
        "initial_conditions_file": {
            "path": str(INITIAL_CONDITIONS_PATH),
            "size_bytes": 173951,
            "sha256": INITIAL_CONDITIONS_SHA256,
            "mode": "0640",
        },
        "initial_condition_index": 0,
        "instruction": "Put all the foods in the bowl",
        "hub_revision": HUB_REVISION,
        "hub_metadata": {
            "initial_conditions": {
                "path": str(INITIAL_METADATA_PATH),
                "size_bytes": 101,
                "sha256": INITIAL_METADATA_SHA256,
                "mode": "0640",
            },
            "scene": {
                "path": str(SCENE_METADATA_PATH),
                "size_bytes": 101,
                "sha256": SCENE_METADATA_SHA256,
                "mode": "0640",
            },
        },
        "camera_sensor_keys": list(CAMERAS),
        "renderer_camera_keys": list(CAMERAS),
    }
    _require(_typed_equal(value, expected), "environment identity drift")
    return expected


def _validate_contracts(value: Any) -> dict[str, Any]:
    _require(
        type(value) is dict
        and set(value)
        == {
            "renderer_conversion",
            "robot_compositing",
            "ego_lap_preprocessing",
            "msgpack_roundtrip",
            "removed_resize_counterfactual",
        },
        "contract schema",
    )
    _require(
        value["renderer_conversion"]
        == {
            "formula": "(clip(raw_float_rgb,0,1)*255).astype(uint8)",
            "pixel_exact": True,
            "shape_preserved": True,
            "channel_order": "RGB",
            "bgr_conversion": False,
        },
        "renderer conversion declaration drift",
    )
    _require(
        value["robot_compositing"]
        == {
            "formula": "np.where(robot_mask,sim_rgb,native_splat_rgb)",
            "pixel_exact": True,
            "native_shape": list(NATIVE_SHAPE),
        },
        "compositing declaration drift",
    )
    preprocessing = value["ego_lap_preprocessing"]
    _require(
        type(preprocessing) is dict
        and set(preprocessing)
        == {
            "actual_client_class",
            "constructor_bypassed_no_network",
            "method",
            "call_events",
            "native_shape",
            "resized_content_shape",
            "preprocessed_shape",
            "padding_rows",
            "wrist_operation_order",
            "operation_order_probe",
            "pixel_exact_request_binding",
        },
        "preprocessing declaration schema",
    )
    expected_without_probe = {
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
        "native_shape": list(NATIVE_SHAPE),
        "resized_content_shape": list(CONTENT_SHAPE),
        "preprocessed_shape": list(PREPROCESSED_SHAPE),
        "padding_rows": {"top": 49, "bottom": 49},
        "wrist_operation_order": "resize_pad_then_rotate_180",
        "pixel_exact_request_binding": True,
    }
    actual_without_probe = dict(preprocessing)
    probe = actual_without_probe.pop("operation_order_probe")
    _require(
        _typed_equal(actual_without_probe, expected_without_probe),
        "preprocessing declaration drift",
    )
    order_probe = _validate_order_probe(probe)
    _require(
        value["removed_resize_counterfactual"]
        == {
            "profile": "removed_cv2_default_linear_half_down_up_v1",
            "live_path": False,
            "required_to_change_pixels": True,
        },
        "removed-resize declaration drift",
    )
    return {
        "renderer_conversion": dict(value["renderer_conversion"]),
        "robot_compositing": dict(value["robot_compositing"]),
        "ego_lap_preprocessing": {
            **expected_without_probe,
            "operation_order_probe": order_probe,
        },
        "msgpack_roundtrip": dict(value["msgpack_roundtrip"]),
        "removed_resize_counterfactual": dict(value["removed_resize_counterfactual"]),
    }


def validate_raw_semantics(
    raw: Any, context: RuntimeContext
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(type(raw) is dict and set(raw) == TOP_LEVEL_FIELDS, "raw schema")
    _require(raw["schema_version"] == 1, "raw schema version")
    _require(raw["profile"] == PROFILE, "raw profile")
    _require(
        raw["scope"]
        == "image_contract_only_no_checkpoint_policy_action_metric_or_canary",
        "raw scope",
    )
    _require(raw["stage"] == "simulation_app_close_pending", "raw stage")
    _require(raw["status"] == "passed", "raw producer status")
    _require(raw["promotion_authorized"] is False, "raw authorization drift")
    _require(raw["host_finalization_required"] is True, "raw finalizer gate")
    _require(raw["failure"] is None, "raw failure present")
    _require(raw["close_failures"] == [], "raw close failure present")
    _require(raw["persistence_failures"] == [], "raw persistence failure present")
    _require(
        raw["source"]
        == {
            "root": str(context.producer_repo),
            "commit": PRODUCER_COMMIT,
            "tree": PRODUCER_TREE,
            "tracked_clean": True,
            "expected_commit": PRODUCER_COMMIT,
            "expected_tree": PRODUCER_TREE,
        },
        "raw source identity drift",
    )
    _require(
        raw["launch_provenance"]
        == {
            "container_image": {
                "path": str(CONTAINER_IMAGE_PATH),
                "expected_sha256": IMAGE_SHA256,
                "exists": True,
                "size_bytes": 7183130624,
            },
            "saved_sbatch": {
                "path": str(context.saved_job_script),
                "expected_sha256": context.saved_job_script_spec.sha256,
                "exists": True,
                "size_bytes": context.saved_job_script_spec.size_bytes,
            },
            "expected_scene_sha256": SCENE_SHA256,
        },
        "raw launch provenance drift",
    )
    result = raw["result"]
    _require(
        type(result) is dict
        and set(result)
        == {"environment", "production_path", "contracts", "cameras", "artifacts"},
        "raw result schema",
    )
    environment = _validate_environment(result["environment"])
    _require(
        result["production_path"]
        == {
            "bound_render_splat_is_production_method": True,
            "events": [
                "ManagerBasedRLSplatEnv.render_splat.enter",
                "SplatRenderer.render",
                "ManagerBasedRLSplatEnv.render_splat.exit",
                "ManagerBasedRLSplatEnv.get_robot_from_sim",
            ],
            "renderer_render_calls": 1,
            "render_splat_calls": 1,
            "get_robot_from_sim_calls": 1,
        },
        "production call path drift",
    )
    contracts = _validate_contracts(result["contracts"])
    identities, pixel_summary = _load_and_validate_artifacts(
        result["artifacts"], result["cameras"], result["contracts"], context
    )
    return {
        "environment": environment,
        "production_path": dict(result["production_path"]),
        "contracts": contracts,
        "pixel_evidence": pixel_summary,
    }, identities


def _identity(path: Path, field: str, spec: ExpectedLeaf) -> tuple[bytes, FileRecord]:
    _require(
        spec.size_bytes > 0 and len(spec.sha256) == 64,
        f"{field} spec is not finalized",
    )
    return _secure_read(
        path,
        field,
        required_mode=spec.mode,
        expected_size=spec.size_bytes,
        expected_sha256=spec.sha256,
    )


def _validate_capture(
    context: RuntimeContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_bytes, raw_record = _identity(context.raw_path, "raw result", context.raw_spec)
    ready_bytes, ready_record = _identity(
        context.ready_path, "ready marker", context.ready_spec
    )
    source_bytes, source_record = _identity(
        context.source_identity_path,
        "source identity",
        context.source_identity_spec,
    )
    post_bytes, post_record = _identity(
        context.post_srun_path,
        "post-srun validation",
        context.post_srun_spec,
    )
    raw = _strict_json(raw_bytes, "raw result")
    semantic_summary, leaves = validate_raw_semantics(raw, context)
    ready = _strict_json(ready_bytes, "ready marker")
    _require(
        ready
        == {
            "schema_version": 1,
            "profile": PROFILE,
            "stage": "simulation_app_close_pending",
            "raw_result": {
                "path": str(context.raw_path),
                "size_bytes": len(raw_bytes),
                "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                "mode": "0444",
            },
        },
        "ready marker drift",
    )
    expected_source = "".join(
        f"{digest}  {relative}\n" for relative, digest in PRODUCER_SOURCE_SHA256.items()
    ).encode()
    _require(source_bytes == expected_source, "source identity content drift")
    post = _strict_json(post_bytes, "post-srun validation")
    _require(
        post
        == {
            "schema_version": 1,
            "profile": "polaris_image_smoke_post_srun_validation_v1",
            "status": "passed_after_zero_srun_exit",
            "slurm_job_id": str(context.job_id),
            "raw": {
                "path": str(context.raw_path),
                "size_bytes": raw_record.size_bytes,
                "sha256": raw_record.sha256,
                "mode": "0444",
            },
            "ready": {
                "path": str(context.ready_path),
                "size_bytes": ready_record.size_bytes,
                "sha256": ready_record.sha256,
                "mode": "0444",
            },
            "source_identity": {
                "path": str(context.source_identity_path),
                "size_bytes": source_record.size_bytes,
                "sha256": source_record.sha256,
                "mode": "0444",
            },
            "artifact_count": 28,
        },
        "post-srun validation content drift",
    )
    captures = {
        "raw": raw_record.__dict__,
        "ready": ready_record.__dict__,
        "source_identity": source_record.__dict__,
        "post_srun_validation": post_record.__dict__,
        "artifact_leaves": leaves,
    }
    return captures, semantic_summary


def _git(repo: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(f"Git provenance failed: {error}") from error
    return completed.stdout.strip()


def _validate_standalone_detached_repository(repo: Path, field: str) -> Path:
    try:
        root_status = os.lstat(repo)
        git_status = os.lstat(repo / ".git")
        resolved = repo.resolve(strict=True)
    except OSError as error:
        raise VerificationError(f"cannot inspect {field}: {error}") from error
    _require(stat.S_ISDIR(root_status.st_mode), f"{field} root is not a directory")
    _require(stat.S_ISDIR(git_status.st_mode), f"{field} .git is not a directory")
    _require(
        Path(_git(repo, "rev-parse", "--show-toplevel")) == resolved,
        f"{field} is not top-level",
    )
    expected_git = (resolved / ".git").resolve(strict=True)
    _require(
        (resolved / _git(repo, "rev-parse", "--git-dir")).resolve(strict=True)
        == expected_git,
        f"{field} git-dir drift",
    )
    _require(
        (resolved / _git(repo, "rev-parse", "--git-common-dir")).resolve(strict=True)
        == expected_git,
        f"{field} common-dir drift",
    )
    _require(
        _git(repo, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD",
        f"{field} is not detached",
    )
    _require(_git(repo, "status", "--porcelain") == "", f"{field} is dirty")
    return resolved


def _validate_producer_repo(context: RuntimeContext) -> dict[str, Any]:
    resolved = _validate_standalone_detached_repository(
        context.producer_repo, "producer repo"
    )
    _require(_git(resolved, "rev-parse", "HEAD") == PRODUCER_COMMIT, "producer commit")
    _require(
        _git(resolved, "rev-parse", "HEAD^{tree}") == PRODUCER_TREE,
        "producer tree",
    )
    _require(_git(resolved, "rev-parse", "HEAD^") == PRODUCER_PARENT, "producer parent")
    actual: dict[str, str] = {}
    for relative, expected in PRODUCER_SOURCE_SHA256.items():
        data, _ = _secure_read(
            resolved / relative,
            f"producer source {relative}",
            required_mode=0o644,
            expected_sha256=expected,
        )
        actual[relative] = hashlib.sha256(data).hexdigest()
    return {
        "repo": str(context.producer_repo),
        "commit": PRODUCER_COMMIT,
        "tree": PRODUCER_TREE,
        "parent": PRODUCER_PARENT,
        "source_sha256": actual,
    }


def _validate_evidence_repo(
    context: RuntimeContext,
    *,
    expected_commit: str,
    expected_tree: str,
    expected_finalizer_sha256: str,
) -> dict[str, Any]:
    resolved = _validate_standalone_detached_repository(
        context.evidence_repo, "evidence repo"
    )
    _require(resolved != context.producer_repo.resolve(strict=True), "repo collision")
    commit = _git(resolved, "rev-parse", "HEAD")
    tree = _git(resolved, "rev-parse", "HEAD^{tree}")
    parent = _git(resolved, "rev-parse", "HEAD^")
    _require(commit == expected_commit, "evidence commit drift")
    _require(tree == expected_tree, "evidence tree drift")
    _require(parent == PRODUCER_COMMIT, "evidence parent drift")
    changed_paths = set(
        _git(
            resolved, "diff", "--name-only", f"{PRODUCER_COMMIT}..{commit}"
        ).splitlines()
    )
    _require(changed_paths == EVIDENCE_CHANGED_PATHS, "evidence changed-path drift")
    finalizer_path = Path(__file__).resolve(strict=True)
    _require(
        finalizer_path
        == (resolved / "scripts/finalize_splat_image_contract_smoke.py").resolve(
            strict=True
        ),
        "finalizer is outside evidence repo",
    )
    _, finalizer_record = _secure_read(
        finalizer_path,
        "finalizer source",
        required_mode=0o644,
        expected_sha256=expected_finalizer_sha256,
    )
    return {
        "repo": str(context.evidence_repo),
        "commit": commit,
        "tree": tree,
        "parent": parent,
        "changed_paths": sorted(changed_paths),
        "finalizer": finalizer_record.__dict__,
    }


def _require_canonical_l401_path(path: Path, field: str) -> None:
    try:
        relative = path.relative_to(LITERAL_USER_ROOT)
    except ValueError as error:
        raise VerificationError(f"{field} is outside the l401 user root") from error
    _require(
        path.resolve(strict=False) == CANONICAL_USER_ROOT / relative,
        f"{field} canonical alias drift",
    )


def _validate_context_paths(context: RuntimeContext) -> None:
    _require(context.job_name == "pol_img_9d29636", "job name contract drift")
    result_base = LITERAL_USER_ROOT / "results/polaris_eval/image_contract_smoke"
    try:
        result_relative = context.result_root.relative_to(result_base)
    except ValueError as error:
        raise VerificationError(
            "result root is outside image-contract results"
        ) from error
    _require(len(result_relative.parts) == 1, "result root nesting drift")
    _require(
        context.producer_repo
        == LITERAL_USER_ROOT
        / "src/PolaRiS-image-contract-smoke-9d29636-20260705T061008Z",
        "producer repo path drift",
    )
    _require(
        context.saved_job_script.parent == LITERAL_USER_ROOT / "launchers/polaris_eval",
        "saved wrapper directory drift",
    )
    _require(
        context.slurm_log
        == LITERAL_USER_ROOT
        / f"slurm_logs/polaris_eval/{context.job_name}-{context.job_id}.out",
        "Slurm log path drift",
    )
    expected = {
        "raw": context.result_root / f"smoke-{context.job_id}.raw.json",
        "ready": context.result_root / f"smoke-{context.job_id}.raw.json.ready.json",
        "source": context.result_root / f"source-identity-{context.job_id}.sha256",
        "post": context.result_root / f"post-srun-validation-{context.job_id}.json",
        "attestation": (
            context.result_root
            / f"smoke-{context.job_id}.image-evidence-attestation.json"
        ),
    }
    _require(context.raw_path == expected["raw"], "raw path relation")
    _require(context.ready_path == expected["ready"], "ready path relation")
    _require(context.source_identity_path == expected["source"], "source path relation")
    _require(context.post_srun_path == expected["post"], "post-srun path relation")
    _require(
        context.attestation_path == expected["attestation"], "attestation path relation"
    )
    for field, path in (
        ("result root", context.result_root),
        ("raw", context.raw_path),
        ("ready", context.ready_path),
        ("source identity", context.source_identity_path),
        ("post-srun", context.post_srun_path),
        ("attestation", context.attestation_path),
        ("producer repo", context.producer_repo),
        ("evidence repo", context.evidence_repo),
        ("saved job script", context.saved_job_script),
        ("Slurm log", context.slurm_log),
        ("sacct snapshot", context.sacct_snapshot),
    ):
        _require_canonical_l401_path(path, field)


SCHEDULER_ROW_FIELDS = {
    "job_id",
    "job_name",
    "account",
    "partition",
    "state",
    "exit_code",
    "elapsed_seconds",
    "start",
    "end",
    "node",
    "allocated_tres",
    "requested_tres",
}


def _parse_sacct_lines(lines: str, context: RuntimeContext) -> dict[str, Any]:
    rows = [line.split("|") for line in lines.splitlines() if line.strip()]
    _require(len(rows) == 4 and all(len(row) == 12 for row in rows), "sacct rows")
    names = {
        str(context.job_id): "allocation",
        f"{context.job_id}.batch": "batch",
        f"{context.job_id}.extern": "extern",
        f"{context.job_id}.0": "srun",
    }
    result: dict[str, Any] = {}
    for row in rows:
        (
            job_id,
            job_name,
            account,
            partition,
            state,
            exit_code,
            elapsed,
            start,
            end,
            node,
            allocated_tres,
            requested_tres,
        ) = row
        _require(job_id in names and names[job_id] not in result, "sacct step identity")
        _require(elapsed.isdigit(), "sacct elapsed")
        result[names[job_id]] = {
            "job_id": job_id,
            "job_name": job_name,
            "account": account,
            "partition": partition,
            "state": state,
            "exit_code": exit_code,
            "elapsed_seconds": int(elapsed),
            "start": start,
            "end": end,
            "node": node,
            "allocated_tres": allocated_tres,
            "requested_tres": requested_tres,
        }
    _require(set(result) == {"allocation", "batch", "extern", "srun"}, "sacct set")
    for name, row in result.items():
        _require(set(row) == SCHEDULER_ROW_FIELDS, f"sacct {name} schema")
        _require(row["state"] == "COMPLETED", f"sacct {name} state")
        _require(row["exit_code"] == "0:0", f"sacct {name} exit")
        _require(row["account"] == "nvr_lpr_rvp", f"sacct {name} account")
        _require(row["node"].startswith("pool0-"), f"sacct {name} node")
    _require(result["allocation"]["job_name"] == context.job_name, "job name")
    _require(result["allocation"]["partition"] == "batch", "partition")
    _require(result["batch"]["job_name"] == "batch", "batch name")
    _require(result["extern"]["job_name"] == "extern", "extern name")
    _require(result["srun"]["job_name"] == "env", "srun name")
    _require(
        result["allocation"]["node"]
        == result["batch"]["node"]
        == result["extern"]["node"]
        == result["srun"]["node"],
        "scheduler node drift",
    )
    _require(
        result["allocation"]["start"]
        == result["batch"]["start"]
        == result["extern"]["start"],
        "scheduler allocation start drift",
    )
    _require(
        result["allocation"]["end"]
        == result["batch"]["end"]
        == result["extern"]["end"]
        == result["srun"]["end"],
        "scheduler end drift",
    )
    _require(
        0
        < result["srun"]["elapsed_seconds"]
        <= result["allocation"]["elapsed_seconds"],
        "scheduler elapsed drift",
    )
    return result


def _scheduler_evidence(context: RuntimeContext) -> tuple[dict[str, Any], FileRecord]:
    snapshot_bytes, snapshot_record = _identity(
        context.sacct_snapshot, "sacct snapshot", context.sacct_snapshot_spec
    )
    snapshot = _strict_json(snapshot_bytes, "sacct snapshot")
    _require(
        type(snapshot) is dict
        and set(snapshot) == {"schema_version", "profile", "job_id", "scheduler"},
        "sacct snapshot schema",
    )
    _require(snapshot["schema_version"] == 1, "sacct snapshot version")
    _require(snapshot["profile"] == "slurm_terminal_job_steps_v1", "sacct profile")
    _require(snapshot["job_id"] == str(context.job_id), "sacct snapshot job")
    scheduler = snapshot["scheduler"]
    _require(type(scheduler) is dict, "sacct scheduler object")
    for name, row in scheduler.items():
        _require(name in {"allocation", "batch", "extern", "srun"}, "sacct row name")
        _require(
            type(row) is dict and set(row) == SCHEDULER_ROW_FIELDS, "sacct row schema"
        )
    live_sacct = shutil.which("sacct")
    _require(live_sacct is not None, "sacct unavailable")
    try:
        completed = subprocess.run(
            [
                live_sacct,
                "-j",
                str(context.job_id),
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,JobName,Account,Partition,State,ExitCode,"
                "ElapsedRaw,Start,End,NodeList,AllocTRES,ReqTRES",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationError(f"sacct query failed: {error}") from error
    actual = _parse_sacct_lines(completed.stdout, context)
    _require(_typed_equal(scheduler, actual), "sealed/live sacct drift")
    return actual, snapshot_record


def _validate_wrapper_and_log(context: RuntimeContext) -> dict[str, Any]:
    wrapper, wrapper_record = _identity(
        context.saved_job_script,
        "saved job script",
        context.saved_job_script_spec,
    )
    try:
        wrapper_text = wrapper.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"saved job script is not UTF-8: {error}") from error
    required_wrapper_fragments = (
        f"expected_commit={PRODUCER_COMMIT}",
        f"expected_tree={PRODUCER_TREE}",
        f"expected_parent={PRODUCER_PARENT}",
        f"expected_image_sha256={IMAGE_SHA256}",
        f"expected_scene_sha256={SCENE_SHA256}",
        f"expected_initial_conditions_sha256={INITIAL_CONDITIONS_SHA256}",
        f"expected_hub_revision={HUB_REVISION}",
        "scripts/smoke_splat_image_contract.py",
        "--container-remap-root",
        "--no-container-mount-home",
        "--expected-source-commit",
        "--expected-source-tree",
        "assert len(artifacts) == 28",
        "passed_after_zero_srun_exit",
    )
    for fragment in required_wrapper_fragments:
        _require(fragment in wrapper_text, f"wrapper lacks {fragment!r}")
    _require("#SBATCH --array" not in wrapper_text, "wrapper uses Slurm array")

    log, log_record = _identity(context.slurm_log, "Slurm log", context.slurm_log_spec)
    try:
        log_text = log.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VerificationError(f"Slurm log is not UTF-8: {error}") from error
    for forbidden in (
        "Traceback (most recent call last)",
        "AssertionError",
        "CUDA out of memory",
        "Out Of Memory",
        "Segmentation fault",
        "core dumped",
        "IMAGE_SMOKE_PREFLIGHT_ERROR=",
    ):
        _require(forbidden not in log_text, f"Slurm log contains {forbidden!r}")
    for required in (
        f"JOB_ID={context.job_id}",
        f"POLARIS_COMMIT={PRODUCER_COMMIT}",
        f"POLARIS_TREE={PRODUCER_TREE}",
        f"POLARIS_PARENT={PRODUCER_PARENT}",
        f"POLARIS_IMAGE_SHA256={IMAGE_SHA256}",
        f"POLARIS_HUB_REVISION={HUB_REVISION}",
        "IMAGE_SMOKE_GPU=NVIDIA L40S",
        f"POLARIS_IMAGE_SMOKE_RAW_SHA256={context.raw_spec.sha256}",
        f"IMAGE_SMOKE_POST_SRUN_VALIDATION_SHA256={context.post_srun_spec.sha256}",
        "IMAGE_SMOKE_EXIT_CODE=0",
    ):
        _require(required in log_text, f"Slurm log lacks {required!r}")
    _require(log_text.count("IMAGE_SMOKE_GPU=NVIDIA L40S") == 1, "L40S count drift")
    return {
        "saved_job_script": wrapper_record.__dict__,
        "slurm_log": log_record.__dict__,
    }


def _validate_small_runtime_input(
    path: Path,
    field: str,
    *,
    expected_size: int,
    expected_sha256: str,
    expected_mode: int,
    must_predate_ns: int,
    first_line: str | None = None,
) -> FileRecord:
    data, record = _secure_read(
        path,
        field,
        required_mode=expected_mode,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
    )
    _require(
        max(record.mtime_ns, record.ctime_ns) <= must_predate_ns,
        f"{field} does not predate srun",
    )
    if first_line is not None:
        try:
            actual_first_line = data.decode("utf-8").splitlines()[0]
        except (UnicodeDecodeError, IndexError) as error:
            raise VerificationError(
                f"{field} metadata text invalid: {error}"
            ) from error
        _require(actual_first_line == first_line, f"{field} revision drift")
    return record


def _validate_runtime_inputs(context: RuntimeContext) -> dict[str, Any]:
    image = _secure_hash(
        CONTAINER_IMAGE_PATH,
        "container image",
        required_mode=0o644,
        expected_size=7183130624,
        expected_sha256=IMAGE_SHA256,
        must_predate_ns=context.srun_start_epoch_ns,
    )
    scene = _validate_small_runtime_input(
        SCENE_PATH,
        "FoodBussing scene",
        expected_size=14914,
        expected_sha256=SCENE_SHA256,
        expected_mode=0o640,
        must_predate_ns=context.srun_start_epoch_ns,
    )
    initial = _validate_small_runtime_input(
        INITIAL_CONDITIONS_PATH,
        "FoodBussing initial conditions",
        expected_size=173951,
        expected_sha256=INITIAL_CONDITIONS_SHA256,
        expected_mode=0o640,
        must_predate_ns=context.srun_start_epoch_ns,
    )
    initial_metadata = _validate_small_runtime_input(
        INITIAL_METADATA_PATH,
        "initial-conditions Hub metadata",
        expected_size=101,
        expected_sha256=INITIAL_METADATA_SHA256,
        expected_mode=0o640,
        must_predate_ns=context.srun_start_epoch_ns,
        first_line=HUB_REVISION,
    )
    scene_metadata = _validate_small_runtime_input(
        SCENE_METADATA_PATH,
        "scene Hub metadata",
        expected_size=101,
        expected_sha256=SCENE_METADATA_SHA256,
        expected_mode=0o640,
        must_predate_ns=context.srun_start_epoch_ns,
        first_line=HUB_REVISION,
    )
    return {
        "container_image": image.__dict__,
        "scene": scene.__dict__,
        "initial_conditions": initial.__dict__,
        "initial_conditions_index": 0,
        "hub_revision": HUB_REVISION,
        "initial_conditions_metadata": initial_metadata.__dict__,
        "scene_metadata": scene_metadata.__dict__,
    }


def _build_expected(
    context: RuntimeContext,
    *,
    expected_evidence_commit: str,
    expected_evidence_tree: str,
    expected_finalizer_sha256: str,
) -> dict[str, Any]:
    _validate_context_paths(context)
    captures, semantics = _validate_capture(context)
    producer = _validate_producer_repo(context)
    reviewer = _validate_evidence_repo(
        context,
        expected_commit=expected_evidence_commit,
        expected_tree=expected_evidence_tree,
        expected_finalizer_sha256=expected_finalizer_sha256,
    )
    scheduler, sacct_record = _scheduler_evidence(context)
    wrapper_and_log = _validate_wrapper_and_log(context)
    runtime_inputs = _validate_runtime_inputs(context)
    return {
        "schema_version": 1,
        "profile": (
            f"polaris_foodbussing_image_contract_evidence_job{context.job_id}_v1"
        ),
        "status": "evidence_bundle_validated",
        "scope": SCOPE,
        "producer": producer,
        "reviewer": reviewer,
        "job": {
            "job_id": str(context.job_id),
            "job_name": context.job_name,
            "scheduler": scheduler,
            "sacct_snapshot": sacct_record.__dict__,
        },
        "runtime_inputs": runtime_inputs,
        "capture": {
            **captures,
            **wrapper_and_log,
        },
        "semantic_evidence": semantics,
        "authorizations": {
            "checkpoint_evaluation": False,
            "policy_serving": False,
            "task_metric": False,
            "benchmark_result": False,
            "controller_behavior": False,
            "canary": False,
            "smoke_suite": False,
            "standard_suite": False,
            "promotion": False,
        },
    }


def _publish_nonoverwriting(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o444)
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_argument(value: str) -> str:
    if len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise argparse.ArgumentTypeError("expected lowercase SHA-256")
    return value


def _git_oid_argument(value: str) -> str:
    if len(value) != 40 or set(value) - set("0123456789abcdef"):
        raise argparse.ArgumentTypeError("expected lowercase 40-character Git OID")
    return value


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected positive integer")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("finalize", "verify"))
    parser.add_argument("--job-id", type=_positive_integer, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--ready-marker", type=Path, required=True)
    parser.add_argument("--source-identity", type=Path, required=True)
    parser.add_argument("--post-srun-validation", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--producer-repo", type=Path, required=True)
    parser.add_argument("--evidence-repo", type=Path, required=True)
    parser.add_argument("--saved-job-script", type=Path, required=True)
    parser.add_argument("--slurm-log", type=Path, required=True)
    parser.add_argument("--sacct-snapshot", type=Path, required=True)
    parser.add_argument("--expected-raw-size", type=_positive_integer, required=True)
    parser.add_argument("--expected-raw-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--expected-ready-size", type=_positive_integer, required=True)
    parser.add_argument("--expected-ready-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--expected-source-size", type=_positive_integer, required=True)
    parser.add_argument(
        "--expected-source-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-post-srun-size", type=_positive_integer, required=True
    )
    parser.add_argument(
        "--expected-post-srun-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument(
        "--expected-wrapper-size", type=_positive_integer, required=True
    )
    parser.add_argument(
        "--expected-wrapper-sha256", type=_sha256_argument, required=True
    )
    parser.add_argument("--expected-log-size", type=_positive_integer, required=True)
    parser.add_argument("--expected-log-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--expected-sacct-size", type=_positive_integer, required=True)
    parser.add_argument("--expected-sacct-sha256", type=_sha256_argument, required=True)
    parser.add_argument(
        "--expected-artifact-manifest-sha256",
        type=_sha256_argument,
        required=True,
    )
    parser.add_argument("--srun-start-epoch-ns", type=_positive_integer, required=True)
    parser.add_argument(
        "--expected-evidence-commit", type=_git_oid_argument, required=True
    )
    parser.add_argument(
        "--expected-evidence-tree", type=_git_oid_argument, required=True
    )
    parser.add_argument(
        "--expected-finalizer-sha256", type=_sha256_argument, required=True
    )
    return parser


def _context(args: argparse.Namespace) -> RuntimeContext:
    return RuntimeContext(
        job_id=args.job_id,
        job_name=args.job_name,
        result_root=args.result_root,
        raw_path=args.raw_result,
        ready_path=args.ready_marker,
        source_identity_path=args.source_identity,
        post_srun_path=args.post_srun_validation,
        attestation_path=args.attestation,
        producer_repo=args.producer_repo,
        evidence_repo=args.evidence_repo,
        saved_job_script=args.saved_job_script,
        slurm_log=args.slurm_log,
        sacct_snapshot=args.sacct_snapshot,
        raw_spec=ExpectedLeaf(args.expected_raw_size, args.expected_raw_sha256, 0o444),
        ready_spec=ExpectedLeaf(
            args.expected_ready_size, args.expected_ready_sha256, 0o444
        ),
        source_identity_spec=ExpectedLeaf(
            args.expected_source_size, args.expected_source_sha256, 0o444
        ),
        post_srun_spec=ExpectedLeaf(
            args.expected_post_srun_size,
            args.expected_post_srun_sha256,
            0o444,
        ),
        saved_job_script_spec=ExpectedLeaf(
            args.expected_wrapper_size,
            args.expected_wrapper_sha256,
            0o444,
        ),
        slurm_log_spec=ExpectedLeaf(
            args.expected_log_size, args.expected_log_sha256, 0o644
        ),
        sacct_snapshot_spec=ExpectedLeaf(
            args.expected_sacct_size, args.expected_sacct_sha256, 0o444
        ),
        artifact_manifest_sha256=args.expected_artifact_manifest_sha256,
        srun_start_epoch_ns=args.srun_start_epoch_ns,
    )


def main() -> int:
    args = _parser().parse_args()
    context = _context(args)
    try:
        expected = _build_expected(
            context,
            expected_evidence_commit=args.expected_evidence_commit,
            expected_evidence_tree=args.expected_evidence_tree,
            expected_finalizer_sha256=args.expected_finalizer_sha256,
        )
        expected_bytes = _serialized(expected)
        if args.mode == "finalize":
            _publish_nonoverwriting(context.attestation_path, expected_bytes)
        actual_bytes, actual_record = _secure_read(
            context.attestation_path,
            "image evidence attestation",
            required_mode=0o444,
        )
        _require(actual_bytes == expected_bytes, "attestation byte drift")
        _require(
            _typed_equal(_strict_json(actual_bytes, "attestation"), expected),
            "attestation semantic drift",
        )
    except (OSError, VerificationError, ValueError) as error:
        print(f"POLARIS_IMAGE_EVIDENCE_INVALID={error}", file=sys.stderr, flush=True)
        return 1
    print(
        f"POLARIS_IMAGE_EVIDENCE_VALID={context.attestation_path}",
        flush=True,
    )
    print(
        f"POLARIS_IMAGE_EVIDENCE_SIZE_BYTES={actual_record.size_bytes}",
        flush=True,
    )
    print(
        f"POLARIS_IMAGE_EVIDENCE_SHA256={actual_record.sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
