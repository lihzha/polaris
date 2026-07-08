#!/usr/bin/env python3
"""Build or verify a closed PolaRiS task-and-robot regular-file manifest.

The simulator and splat renderer consume more than the top-level scene and robot
USD files: task payload meshes and splats are loaded from the complete task tree,
while robot splats are discovered by globbing ``nvidia_droid/SEGMENTED``.  Binding
every regular file beneath both roots is a deliberately conservative closure that
also detects future, otherwise-unreviewed files appearing in either runtime tree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any


ASSET_MANIFEST_PROFILE = "polaris_task_robot_complete_regular_file_tree_v1"
POLARIS_DATA_REVISION = "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b"
ROBOT_ASSET_SUBDIR = "nvidia_droid"
EXPECTED_ASSET_TREES = {
    "block_stack_kitchen": {
        "file_count": 38,
        "total_bytes": 483_960_124,
        "tree_sha256": (
            "f5f1fe057fc5daf7edc4b597eebedf5ac02796509cebee481059422b875e3c1c"
        ),
    },
    "food_bussing": {
        "file_count": 36,
        "total_bytes": 380_426_267,
        "tree_sha256": (
            "36b80c4a9499b0643bec2775e6c33a0517212b494065cba6c1f94e511f9fd094"
        ),
    },
    "pan_clean": {
        "file_count": 38,
        "total_bytes": 385_693_281,
        "tree_sha256": (
            "2a2f3a46821371564dbefe22af256553e58a5d6116f7b795abb0289bf4e5871a"
        ),
    },
    "move_latte_cup": {
        "file_count": 62,
        "total_bytes": 471_416_390,
        "tree_sha256": (
            "41c6b70bb3d5cc25f46ea15539308800cc9bb2abdcb168452541065b30f9173f"
        ),
    },
    "organize_tools": {
        "file_count": 53,
        "total_bytes": 376_291_177,
        "tree_sha256": (
            "3e5039dcab1a78dba193e2cdc3e460095dd4e119f01fd5a7b702732947d2cce6"
        ),
    },
    "tape_into_container": {
        "file_count": 44,
        "total_bytes": 374_778_953,
        "tree_sha256": (
            "de58ccda5f7090c3c128989f01be893a171349b632035cd50bfa04462949f87c"
        ),
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    before = path.stat(follow_symlinks=False)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ValueError(f"Asset changed while hashing: {path}")
    return digest.hexdigest()


def _roles(relative_path: str, *, task_subdir: str) -> list[str]:
    path = Path(relative_path)
    roles = ["task_tree" if path.parts[0] == task_subdir else "robot_tree"]
    if relative_path == f"{task_subdir}/initial_conditions.json":
        roles.append("initial_conditions")
    if relative_path == f"{task_subdir}/scene.usda":
        roles.append("scene_root_layer")
    if relative_path == f"{ROBOT_ASSET_SUBDIR}/noninstanceable.usd":
        roles.append("robot_root_layer")
    if path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
        roles.append("usd_layer_or_payload")
    if path.name == "splat.ply":
        roles.append("splat_runtime")
    if (
        path.parts[0] == ROBOT_ASSET_SUBDIR
        and "SEGMENTED" in path.parts
        and path.suffix.lower() == ".ply"
    ):
        roles.append("robot_segmented_splat")
    return sorted(roles)


def scan_asset_tree(data_root: Path, task_subdir: str) -> dict[str, Any]:
    """Hash the complete selected task and shared robot trees."""

    if task_subdir not in EXPECTED_ASSET_TREES:
        raise ValueError(f"Unsupported PolaRiS task asset directory: {task_subdir}")
    raw_root = Path(data_root)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise ValueError("PolaRiS data root must be one regular directory")
    root = raw_root.resolve()
    records = []
    for subdir in (task_subdir, ROBOT_ASSET_SUBDIR):
        tree = root / subdir
        if tree.is_symlink() or not tree.is_dir():
            raise ValueError(f"Missing regular asset tree: {tree}")
        for directory, directory_names, file_names in os.walk(
            tree, topdown=True, followlinks=False
        ):
            directory_path = Path(directory)
            for name in sorted(directory_names):
                candidate = directory_path / name
                if candidate.is_symlink():
                    raise ValueError(f"Asset tree contains a symlink: {candidate}")
            for name in sorted(file_names):
                candidate = directory_path / name
                metadata = candidate.stat(follow_symlinks=False)
                if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
                    raise ValueError(
                        f"Asset tree contains a non-regular file: {candidate}"
                    )
                resolved = candidate.resolve()
                try:
                    relative = resolved.relative_to(root).as_posix()
                except ValueError as error:
                    raise ValueError(
                        f"Asset escaped the data root: {candidate}"
                    ) from error
                records.append(
                    {
                        "relative_path": relative,
                        "size": metadata.st_size,
                        "sha256": _sha256_file(candidate),
                        "roles": _roles(relative, task_subdir=task_subdir),
                    }
                )
    records.sort(key=lambda record: record["relative_path"])
    identity_records = [
        {
            "relative_path": record["relative_path"],
            "size": record["size"],
            "sha256": record["sha256"],
        }
        for record in records
    ]
    return {
        "records": records,
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "tree_sha256": hashlib.sha256(
            canonical_json_bytes(identity_records)
        ).hexdigest(),
    }


def make_asset_manifest(
    data_root: Path,
    task_subdir: str,
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scanned = scan_asset_tree(data_root, task_subdir)
    expected = copy.deepcopy(
        EXPECTED_ASSET_TREES[task_subdir] if expected is None else expected
    )
    actual_identity = {
        key: scanned[key] for key in ("file_count", "total_bytes", "tree_sha256")
    }
    if actual_identity != expected:
        raise ValueError(
            f"PolaRiS {task_subdir} asset-tree identity mismatch: "
            f"expected={expected}, actual={actual_identity}"
        )
    manifest = {
        "schema_version": 1,
        "profile": ASSET_MANIFEST_PROFILE,
        "status": "pass",
        "polaris_data_revision": POLARIS_DATA_REVISION,
        "task_subdir": task_subdir,
        "root_subdirs": [task_subdir, ROBOT_ASSET_SUBDIR],
        "closure": (
            "all_regular_files_under_selected_task_and_nvidia_droid_"
            "including_payloads_splats_and_segmented_meshes"
        ),
        **actual_identity,
        "records": scanned["records"],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(manifest)
    ).hexdigest()
    return validate_asset_manifest(manifest, expected=expected)


def validate_asset_manifest(
    value: Any, *, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    required = {
        "schema_version",
        "profile",
        "status",
        "polaris_data_revision",
        "task_subdir",
        "root_subdirs",
        "closure",
        "file_count",
        "total_bytes",
        "tree_sha256",
        "records",
        "manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("PolaRiS asset manifest schema mismatch")
    task_subdir = value["task_subdir"]
    if task_subdir not in EXPECTED_ASSET_TREES:
        raise ValueError("PolaRiS asset manifest task is unsupported")
    if (
        value["schema_version"] != 1
        or value["profile"] != ASSET_MANIFEST_PROFILE
        or value["status"] != "pass"
        or value["polaris_data_revision"] != POLARIS_DATA_REVISION
        or value["root_subdirs"] != [task_subdir, ROBOT_ASSET_SUBDIR]
        or value["closure"]
        != (
            "all_regular_files_under_selected_task_and_nvidia_droid_"
            "including_payloads_splats_and_segmented_meshes"
        )
    ):
        raise ValueError("PolaRiS asset manifest identity mismatch")
    records = value["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("PolaRiS asset manifest records are empty")
    if records != sorted(records, key=lambda record: record.get("relative_path", "")):
        raise ValueError("PolaRiS asset manifest records are not sorted")
    seen = set()
    identity_records = []
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "size",
            "sha256",
            "roles",
        }:
            raise ValueError(f"PolaRiS asset record {index} schema mismatch")
        relative = record["relative_path"]
        digest = record["sha256"]
        roles = record["roles"]
        if (
            not isinstance(relative, str)
            or relative.startswith(("/", "../"))
            or "/../" in relative
            or relative in seen
            or relative.split("/", 1)[0] not in {task_subdir, ROBOT_ASSET_SUBDIR}
        ):
            raise ValueError(f"PolaRiS asset record {index} path mismatch")
        if type(record["size"]) is not int or record["size"] < 0:
            raise ValueError(f"PolaRiS asset record {index} size mismatch")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"PolaRiS asset record {index} digest mismatch")
        if (
            not isinstance(roles, list)
            or not roles
            or roles != sorted(set(roles))
            or roles != _roles(relative, task_subdir=task_subdir)
        ):
            raise ValueError(f"PolaRiS asset record {index} role mismatch")
        seen.add(relative)
        identity_records.append(
            {
                "relative_path": relative,
                "size": record["size"],
                "sha256": digest,
            }
        )
    actual_identity = {
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "tree_sha256": hashlib.sha256(
            canonical_json_bytes(identity_records)
        ).hexdigest(),
    }
    expected = EXPECTED_ASSET_TREES[task_subdir] if expected is None else expected
    if any(value[key] != actual_identity[key] for key in actual_identity):
        raise ValueError("PolaRiS asset manifest aggregate mismatch")
    if actual_identity != expected:
        raise ValueError("PolaRiS asset manifest differs from pinned tree identity")
    payload = copy.deepcopy(value)
    claimed = payload.pop("manifest_sha256")
    if claimed != hashlib.sha256(canonical_json_bytes(payload)).hexdigest():
        raise ValueError("PolaRiS asset manifest SHA-256 mismatch")
    required_roles = {
        "initial_conditions",
        "scene_root_layer",
        "robot_root_layer",
        "usd_layer_or_payload",
        "splat_runtime",
        "robot_segmented_splat",
    }
    observed_roles = {role for record in records for role in record["roles"]}
    if not required_roles.issubset(observed_roles):
        raise ValueError("PolaRiS asset manifest lacks required runtime file roles")
    return copy.deepcopy(value)


def publish_asset_manifest(
    path: Path,
    manifest: dict[str, Any],
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = validate_asset_manifest(manifest, expected=expected)
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"PolaRiS asset manifest exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    payload = (
        json.dumps(canonical, indent=2, sort_keys=True, allow_nan=False).encode("ascii")
        + b"\n"
    )
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o444)
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
    return validate_asset_manifest_artifact(destination, expected=expected)


def validate_asset_manifest_artifact(
    path: Path,
    *,
    data_root: Path | None = None,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = Path(path)
    if artifact.is_symlink() or not artifact.is_file():
        raise ValueError("PolaRiS asset manifest must be a regular file")
    metadata = artifact.stat()
    if metadata.st_mode & 0o777 != 0o444 or metadata.st_nlink != 1:
        raise ValueError("PolaRiS asset manifest must be immutable mode 0444")
    raw = artifact.read_bytes()
    manifest = validate_asset_manifest(json.loads(raw), expected=expected)
    if data_root is not None:
        rescanned = scan_asset_tree(Path(data_root), manifest["task_subdir"])
        rescanned_identity = {
            key: rescanned[key] for key in ("file_count", "total_bytes", "tree_sha256")
        }
        expected_identity = {
            key: manifest[key] for key in ("file_count", "total_bytes", "tree_sha256")
        }
        if (
            rescanned_identity != expected_identity
            or rescanned["records"] != manifest["records"]
        ):
            raise ValueError("Live PolaRiS asset tree differs from immutable manifest")
    return {
        "path": str(artifact.resolve()),
        "size": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_sha256": manifest["manifest_sha256"],
        "tree_sha256": manifest["tree_sha256"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "mode": "0444",
        "nlink": metadata.st_nlink,
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--task-subdir", choices=sorted(EXPECTED_ASSET_TREES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if (args.output is None) == (args.verify is None):
        parser.error("exactly one of --output or --verify is required")
    if args.output is not None:
        if args.task_subdir is None:
            parser.error("--task-subdir is required with --output")
        manifest = make_asset_manifest(args.data_root, args.task_subdir)
        result = publish_asset_manifest(args.output, manifest)
    else:
        if args.task_subdir is not None:
            parser.error("--task-subdir is inferred from --verify")
        result = validate_asset_manifest_artifact(args.verify, data_root=args.data_root)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
