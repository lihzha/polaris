#!/usr/bin/env python3
"""Verify the released pi0.5-Polaris checkpoint against its GCS object manifest."""

import argparse
import base64
import hashlib
import json
from pathlib import Path


PREFIX = "checkpoints/polaris/pi05_droid_jointpos_polaris/"


def _manifest_entries(manifest: Path) -> list[tuple[str, int, str]]:
    entries = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 3 or not fields[0].startswith(PREFIX):
            raise ValueError(f"Invalid manifest line {line_number}: {line!r}")
        entries.append((fields[0][len(PREFIX) :], int(fields[1]), fields[2]))
    if not entries:
        raise ValueError("Checkpoint manifest is empty")
    return entries


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return base64.b64encode(digest.digest()).decode()


def verify_checkpoint(
    checkpoint: Path,
    manifest: Path,
    *,
    full_md5: bool,
) -> dict:
    checkpoint = checkpoint.resolve()
    manifest = manifest.resolve()
    entries = _manifest_entries(manifest)
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()

    expected_paths = {relative_path for relative_path, _, _ in entries}
    actual_paths = set()
    for path in checkpoint.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Checkpoint must not contain symlinks: {path}")
        if path.is_file():
            actual_paths.add(path.relative_to(checkpoint).as_posix())
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ValueError(
            f"Checkpoint file-set mismatch: missing={missing}, extra={extra}"
        )

    verified_bytes = 0
    for relative_path, expected_size, expected_md5 in entries:
        path = checkpoint / relative_path
        if not path.is_file():
            raise ValueError(f"Missing checkpoint object: {path}")
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"Checkpoint object size mismatch for {relative_path}: "
                f"expected {expected_size}, got {actual_size}"
            )
        if full_md5:
            actual_md5 = _md5_base64(path)
            if actual_md5 != expected_md5:
                raise ValueError(
                    f"Checkpoint object MD5 mismatch for {relative_path}: "
                    f"expected {expected_md5}, got {actual_md5}"
                )
        verified_bytes += actual_size

    return {
        "schema_version": 1,
        "status": "pass",
        "checkpoint_path": str(checkpoint),
        "manifest_path": str(manifest),
        "manifest_sha256": manifest_sha256,
        "object_count": len(entries),
        "total_bytes": verified_bytes,
        "full_md5": full_md5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--full-md5", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = verify_checkpoint(
        args.checkpoint,
        args.manifest,
        full_md5=args.full_md5,
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
