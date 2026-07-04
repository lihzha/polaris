#!/usr/bin/env python3
"""Pure host-side namespace validation and immutable SUCCESS publication."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Callable


class GateIoError(RuntimeError):
    """One fail-closed gate-I/O invariant was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateIoError(message)


def _lexical_directory(raw: str, *, field: str) -> str:
    require(type(raw) is str and bool(raw), f"{field} must be a nonempty string")
    require(
        not any(character in raw for character in ("\x00", "\n", "\r", "\t")),
        f"{field} contains a forbidden control character",
    )
    require(os.path.isabs(raw), f"{field} must be absolute")
    lexical = raw if raw == os.sep else raw.rstrip(os.sep)
    require(bool(lexical), f"{field} lexical path is empty")
    require(
        os.path.normpath(raw) == lexical and not lexical.startswith(os.sep * 2),
        f"{field} must not contain lexical aliases",
    )
    try:
        metadata = os.lstat(lexical)
    except OSError as error:
        raise GateIoError(f"{field} lstat failed: {error}") from error
    require(
        stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode),
        f"{field} must be a non-symlink directory",
    )
    canonical = os.path.realpath(lexical)
    try:
        canonical_metadata = os.stat(canonical)
    except OSError as error:
        raise GateIoError(f"{field} canonical stat failed: {error}") from error
    try:
        final_metadata = os.lstat(lexical)
    except OSError as error:
        raise GateIoError(f"{field} final lstat failed: {error}") from error
    require(
        stat.S_ISDIR(canonical_metadata.st_mode)
        and stat.S_ISDIR(final_metadata.st_mode)
        and not stat.S_ISLNK(final_metadata.st_mode)
        and (metadata.st_dev, metadata.st_ino)
        == (canonical_metadata.st_dev, canonical_metadata.st_ino)
        and (final_metadata.st_dev, final_metadata.st_ino)
        == (canonical_metadata.st_dev, canonical_metadata.st_ino),
        f"{field} changed during canonicalization",
    )
    require(
        not any(character in canonical for character in ("\n", "\r", "\t")),
        f"{field} canonical path contains a forbidden control character",
    )
    return canonical


def validate_disjoint_roots(output_root: str, cache_root: str) -> tuple[str, str]:
    """Validate lexical roots before canonical overlap comparison."""

    output = _lexical_directory(output_root, field="output root")
    cache = _lexical_directory(cache_root, field="cache root")
    try:
        common = os.path.commonpath((output, cache))
    except ValueError as error:
        raise GateIoError(f"output/cache root comparison failed: {error}") from error
    require(
        common not in {output, cache},
        f"resolved output/cache roots overlap: {output!r}, {cache!r}",
    )
    return output, cache


def success_payload(
    *,
    job_id: str,
    variant: str,
    result_sha256: str,
    video_sha256: str,
    manifest_sha256: str,
) -> bytes:
    require(re.fullmatch(r"[0-9]+", job_id) is not None, "invalid SUCCESS job ID")
    require(
        re.fullmatch(r"[a-z0-9_]+", variant) is not None,
        "invalid SUCCESS variant",
    )
    for field, value in (
        ("result", result_sha256),
        ("video", video_sha256),
        ("manifest", manifest_sha256),
    ):
        require(
            re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"invalid SUCCESS {field} SHA-256",
        )
    return (
        "profile=production_v4_core_fulltrace_success_v1\n"
        f"job_id={job_id}\n"
        f"variant={variant}\n"
        f"result_sha256={result_sha256}\n"
        f"video_sha256={video_sha256}\n"
        f"manifest_sha256={manifest_sha256}\n"
    ).encode()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_owned(path: Path, identity: tuple[int, int]) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    if (metadata.st_dev, metadata.st_ino) != identity:
        return False
    os.unlink(path)
    return True


def _read_regular_file(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        require(stat.S_ISREG(metadata.st_mode), "SUCCESS is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        final_metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    require(
        (metadata.st_dev, metadata.st_ino, metadata.st_size)
        == (final_metadata.st_dev, final_metadata.st_ino, sum(map(len, chunks))),
        "SUCCESS changed during reread",
    )
    return b"".join(chunks), final_metadata


def publish_success(
    temporary: Path,
    marker: Path,
    payload: bytes,
    *,
    after_link: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Publish SUCCESS without replacement and remove only owned inodes."""

    temporary = Path(temporary)
    marker = Path(marker)
    require(temporary.parent == marker.parent, "SUCCESS temp/marker parent mismatch")
    parent = temporary.parent
    try:
        parent_metadata = os.lstat(parent)
    except OSError as error:
        raise GateIoError(f"SUCCESS parent lstat failed: {error}") from error
    require(
        stat.S_ISDIR(parent_metadata.st_mode)
        and not stat.S_ISLNK(parent_metadata.st_mode),
        "SUCCESS parent must be a non-symlink directory",
    )
    require(type(payload) is bytes and bool(payload), "SUCCESS payload must be bytes")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor: int | None = None
    owned_identity: tuple[int, int] | None = None
    marker_link_attempted = False
    marker_link_created = False
    try:
        descriptor = os.open(temporary, flags, 0o600)
        temporary_metadata = os.fstat(descriptor)
        owned_identity = (temporary_metadata.st_dev, temporary_metadata.st_ino)
        require(
            stat.S_ISREG(temporary_metadata.st_mode)
            and temporary_metadata.st_nlink == 1,
            "exclusive SUCCESS temp metadata drift",
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "short SUCCESS marker write")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        marker_link_attempted = True
        os.link(temporary, marker, follow_symlinks=False)
        marker_link_created = True
        if after_link is not None:
            after_link()
        os.unlink(temporary)
        _fsync_directory(parent)
        published = os.lstat(marker)
        require(
            (published.st_dev, published.st_ino) == owned_identity
            and stat.S_ISREG(published.st_mode)
            and stat.S_IMODE(published.st_mode) == 0o444
            and published.st_nlink == 1
            and published.st_size == len(payload),
            "published SUCCESS metadata drift",
        )
        reread, reread_metadata = _read_regular_file(marker)
        require(
            reread == payload
            and (reread_metadata.st_dev, reread_metadata.st_ino) == owned_identity
            and stat.S_IMODE(reread_metadata.st_mode) == 0o444
            and reread_metadata.st_nlink == 1,
            "published SUCCESS reread drift",
        )
        final_marker = os.lstat(marker)
        require(
            (final_marker.st_dev, final_marker.st_ino) == owned_identity
            and stat.S_ISREG(final_marker.st_mode)
            and stat.S_IMODE(final_marker.st_mode) == 0o444
            and final_marker.st_nlink == 1
            and final_marker.st_size == len(payload),
            "published SUCCESS final-path drift",
        )
        return {
            "path": str(marker.resolve(strict=True)),
            "size_bytes": len(reread),
            "sha256": hashlib.sha256(reread).hexdigest(),
            "mode": f"{stat.S_IMODE(reread_metadata.st_mode):04o}",
            "nlink": reread_metadata.st_nlink,
        }
    except BaseException as error:
        cleanup_errors: list[str] = []
        if descriptor is not None:
            if owned_identity is None:
                try:
                    temporary_metadata = os.fstat(descriptor)
                    owned_identity = (
                        temporary_metadata.st_dev,
                        temporary_metadata.st_ino,
                    )
                except OSError as cleanup_error:
                    cleanup_errors.append(
                        f"recover temp descriptor identity: {cleanup_error}"
                    )
            try:
                os.close(descriptor)
            except OSError as cleanup_error:
                cleanup_errors.append(f"close temp descriptor: {cleanup_error}")
            descriptor = None
        if owned_identity is not None and (
            marker_link_attempted or marker_link_created
        ):
            try:
                _unlink_if_owned(marker, owned_identity)
            except OSError as cleanup_error:
                cleanup_errors.append(f"unlink owned marker: {cleanup_error}")
        if owned_identity is not None:
            try:
                _unlink_if_owned(temporary, owned_identity)
            except OSError as cleanup_error:
                cleanup_errors.append(f"unlink owned temp: {cleanup_error}")
            try:
                remaining_temp = os.lstat(temporary)
            except FileNotFoundError:
                pass
            except OSError as cleanup_error:
                cleanup_errors.append(f"verify owned temp removal: {cleanup_error}")
            else:
                if (remaining_temp.st_dev, remaining_temp.st_ino) == owned_identity:
                    cleanup_errors.append("owned SUCCESS temp remains after cleanup")
        try:
            _fsync_directory(parent)
        except OSError as cleanup_error:
            cleanup_errors.append(f"fsync SUCCESS directory: {cleanup_error}")
        if cleanup_errors:
            add_note = getattr(error, "add_note", None)
            if add_note is not None:
                add_note("SUCCESS cleanup errors: " + "; ".join(cleanup_errors))
            else:
                error.args = (
                    *error.args,
                    "SUCCESS cleanup errors: " + "; ".join(cleanup_errors),
                )
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _forced_post_link_failure() -> None:
    raise GateIoError("forced post-link publication failure")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    roots = subparsers.add_parser("validate-roots")
    roots.add_argument("--output-root", required=True)
    roots.add_argument("--cache-root", required=True)
    publish = subparsers.add_parser("publish-success")
    publish.add_argument("--temporary", type=Path, required=True)
    publish.add_argument("--marker", type=Path, required=True)
    publish.add_argument("--job-id", required=True)
    publish.add_argument("--variant", required=True)
    publish.add_argument("--result-sha256", required=True)
    publish.add_argument("--video-sha256", required=True)
    publish.add_argument("--manifest-sha256", required=True)
    publish.add_argument(
        "--test-fail-after-link", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    try:
        if args.command == "validate-roots":
            output, cache = validate_disjoint_roots(args.output_root, args.cache_root)
            print(f"{output}\t{cache}")
        else:
            payload = success_payload(
                job_id=args.job_id,
                variant=args.variant,
                result_sha256=args.result_sha256,
                video_sha256=args.video_sha256,
                manifest_sha256=args.manifest_sha256,
            )
            publish_success(
                args.temporary,
                args.marker,
                payload,
                after_link=(
                    _forced_post_link_failure if args.test_fail_after_link else None
                ),
            )
    except (GateIoError, OSError) as error:
        print(f"FULLTRACE_GATE_IO_ERROR={error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
