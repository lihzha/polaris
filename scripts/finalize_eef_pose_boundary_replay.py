#!/usr/bin/env python3
"""Finalize or verify a PolaRiS EEF boundary-replay smoke attestation.

Run this stdlib-only program on the Slurm host after the exact ``srun`` has
returned zero.  It validates the immutable raw/ready pair, all live boundary
evidence, the committed replay source, current asset bytes, and launch
provenance before non-overwriting publication of a mode-0444 attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any

try:
    from scripts import smoke_eef_pose_boundary_replay as smoke
except ImportError:  # Direct ``python scripts/finalize_...py`` execution.
    import smoke_eef_pose_boundary_replay as smoke

BoundaryError = smoke.BoundaryReplayValidationError


ATTESTATION_FIELDS = {
    "schema_version",
    "finalized",
    "passed",
    "stage",
    "job_id",
    "srun_rc",
    "raw_result",
    "validation_summary",
    "provenance",
}


class FinalizationError(ValueError):
    """Raw evidence or provenance does not satisfy the promotion contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mode(path: Path) -> str:
    return f"{stat.S_IMODE(path.stat().st_mode):04o}"


def _file_identity(path: Path, field: str) -> dict[str, Any]:
    _require(path.is_file(), f"{field} does not exist: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "size_bytes": size,
        "sha256": digest.hexdigest(),
        "mode": _mode(path),
    }


def _read_json(path: Path, field: str) -> tuple[dict[str, Any], bytes, str]:
    identity = _file_identity(path, field)
    data = path.read_bytes()
    value = smoke.strict_json_loads(data, field=field)
    return value, data, identity["sha256"]


def _same_typed(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _same_typed(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_typed(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _require_hex_digest(value: str, field: str, *, length: int = 64) -> None:
    _require(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value),
        f"{field} is malformed",
    )


def _require_live_identity(recorded: Any, field: str) -> dict[str, Any]:
    _require(isinstance(recorded, dict), f"{field} identity is not an object")
    _require(
        {"path", "size_bytes", "sha256", "mode"}.issubset(recorded),
        f"{field} identity schema",
    )
    path = recorded.get("path")
    _require(isinstance(path, str) and path, f"{field} path")
    actual = _file_identity(Path(path), field)
    for name in ("path", "size_bytes", "sha256", "mode"):
        _require(recorded.get(name) == actual[name], f"{field} {name} changed")
    return actual


def _strict_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _publish_nonoverwriting(path: Path, payload: dict[str, Any]) -> None:
    _require(not path.exists(), f"attestation already exists: {path}")
    serialized = _strict_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
        published_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(published_fd)
        finally:
            os.close(published_fd)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_asset_identities(raw: dict[str, Any]) -> dict[str, Any]:
    assets = raw.get("assets")
    _require(isinstance(assets, dict), "raw assets missing")
    scene = _require_live_identity(assets.get("scene"), "FoodBussing scene")
    initial_conditions = _require_live_identity(
        assets.get("initial_conditions"), "FoodBussing initial conditions"
    )
    _require(
        scene["sha256"] == smoke.EXPECTED_ASSET_CONTRACT["scene_sha256"],
        "live scene no longer matches the pinned SHA",
    )
    _require(
        initial_conditions["sha256"]
        == smoke.EXPECTED_ASSET_CONTRACT["initial_conditions_sha256"],
        "live initial conditions no longer match the pinned SHA",
    )
    metadata = assets.get("revision_metadata")
    _require(isinstance(metadata, dict), "raw revision metadata missing")
    metadata_identities: dict[str, Any] = {}
    for filename in ("initial_conditions.json", "scene.usda"):
        recorded = metadata.get(filename)
        actual = _require_live_identity(recorded, f"Hub metadata {filename}")
        _require(
            recorded.get("revision")
            == smoke.EXPECTED_ASSET_CONTRACT["polaris_hub_revision"],
            f"Hub metadata {filename} revision drift",
        )
        metadata_identities[filename] = {
            **actual,
            "revision": recorded["revision"],
        }
    return {
        "scene": scene,
        "initial_conditions": initial_conditions,
        "revision_metadata": metadata_identities,
        "polaris_hub_revision": smoke.EXPECTED_ASSET_CONTRACT["polaris_hub_revision"],
    }


def build_expected_attestation(args: argparse.Namespace) -> dict[str, Any]:
    """Reconstruct the one acceptable attestation from raw bytes and provenance."""

    _require(args.srun_rc == 0, "srun_rc must be exactly zero")
    _require(type(args.job_id) is int and args.job_id > 0, "job_id must be positive")
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    _require(
        isinstance(slurm_job_id, str)
        and slurm_job_id.isdecimal()
        and int(slurm_job_id) == args.job_id,
        "CLI job_id does not match SLURM_JOB_ID",
    )
    slurm_nodelist = os.environ.get("SLURM_NODELIST")
    _require(
        isinstance(slurm_nodelist, str) and bool(slurm_nodelist.strip()),
        "SLURM_NODELIST must be a nonempty exact string",
    )
    for value, field, length in (
        (args.expected_polaris_commit, "expected PolaRiS commit", 40),
        (args.expected_runner_sha256, "expected runner SHA-256", 64),
        (args.expected_fixture_sha256, "expected fixture SHA-256", 64),
        (args.expected_image_sha256, "expected image SHA-256", 64),
        (args.expected_finalizer_sha256, "expected finalizer SHA-256", 64),
        (
            args.expected_saved_job_script_sha256,
            "expected saved job script SHA-256",
            64,
        ),
    ):
        _require_hex_digest(value, field, length=length)
    _require(
        args.expected_fixture_sha256 == smoke.EXPECTED_FIXTURE_SHA256,
        "CLI fixture digest is not the protocol-pinned fixture",
    )
    _require(
        args.expected_safety_profile
        in {
            smoke.BASE_SAFETY_PROFILE,
            smoke.WRIST_ENERGY_BRAKE_CANDIDATE_PROFILE,
        },
        "expected safety profile is not supported",
    )

    expected_raw_name = f"boundary-replay-smoke-{args.job_id}.json"
    expected_attestation_name = f"boundary-replay-smoke-{args.job_id}.attestation.json"
    _require(args.raw_result.name == expected_raw_name, "raw filename/job mismatch")
    _require(
        args.attestation.name == expected_attestation_name,
        "attestation filename/job mismatch",
    )
    _require(
        args.raw_result.parent.resolve() == args.attestation.parent.resolve(),
        "raw and attestation directories differ",
    )
    _require(
        args.raw_result.resolve() != args.attestation.resolve(),
        "raw/attestation path collision",
    )

    raw, raw_bytes, raw_sha256 = _read_json(args.raw_result, "raw result")
    _require(_mode(args.raw_result) == "0444", "raw result mode must be 0444")
    validation_summary = smoke.validate_success_payload(raw)
    _require(
        validation_summary.get("safety_profile") == args.expected_safety_profile,
        "raw result safety profile does not match the expected controller",
    )
    _require(
        raw.get("initial_ik_safety_capture", {}).get("profile")
        == raw.get("runtime_frame", {}).get("ik_safety_profile")
        == raw.get("ik_safety", {}).get("profile")
        == args.expected_safety_profile,
        "raw initial/frame/final safety profiles are not cross-bound",
    )
    raw_identity = {
        "path": str(args.raw_result.resolve()),
        "size_bytes": len(raw_bytes),
        "sha256": raw_sha256,
        "mode": "0444",
    }
    marker_path = args.raw_result.with_name(args.raw_result.name + ".ready.json")
    marker, marker_bytes, marker_sha256 = _read_json(marker_path, "ready marker")
    _require(_mode(marker_path) == "0444", "ready marker mode must be 0444")
    expected_marker = {
        "schema_version": 1,
        "stage": "simulation_app_close_pending",
        "raw_result": raw_identity,
    }
    _require(
        _same_typed(marker, expected_marker),
        "ready marker does not bind the exact raw result",
    )

    repo = args.polaris_repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    _require(commit == args.expected_polaris_commit, "PolaRiS commit mismatch")
    _require(_git(repo, "status", "--porcelain") == "", "PolaRiS worktree dirty")
    runner = _file_identity(
        repo / "scripts" / "smoke_eef_pose_boundary_replay.py", "boundary runner"
    )
    _require(
        Path(smoke.__file__).resolve() == Path(runner["path"]),
        "imported boundary validator is not the hashed runner source",
    )
    _require(
        runner["sha256"] == args.expected_runner_sha256,
        "boundary runner digest mismatch",
    )
    fixture_path = (
        repo
        / "scripts"
        / "fixtures"
        / "official_lap3b_foodbussing_v3_boundary_actions.json"
    )
    fixture, actions = smoke.load_replay_fixture(fixture_path)
    _require(len(actions) == 378, "live replay fixture action count")
    _require(
        fixture["sha256"] == args.expected_fixture_sha256,
        "live replay fixture digest mismatch",
    )
    recorded_fixture = raw.get("fixture")
    _require(
        isinstance(recorded_fixture, dict)
        and all(
            recorded_fixture.get(field) == fixture[field]
            for field in (
                "path",
                "size_bytes",
                "sha256",
                "mode",
                "fixture_profile",
                "source_trace_sha256",
                "action_float32_sha256",
                "action_count",
            )
        ),
        "raw fixture identity differs from the committed replay fixture",
    )

    finalizer = _file_identity(Path(__file__).resolve(), "boundary finalizer")
    _require(
        finalizer["sha256"] == args.expected_finalizer_sha256,
        "boundary finalizer digest mismatch",
    )
    image = _file_identity(args.container_image, "container image")
    _require(
        image["sha256"] == args.expected_image_sha256,
        "container image digest mismatch",
    )
    runtime_script = _file_identity(args.runtime_job_script, "runtime job script")
    saved_script = _file_identity(args.saved_job_script, "saved job script")
    _require(
        runtime_script["sha256"] == saved_script["sha256"],
        "runtime/saved job scripts differ",
    )
    _require(saved_script["mode"] == "0444", "saved job script must be mode 0444")
    _require(
        saved_script["sha256"] == args.expected_saved_job_script_sha256,
        "saved job script digest mismatch",
    )
    asset_identities = _validate_asset_identities(raw)

    result = {
        "schema_version": 1,
        "finalized": True,
        "passed": True,
        "stage": "complete",
        "job_id": args.job_id,
        "srun_rc": args.srun_rc,
        "raw_result": {
            **raw_identity,
            "ready_marker": {
                "path": str(marker_path.resolve()),
                "size_bytes": len(marker_bytes),
                "sha256": marker_sha256,
                "mode": "0444",
            },
        },
        "validation_summary": validation_summary,
        "provenance": {
            "slurm": {
                "job_id": args.job_id,
                "nodelist": slurm_nodelist,
            },
            "polaris_repo": str(repo),
            "polaris_commit": commit,
            "expected_safety_profile": args.expected_safety_profile,
            "runner": runner,
            "fixture": fixture,
            "fixture_source": smoke.EXPECTED_SOURCE,
            "assets": asset_identities,
            "container_image": image,
            "runtime_job_script": runtime_script,
            "saved_job_script": saved_script,
            "finalizer": finalizer,
        },
    }
    _require(set(result) == ATTESTATION_FIELDS, "attestation schema drift")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("finalize", "verify"))
    parser.add_argument("--raw-result", required=True, type=Path)
    parser.add_argument("--attestation", required=True, type=Path)
    parser.add_argument("--srun-rc", required=True, type=int)
    parser.add_argument("--job-id", required=True, type=int)
    parser.add_argument("--runtime-job-script", required=True, type=Path)
    parser.add_argument("--saved-job-script", required=True, type=Path)
    parser.add_argument("--polaris-repo", required=True, type=Path)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--expected-safety-profile", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--container-image", required=True, type=Path)
    parser.add_argument("--expected-image-sha256", required=True)
    parser.add_argument("--expected-finalizer-sha256", required=True)
    parser.add_argument("--expected-saved-job-script-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        expected = build_expected_attestation(args)
        if args.mode == "finalize":
            _publish_nonoverwriting(args.attestation, expected)
        attestation, attestation_bytes, attestation_sha256 = _read_json(
            args.attestation, "attestation"
        )
        _require(_mode(args.attestation) == "0444", "attestation mode must be 0444")
        _require(
            _same_typed(attestation, expected),
            "attestation content differs from reconstructed evidence",
        )
    except (
        BoundaryError,
        FinalizationError,
        OSError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"BOUNDARY_REPLAY_ATTESTATION_FAIL={error}", file=sys.stderr, flush=True)
        return 1
    print(f"BOUNDARY_REPLAY_ATTESTATION_PASS={args.attestation}", flush=True)
    print(
        f"BOUNDARY_REPLAY_ATTESTATION_SIZE_BYTES={len(attestation_bytes)}",
        flush=True,
    )
    print(
        f"BOUNDARY_REPLAY_ATTESTATION_SHA256={attestation_sha256}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
