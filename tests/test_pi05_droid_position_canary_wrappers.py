from __future__ import annotations

import copy
import base64
import hashlib
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest

from scripts.polaris import finalize_pi05_droid_position_eval as finalizer
from scripts.polaris import verify_pi05_droid_position_checkpoint as checkpoint


def _identity(path: str, digest: str) -> dict[str, object]:
    return {
        "path": path,
        "size": 123,
        "sha256": digest,
        "mode": "0444",
        "nlink": 1,
    }


def _canary_source() -> dict[str, object]:
    files = {
        relative: {
            "size": index + 1,
            "sha256": f"{index + 1:064x}",
            "git_blob_sha1": f"{index + 1:040x}",
        }
        for index, relative in enumerate(finalizer.CONTROLLER_GOVERNED_PATHS)
    }
    return {
        "root": "/canary",
        "commit": "c" * 40,
        "detached_head": True,
        "tracked_and_untracked_clean": True,
        "files": files,
    }


def _assets() -> dict[str, object]:
    return {
        "root": "/assets",
        "hub_revision": "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b",
        "assets": {
            relative: {
                **_identity(f"/assets/{relative}", f"{index + 31:064x}"),
                "metadata": _identity(
                    f"/assets/.cache/{relative}.metadata", f"{index + 61:064x}"
                ),
                "hub_revision": "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b",
            }
            for index, relative in enumerate(finalizer.PI05_DROID_CANARY_ASSETS)
        },
    }


def _hardened_attestation(
    canary_source: dict[str, object],
    image: dict[str, object],
    assets: dict[str, object],
) -> dict[str, object]:
    governed = {
        relative: {
            "relative_path": relative,
            **copy.deepcopy(canary_source["files"][relative]),
        }
        for relative in finalizer.CONTROLLER_GOVERNED_PATHS
    }
    smoke_assets = {
        relative: {
            "asset": {
                key: record[key] for key in ("path", "size", "sha256", "mode", "nlink")
            },
            "metadata": record["metadata"],
            "hub_revision": record["hub_revision"],
        }
        for relative, record in assets["assets"].items()
    }
    smoke_sbatch = governed[
        "scripts/polaris/l40s_pi05_droid_position_controller_smoke.sbatch"
    ]
    saved_job_script = {
        "path": "/evidence/job.sbatch",
        "size": smoke_sbatch["size"],
        "sha256": smoke_sbatch["sha256"],
        "mode": "0444",
        "nlink": 1,
    }
    return {
        "schema_version": 1,
        "profile": finalizer.CONTROLLER_ATTESTATION_PROFILE,
        "status": "pass",
        "scope": "position_controller_only_no_model_or_checkpoint",
        "promotion": "forbidden_without_separate_checkpoint_canary",
        "slurm": {
            "job_id": 1234,
            "srun_exit_code": 0,
            "status_artifact": _identity("/evidence/status.json", "1" * 64),
            "gpu_inventory": {
                "artifact": _identity("/evidence/gpu.json", "2" * 64),
                "gpus": [
                    {
                        "uuid": "GPU-test",
                        "name": "NVIDIA L40S",
                        "driver_version": "1",
                    }
                ],
            },
            "saved_job_script": saved_job_script,
        },
        "source": {
            "root": "/smoke-source",
            "commit": "a" * 40,
            "detached_clean": True,
            "files": governed,
        },
        "container_image": image,
        "assets": smoke_assets,
        "runtime_identity": {
            "container_pinned_by_sha256": True,
            "isaaclab_version": "2.2.0",
            "isaaclab_source_sha256": {"source": "4" * 64},
            "polaris_runtime_source_sha256": {"source": "5" * 64},
            "action_term_class": "position.Action",
            "action_cfg_class": "position.ActionCfg",
            "host_finalizer_python": {
                "executable": "/python",
                "implementation": "CPython",
                "version": "3.11",
            },
        },
        "smoke": _identity("/evidence/smoke.json", "6" * 64),
        "child_close_capture": _identity("/evidence/raw.json", "7" * 64),
        "child_ready_marker": _identity("/evidence/ready.json", "8" * 64),
    }


def _install_attestation_fakes(
    monkeypatch: pytest.MonkeyPatch,
    attestation: dict[str, object],
) -> None:
    artifact_by_path = {
        "/attestation.json": {
            "path": "/attestation.json",
            "size": 999,
            "sha256": "f" * 64,
            "mode": "0444",
            "nlink": 1,
            "value": attestation,
        },
        "/evidence/status.json": {
            **attestation["slurm"]["status_artifact"],
            "value": {"job_id": 1234, "srun_exit_code": 0},
        },
        "/evidence/gpu.json": {
            **attestation["slurm"]["gpu_inventory"]["artifact"],
            "value": {
                "schema_version": 1,
                "job_id": 1234,
                "gpus": attestation["slurm"]["gpu_inventory"]["gpus"],
            },
        },
        "/evidence/smoke.json": {
            **attestation["smoke"],
            "value": {"kind": "parent"},
        },
        "/evidence/raw.json": {
            **attestation["child_close_capture"],
            "value": {"kind": "child"},
        },
    }

    def fake_validate(path: Path):
        return copy.deepcopy(artifact_by_path[str(path)])

    identities = {
        str(attestation["slurm"]["status_artifact"]["path"]): attestation["slurm"][
            "status_artifact"
        ],
        str(attestation["slurm"]["gpu_inventory"]["artifact"]["path"]): attestation[
            "slurm"
        ]["gpu_inventory"]["artifact"],
        str(attestation["slurm"]["saved_job_script"]["path"]): attestation["slurm"][
            "saved_job_script"
        ],
        str(attestation["smoke"]["path"]): attestation["smoke"],
        str(attestation["child_close_capture"]["path"]): attestation[
            "child_close_capture"
        ],
        str(attestation["child_ready_marker"]["path"]): attestation[
            "child_ready_marker"
        ],
    }

    monkeypatch.setattr(finalizer, "validate_immutable_json", fake_validate)
    monkeypatch.setattr(
        finalizer,
        "_identity",
        lambda path, **_: copy.deepcopy(identities[str(path)]),
    )
    smoke = {
        "completion": {"raw_sha256": "7" * 64, "ready_sha256": "8" * 64},
        "runtime_contract": {
            "isaaclab_version": "2.2.0",
            "isaaclab_source_sha256": {"source": "4" * 64},
            "polaris_runtime_source_sha256": {"source": "5" * 64},
            "action_term_class": "position.Action",
            "action_cfg_class": "position.ActionCfg",
        },
    }
    monkeypatch.setattr(
        finalizer, "validate_position_smoke", lambda *_args, **_kwargs: smoke
    )
    monkeypatch.setattr(
        finalizer.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )


def test_hardened_ancestor_attestation_accepts_identical_governed_bytes(monkeypatch):
    source = _canary_source()
    image = _identity("/image.sqsh", "9" * 64)
    assets = _assets()
    attestation = _hardened_attestation(source, image, assets)
    _install_attestation_fakes(monkeypatch, attestation)

    validated = finalizer.validate_position_controller_attestation(
        Path("/attestation.json"),
        "f" * 64,
        canary_source=source,
        expected_image=image,
        expected_assets=assets,
    )

    assert validated["source_commit"] == "a" * 40
    assert validated["source_is_ancestor"] is True
    assert validated["governed_source_identity_match"] is True
    assert validated["value"] == attestation


def test_hardened_attestation_rejects_divergent_governed_bytes(monkeypatch):
    source = _canary_source()
    image = _identity("/image.sqsh", "9" * 64)
    assets = _assets()
    attestation = _hardened_attestation(source, image, assets)
    first = finalizer.CONTROLLER_GOVERNED_PATHS[0]
    attestation["source"]["files"][first]["sha256"] = "0" * 64
    _install_attestation_fakes(monkeypatch, attestation)

    with pytest.raises(ValueError, match="governed source diverged"):
        finalizer.validate_position_controller_attestation(
            Path("/attestation.json"),
            "f" * 64,
            canary_source=source,
            expected_image=image,
            expected_assets=assets,
        )


def test_obsolete_minimal_controller_attestation_is_rejected(monkeypatch):
    source = _canary_source()
    image = _identity("/image.sqsh", "9" * 64)
    assets = _assets()
    obsolete = {
        "schema_version": 1,
        "profile": finalizer.CONTROLLER_ATTESTATION_PROFILE,
        "status": "pass",
        "job_id": 1234,
        "polaris_commit": "a" * 40,
        "smoke": _identity("/evidence/smoke.json", "6" * 64),
    }
    monkeypatch.setattr(
        finalizer,
        "validate_immutable_json",
        lambda _path: {
            "path": "/attestation.json",
            "size": 999,
            "sha256": "f" * 64,
            "mode": "0444",
            "nlink": 1,
            "value": obsolete,
        },
    )

    with pytest.raises(ValueError, match="attestation identity mismatch"):
        finalizer.validate_position_controller_attestation(
            Path("/attestation.json"),
            "f" * 64,
            canary_source=source,
            expected_image=image,
            expected_assets=assets,
        )


def test_run_snapshot_is_independent_mode_locked_and_nlink_one(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    payload = b"position-checkpoint-bytes"
    source_file = source / "params/data"
    source_file.parent.mkdir()
    source_file.write_bytes(payload)
    md5 = base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode(
        "ascii"
    )
    monkeypatch.setattr(
        checkpoint,
        "_read_manifest",
        lambda _path: [("params/data", len(payload), md5)],
    )

    destination = tmp_path / "run/checkpoint_snapshot"
    destination.parent.mkdir()
    result = checkpoint.create_run_snapshot(
        source, destination, tmp_path / "manifest.tsv"
    )

    copied = destination / "params/data"
    assert result["snapshot_dir"] == str(destination)
    assert result["snapshot_root_stat"]["mode"] == "0555"
    assert copied.read_bytes() == payload
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert stat.S_IMODE(copied.parent.stat().st_mode) == 0o555
    assert stat.S_IMODE(copied.stat().st_mode) == 0o444
    assert copied.stat().st_nlink == 1
    assert (copied.stat().st_dev, copied.stat().st_ino) != (
        source_file.stat().st_dev,
        source_file.stat().st_ino,
    )


def test_checkpoint_pre_post_comparison_excludes_only_phase():
    value = {
        "verification_phase": "pre_server",
        "root_stat": {"inode": 1},
        "objects": [{"relative_path": "x", "stat": {"inode": 2}}],
    }
    before = {"checkpoint": copy.deepcopy(value)}
    after = {"checkpoint": copy.deepcopy(value)}
    after["checkpoint"]["verification_phase"] = "post_server"
    finalizer._require_checkpoint_unchanged(before, after)

    after["checkpoint"]["objects"][0]["stat"]["inode"] = 3
    with pytest.raises(ValueError, match="changed across"):
        finalizer._require_checkpoint_unchanged(before, after)


def test_launchers_are_non_array_and_fail_closed_on_old_gates():
    root = Path(__file__).resolve().parents[1]
    launchers = (
        root / "scripts/polaris/eval_pi05_droid_position.sh",
        root / "scripts/polaris/l40s_pi05_droid_position_canary.sbatch",
        root / "scripts/polaris/submit_pi05_droid_position_canary.sh",
    )
    for path in launchers:
        text = path.read_text(encoding="utf-8")
        assert "#SBATCH --array" not in text
        assert "sbatch --array" not in text
        assert "CONTROLLER_COMPLETION" in text
        assert "ALL_SIX_CONTROLLER_COMPLETION" in text
        assert "Old direct-rad/s controller gate is forbidden" in text


def test_resolved_contract_binds_train_matched_images_stats_and_flow():
    contract = finalizer.resolved_contract()
    assert contract["protocol"] == (
        "polaris-native-droid-freshq-delta0p2-position-h8-canary1-v1"
    )
    assert contract["inference"] == {
        "mode": "flow",
        "sampler": "flow_euler_t1_to_t0_num_steps10_rng_key0_v1",
        "flow_steps": 10,
        "rng_seed": 0,
        "response_shape": [15, 8],
        "execute_first": 8,
    }
    assert contract["observations"]["model_image_order"] == [
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb_masked",
    ]
    assert contract["observations"]["native_images"] == {
        "external": [720, 1280, 3],
        "wrist": [720, 1280, 3],
        "dtype": "uint8_rgb",
    }
    assert contract["observations"]["wrist_rotation_degrees"] == 0
    assert contract["normalization"]["scope"] == "checkpoint_global_droid"
    assert contract["normalization"]["category_override"] == "forbidden"
    assert contract["normalization"]["single_arm"] == (
        "forbidden_checkpoint_has_global_stats"
    )
    assert contract["execution"]["arm_formula"] == (
        "q_target_t=fresh_measured_q_t+0.2*clip(command_t,-1,1)"
    )
