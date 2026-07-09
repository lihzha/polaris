import hashlib
import json
from pathlib import Path

import pytest

import polaris.pi05_droid_jointpos_evidence as evidence
import polaris.pi05_droid_jointpos_runtime as runtime
from polaris.pi05_droid_jointpos_image_contract import (
    CLIENT_RESIZE_PROFILE,
    IMAGE_PROFILE,
    static_image_contract,
)


SERVER_SHA = "1" * 64
RUNTIME_SHA = "2" * 64
COMMIT = "3" * 40
GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"


def _write_inputs(root: Path, *, rollouts: int = 2):
    run_dir = root / "run"
    task_dir = run_dir / "DROID-FoodBussing"
    task_dir.mkdir(parents=True)
    paths = {
        **{
            name: run_dir / relative
            for name, relative in evidence._RUN_ARTIFACTS.items()
        },
        **{
            name: task_dir / relative
            for name, relative in evidence._TASK_ARTIFACTS.items()
        },
    }
    videos = [task_dir / f"episode_{index}.mp4" for index in range(rollouts)]
    terminal_images = [
        task_dir / f"episode_{index}_terminal.png" for index in range(rollouts)
    ]
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic {name}\n".encode())
    trace = paths["policy_trace"]
    trace.write_bytes(b'{"record":"synthetic"}\n')
    trace_sha = hashlib.sha256(trace.read_bytes()).hexdigest()
    metrics_sha = hashlib.sha256(paths["metrics_csv"].read_bytes()).hexdigest()
    paths["trace_summary"].write_text(
        json.dumps(
            {
                "schema_version": 5,
                "status": "pass",
                "trace_sha256": trace_sha,
                "metrics_sha256": metrics_sha,
                "reset_count": rollouts,
                "episode_lengths": [450] * rollouts,
                "episode_query_counts": [57] * rollouts,
                "cumulative_query_counts": [
                    57 * (index + 1) for index in range(rollouts)
                ],
                "query_records": 57 * rollouts,
                "global_query_indices_contiguous": True,
                "environment_image_profile": IMAGE_PROFILE,
                "environment_image_contract": static_image_contract(),
                "final_composite_image_shape": [720, 1280, 3],
                "request_image_shape": [224, 224, 3],
                "request_image_dtype": "uint8",
                "client_model_spatial_transform": CLIENT_RESIZE_PROFILE,
                "server_model_resize": (
                    evidence.PI05_DROID_JOINTPOS_SERVER_MODEL_RESIZE
                ),
                "model_image_shape": [224, 224, 3],
                "server_resize_behavior": "early_return_same_array_no_pixel_change",
                "visualization_image_shape": [224, 224, 3],
                "query_visualization_is_model_input": True,
                "query_visualization_source": (
                    "byte_identical_client224_wire_model_input"
                ),
                "interquery_visualization_is_model_input": False,
                "interquery_visualization_source": (
                    "client224_resize_of_nonexpensive_sim_camera_non_model_input"
                ),
                "expensive_render_cadence": (
                    "reset_then_post_actions_7_15_through_447_for_next_query"
                ),
                "query_frames_per_episode": 57,
                "diagnostic_video_frames_per_episode": 450,
                "terminal_visualization_shape": [224, 448, 3],
                "terminal_visualization_dtype": "uint8",
                "terminal_visualization_source": (
                    "post_action450_returned_nonexpensive_sim_camera_observation"
                ),
                "terminal_visualization_sha256": ["7" * 64] * rollouts,
                "server_contract_sha256": SERVER_SHA,
                "runtime_contract_sha256": RUNTIME_SHA,
            }
        )
        + "\n"
    )
    for index, path in enumerate(videos):
        path.write_bytes(f"video {index}\n".encode())
    for index, path in enumerate(terminal_images):
        path.write_bytes(f"terminal image {index}\n".encode())
    return run_dir, task_dir, paths, videos, terminal_images


def _stub_contracts(
    _paths,
    _expected_rollouts,
    _video_identities,
    _terminal_image_identities,
    _terminal_pixel_sha256,
):
    return {
        "server_contract_sha256": SERVER_SHA,
        "runtime_contract_sha256": RUNTIME_SHA,
    }


def _stub_sealed_trace_audit(paths, **_kwargs):
    for name in ("policy_trace", "metrics_csv"):
        assert paths[name].stat().st_mode & 0o777 == 0o444
    return json.loads(paths["trace_summary"].read_text())


def _gpu_vulkan_contracts():
    gpu = {
        "query": [
            "/usr/bin/nvidia-smi",
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        "uuid": GPU_UUID,
        "name": "NVIDIA L40S",
        "driver_version": "580.105.08",
    }
    return (
        {"jax": {"nvidia_smi": dict(gpu)}},
        {
            "nvidia_smi": dict(gpu),
            "vulkan": {
                "vk_driver_files": "/etc/vulkan/icd.d/nvidia_icd.json",
                "icd": {
                    "path": "/etc/vulkan/icd.d/nvidia_icd.json",
                    "size": 140,
                    "sha256": (
                        "7bdb6f27d35b66fc848df6f94b8773b"
                        "ba30ea3a7f06f114100d14154a235a34b"
                    ),
                },
            },
            "graphics_runtime": {
                "profile": evidence.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE,
                "cv2_loader": {
                    "profile": (
                        runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_PROFILE
                    ),
                    "files": [
                        {"path": path, "size": size, "sha256": sha256}
                        for path, size, sha256 in (
                            runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_FILES
                        )
                    ],
                    "module": {
                        **dict(
                            runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_MODULE_IDENTITY
                        ),
                        "native_maps_device": "0:1",
                        "native_maps_inode": 1000,
                    },
                    "loader_search_safety": {
                        "profile": (
                            runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_SEARCH_SAFETY_PROFILE
                        ),
                        "working_directory": "/immutable/polaris",
                        "working_directory_binding": (
                            "equals_runtime_module_repository_root"
                        ),
                        "working_directory_read_only": True,
                        "normalized_cv2_binary_path": ("/.venv/lib/python3.11/lib64"),
                        "normalized_cv2_binary_path_exists": False,
                        "working_directory_library_candidates": [],
                    },
                },
                "libraries": [
                    {} for _ in evidence.PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES
                ],
                "graphics_runtime_sha256": (
                    evidence.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_SHA256
                ),
            },
        },
    )


def _patch_closed_contracts(monkeypatch):
    monkeypatch.setattr(evidence, "_validate_specialized_contracts", _stub_contracts)
    monkeypatch.setattr(
        evidence, "_independently_audit_sealed_trace", _stub_sealed_trace_audit
    )


def test_finalize_seals_every_output_and_revalidates_manifest(tmp_path, monkeypatch):
    run_dir, task_dir, paths, videos, terminal_images = _write_inputs(tmp_path)
    _patch_closed_contracts(monkeypatch)

    result = evidence.finalize_evidence(
        run_dir=run_dir,
        task_dir=task_dir,
        environment="DROID-FoodBussing",
        expected_environment_seed=0,
        expected_rollouts=2,
        polaris_commit=COMMIT,
    )

    manifest_path = run_dir / evidence.PI05_DROID_JOINTPOS_EVIDENCE_MANIFEST
    assert result["manifest"]["path"] == str(manifest_path.resolve())
    assert result["value"]["contracts"] == {
        **_stub_contracts(
            paths,
            2,
            result["value"]["videos"],
            result["value"]["terminal_images"],
            ["7" * 64] * 2,
        ),
        "sealed_trace_csv_reaudit_sha256": hashlib.sha256(
            evidence._canonical_json_bytes(
                json.loads(paths["trace_summary"].read_text())
            )
        ).hexdigest(),
    }
    assert set(result["value"]["artifacts"]) == set(paths)
    assert len(result["value"]["videos"]) == 2
    assert len(result["value"]["terminal_images"]) == 2
    for path in [*paths.values(), *videos, *terminal_images, manifest_path]:
        stat = path.stat()
        assert stat.st_mode & 0o777 == 0o444
        assert stat.st_nlink == 1

    again = evidence.validate_evidence_manifest(
        manifest_path,
        run_dir=run_dir,
        task_dir=task_dir,
        environment="DROID-FoodBussing",
        expected_environment_seed=0,
        expected_rollouts=2,
        polaris_commit=COMMIT,
    )
    assert again == result


def test_manifest_revalidation_rejects_post_close_mutation(tmp_path, monkeypatch):
    run_dir, task_dir, paths, _videos, _terminal_images = _write_inputs(
        tmp_path, rollouts=1
    )
    _patch_closed_contracts(monkeypatch)
    evidence.finalize_evidence(
        run_dir=run_dir,
        task_dir=task_dir,
        environment="DROID-FoodBussing",
        expected_environment_seed=0,
        expected_rollouts=1,
        polaris_commit=COMMIT,
    )
    paths["metrics_csv"].chmod(0o644)

    with pytest.raises(ValueError, match="Immutable file identity mismatch"):
        evidence.validate_evidence_manifest(
            run_dir / evidence.PI05_DROID_JOINTPOS_EVIDENCE_MANIFEST,
            run_dir=run_dir,
            task_dir=task_dir,
            environment="DROID-FoodBussing",
            expected_environment_seed=0,
            expected_rollouts=1,
            polaris_commit=COMMIT,
        )


def test_finalize_rejects_extra_or_missing_video_before_publication(tmp_path):
    run_dir, task_dir, _paths, _videos, _terminal_images = _write_inputs(
        tmp_path, rollouts=1
    )
    (task_dir / "episode_7.mp4").write_bytes(b"substitution\n")

    with pytest.raises(ValueError, match="video filename/count mismatch"):
        evidence.finalize_evidence(
            run_dir=run_dir,
            task_dir=task_dir,
            environment="DROID-FoodBussing",
            expected_environment_seed=0,
            expected_rollouts=1,
            polaris_commit=COMMIT,
        )
    assert not (run_dir / evidence.PI05_DROID_JOINTPOS_EVIDENCE_MANIFEST).exists()


def test_finalize_rejects_missing_terminal_image_before_publication(tmp_path):
    run_dir, task_dir, _paths, _videos, terminal_images = _write_inputs(
        tmp_path, rollouts=1
    )
    terminal_images[0].unlink()

    with pytest.raises(ValueError, match="terminal-image filename/count mismatch"):
        evidence.finalize_evidence(
            run_dir=run_dir,
            task_dir=task_dir,
            environment="DROID-FoodBussing",
            expected_environment_seed=0,
            expected_rollouts=1,
            polaris_commit=COMMIT,
        )
    assert not (run_dir / evidence.PI05_DROID_JOINTPOS_EVIDENCE_MANIFEST).exists()


def test_jointpos_launchers_bind_snapshot_or_reject_untracked_source():
    root = Path(__file__).parents[1]
    snapshot_launchers = (
        "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh",
        "scripts/polaris/l40s_pi05_eval_job.sbatch",
        "scripts/polaris/submit_pi05_droid_jointpos_polaris.sh",
    )
    for relative in snapshot_launchers:
        source = (root / relative).read_text()
        assert "EXPECTED_POLARIS_SOURCE_TREE_SHA256" in source
        assert (
            "pi05_droid_jointpos_consumer_binding" in source
            or "--source-digest" in source
            or relative.endswith("l40s_pi05_eval_job.sbatch")
        )
    for relative in (
        "scripts/polaris/l40s_pi05_jointpos_seed_repeat.sbatch",
        "scripts/polaris/submit_pi05_jointpos_seed_repeat.sh",
    ):
        source = (root / relative).read_text()
        assert "status --porcelain=v1 --untracked-files=all" in source
        assert "diff-index --quiet HEAD" not in source


def test_worker_finalizes_server_rng_then_evidence_before_success_marker():
    root = Path(__file__).parents[1]
    source = (root / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh").read_text()
    final_attestation = source.index("final_server_attestation_line=")
    signal = source.index('kill -USR1 "${rng_server_pid}"', final_attestation)
    rng_proof = source.index("verify_pi05_droid_jointpos_rng_stream.py", signal)
    video_decode = source.index('"${video_validation_command[@]}"', rng_proof)
    finalize = source.index("-m polaris.pi05_droid_jointpos_evidence", video_decode)
    finalized = source.index("EVIDENCE_FINALIZED=1", finalize)
    success = source.index('publish_terminal_marker "${TASK_DIR}/SUCCESS"', finalized)
    assert (
        final_attestation
        < signal
        < rng_proof
        < video_decode
        < finalize
        < finalized
        < success
    )


def test_gpu_vulkan_contract_requires_model_simulator_agreement():
    assert evidence.PI05_DROID_JOINTPOS_EVIDENCE_PROFILE == (
        "openpi_pi05_droid_jointpos_polaris_evidence_transaction_v10"
    )
    model, simulator = _gpu_vulkan_contracts()
    assert evidence._validate_gpu_vulkan_runtime_agreement(model, simulator) == {
        "nvidia_gpu_uuid": GPU_UUID,
        "nvidia_gpu_name": "NVIDIA L40S",
        "nvidia_driver_version": "580.105.08",
        "vulkan_icd_container_path": "/etc/vulkan/icd.d/nvidia_icd.json",
        "vulkan_icd_sha256": (
            "7bdb6f27d35b66fc848df6f94b8773bba30ea3a7f06f114100d14154a235a34b"
        ),
        "graphics_runtime_profile": (
            evidence.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE
        ),
        "graphics_runtime_sha256": (
            evidence.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_SHA256
        ),
        "graphics_cv2_loader_profile": (
            runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_PROFILE
        ),
        "graphics_cv2_module_identity": {
            **dict(runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_MODULE_IDENTITY),
            "native_maps_device": "0:1",
            "native_maps_inode": 1000,
        },
        "graphics_cv2_loader_search_safety": {
            "profile": (
                runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_SEARCH_SAFETY_PROFILE
            ),
            "working_directory": "/immutable/polaris",
            "working_directory_binding": "equals_runtime_module_repository_root",
            "working_directory_read_only": True,
            "normalized_cv2_binary_path": "/.venv/lib/python3.11/lib64",
            "normalized_cv2_binary_path_exists": False,
            "working_directory_library_candidates": [],
        },
        "graphics_cv2_loader_files": [
            {"path": path, "size": size, "sha256": sha256}
            for path, size, sha256 in (
                runtime.PI05_DROID_JOINTPOS_GRAPHICS_CV2_LOADER_FILES
            )
        ],
        "graphics_library_count": len(
            evidence.PI05_DROID_JOINTPOS_GRAPHICS_LIBRARY_IDENTITIES
        ),
    }

    for field, replacement in (
        ("uuid", "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
        ("name", "NVIDIA A100-SXM4-80GB"),
        ("driver_version", "580.159.03"),
    ):
        model, simulator = _gpu_vulkan_contracts()
        simulator["nvidia_smi"][field] = replacement
        with pytest.raises(ValueError, match="NVIDIA identities differ"):
            evidence._validate_gpu_vulkan_runtime_agreement(model, simulator)


def test_run_metadata_binds_outer_package_probes_to_model_runtime(tmp_path):
    packages = {
        "required_versions": {"numpydantic": "1.6.9"},
        "import_generated_stub_seals": {"sealed_read_only_required": True},
    }
    digest = hashlib.sha256(evidence._canonical_json_bytes(packages)).hexdigest()
    host_runtime = {
        "packages": packages,
        "package_import_stability": {
            "preimport_sha256": digest,
            "postimport_sha256": digest,
            "unchanged": True,
        },
    }
    warning_filter = evidence.PI05_DROID_JOINTPOS_NUMPYDANTIC_WARNING_FILTER
    quoted_warning_filter = warning_filter.replace(" ", "\\ ")
    metadata = tmp_path / "run_metadata.env"
    source_contracts = {
        "source_tree_sha256": "1" * 64,
        "source_implementation_commit": "2" * 40,
        "source_approval_artifact_sha256": "3" * 64,
        "source_approval_path": str((tmp_path / "polaris_source_approval.json").resolve()),
        "trusted_source_hasher_sha256": "4" * 64,
        "consumer_binding_sha256": "5" * 64,
        "polaris_base_commit": "6" * 40,
        "source_openpi_commit": "7" * 40,
    }

    def write(*, pre=digest, post=digest, warning=quoted_warning_filter, **drift):
        source_values = {
            "POLARIS_SOURCE_TREE_SHA256": source_contracts["source_tree_sha256"],
            "POLARIS_IMPLEMENTATION_COMMIT": source_contracts[
                "source_implementation_commit"
            ],
            "SOURCE_APPROVAL_SHA256": source_contracts[
                "source_approval_artifact_sha256"
            ],
            "RUN_SOURCE_APPROVAL_FILE": source_contracts["source_approval_path"],
            "TRUSTED_SOURCE_HASHER_SHA256": source_contracts[
                "trusted_source_hasher_sha256"
            ],
            "CONSUMER_BINDING_SHA256": source_contracts[
                "consumer_binding_sha256"
            ],
            "FINAL_CONSUMER_BINDING_SHA256": source_contracts[
                "consumer_binding_sha256"
            ],
            "POLARIS_COMMIT": source_contracts["polaris_base_commit"],
            "OPENPI_COMMIT": source_contracts["source_openpi_commit"],
        }
        source_values.update(drift)
        metadata.write_text(
            f"PYTHONWARNINGS={warning}\n"
            f"PREFLIGHT_PACKAGE_ENVIRONMENT_SHA256={pre}\n"
            f"POSTRUN_PACKAGE_ENVIRONMENT_SHA256={post}\n"
            + "".join(f"{key}={value}\n" for key, value in source_values.items()),
            encoding="utf-8",
        )

    write()
    assert evidence._validate_package_run_metadata(
        metadata, host_runtime, source_contracts
    ) == {
        "openpi_package_environment_sha256": digest,
        "openpi_package_preflight_postrun_unchanged": True,
        "numpydantic_warning_filter": warning_filter,
    }

    write(post="0" * 64)
    with pytest.raises(ValueError, match="outer evaluation package probes"):
        evidence._validate_package_run_metadata(metadata, host_runtime, source_contracts)
    write(warning="error")
    with pytest.raises(ValueError, match="warning filter"):
        evidence._validate_package_run_metadata(metadata, host_runtime, source_contracts)
    for key in (
        "POLARIS_SOURCE_TREE_SHA256",
        "POLARIS_IMPLEMENTATION_COMMIT",
        "SOURCE_APPROVAL_SHA256",
        "RUN_SOURCE_APPROVAL_FILE",
        "TRUSTED_SOURCE_HASHER_SHA256",
        "CONSUMER_BINDING_SHA256",
        "FINAL_CONSUMER_BINDING_SHA256",
        "POLARIS_COMMIT",
        "OPENPI_COMMIT",
    ):
        write(**{key: "drift"})
        with pytest.raises(ValueError, match="source approval/binding"):
            evidence._validate_package_run_metadata(
                metadata, host_runtime, source_contracts
            )
    write()
    with metadata.open("a", encoding="utf-8") as stream:
        stream.write(f"POSTRUN_PACKAGE_ENVIRONMENT_SHA256={digest}\n")
    with pytest.raises(ValueError, match="duplicate run metadata"):
        evidence._validate_package_run_metadata(metadata, host_runtime, source_contracts)


def test_gpu_vulkan_contract_rejects_noncanonical_shared_runtime():
    model, simulator = _gpu_vulkan_contracts()
    model["jax"]["nvidia_smi"]["driver_version"] = "580.159.03"
    simulator["nvidia_smi"]["driver_version"] = "580.159.03"
    with pytest.raises(ValueError, match="canonical L40S driver"):
        evidence._validate_gpu_vulkan_runtime_agreement(model, simulator)

    model, simulator = _gpu_vulkan_contracts()
    simulator["vulkan"]["icd"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Vulkan ICD identity"):
        evidence._validate_gpu_vulkan_runtime_agreement(model, simulator)

    model, simulator = _gpu_vulkan_contracts()
    simulator["vulkan"]["vk_driver_files"] = "/tmp/nvidia_icd.json"
    with pytest.raises(ValueError, match="Vulkan ICD identity"):
        evidence._validate_gpu_vulkan_runtime_agreement(model, simulator)

    model, simulator = _gpu_vulkan_contracts()
    simulator["graphics_runtime"]["libraries"].pop()
    with pytest.raises(ValueError, match="mapped graphics-runtime identity"):
        evidence._validate_gpu_vulkan_runtime_agreement(model, simulator)

    model, simulator = _gpu_vulkan_contracts()
    simulator["graphics_runtime"]["cv2_loader"]["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mapped graphics-runtime identity"):
        evidence._validate_gpu_vulkan_runtime_agreement(model, simulator)

    model, simulator = _gpu_vulkan_contracts()
    simulator["graphics_runtime"]["cv2_loader"]["loader_search_safety"][
        "working_directory_read_only"
    ] = False
    with pytest.raises(ValueError, match="mapped graphics-runtime identity"):
        evidence._validate_gpu_vulkan_runtime_agreement(model, simulator)


def test_worker_rechecks_gpu_vulkan_runtime_before_evidence():
    source = (
        Path(__file__).parents[1]
        / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
    ).read_text()
    post_run = source.rindex("capture_gpu_runtime")
    evaluator_success = source.index("printf 'EVALUATOR_EXIT_CODE=0\\n'")
    finalize = source.index("-m polaris.pi05_droid_jointpos_evidence")
    assert post_run < evaluator_success < finalize
    assert 'EXPECTED_NVIDIA_DRIVER_VERSION="580.105.08"' in source
    assert 'EXPECTED_VULKAN_ICD_SHA256="7bdb6f27' in source
    assert "/usr/bin/env -i" in source


def test_finalizer_rejects_persisted_summary_that_differs_from_sealed_reaudit(
    tmp_path, monkeypatch
):
    run_dir, task_dir, _paths, _videos, _terminal_images = _write_inputs(
        tmp_path, rollouts=1
    )
    monkeypatch.setattr(evidence, "_validate_specialized_contracts", _stub_contracts)

    def mismatched_audit(paths, **_kwargs):
        value = json.loads(paths["trace_summary"].read_text())
        value["query_records"] -= 1
        return value

    monkeypatch.setattr(evidence, "_independently_audit_sealed_trace", mismatched_audit)
    with pytest.raises(ValueError, match="sealed trace/CSV re-audit differs"):
        evidence.finalize_evidence(
            run_dir=run_dir,
            task_dir=task_dir,
            environment="DROID-FoodBussing",
            expected_environment_seed=0,
            expected_rollouts=1,
            polaris_commit=COMMIT,
        )
    assert not (run_dir / evidence.PI05_DROID_JOINTPOS_EVIDENCE_MANIFEST).exists()


def test_worker_dry_run_cannot_publish_normal_success_and_markers_are_atomic():
    source = (
        Path(__file__).parents[1]
        / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
    ).read_text()
    dry_branch = source[
        source.index('if (( final_code == 0 )) && [[ "${DRY_RUN}" == 1 ]]') :
    ]
    dry_branch = dry_branch[: dry_branch.index("elif (( final_code == 0 ))")]
    assert 'publish_terminal_marker "${RUN_DIR}/DRY_RUN"' in dry_branch
    assert '"${RUN_DIR}/SUCCESS"' not in dry_branch
    assert '[[ "${EVIDENCE_FINALIZED}" == 1 ]]' in source
    assert "os.O_CREAT | os.O_EXCL" in source
    assert "os.link(temporary, destination" in source
    assert "os.fsync(directory)" in source
    assert "marker_code=$?" in source
