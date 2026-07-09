from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import pytest

from polaris import pi05_droid_jointpos_consumer_binding as binding


def _md5(payload: bytes) -> str:
    return base64.b64encode(
        hashlib.md5(payload, usedforsecurity=False).digest()
    ).decode("ascii")


@pytest.fixture
def consumer_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    checkpoint = tmp_path / "checkpoint"
    objects = {
        "_CHECKPOINT_METADATA": b"metadata\n",
        "assets/droid/norm_stats.json": b'{"norm_stats":{}}\n',
        "params/d/one": b"first parameter object",
        "params/d/two": b"second parameter object",
    }
    for relative, payload in objects.items():
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "".join(
            f"{binding.CHECKPOINT_MANIFEST_PREFIX}{relative}\t{len(payload)}\t{_md5(payload)}\n"
            for relative, payload in objects.items()
        ),
        encoding="ascii",
    )
    monkeypatch.setattr(
        binding,
        "CHECKPOINT_MANIFEST_SHA256",
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(binding, "CHECKPOINT_OBJECT_COUNT", len(objects))
    monkeypatch.setattr(binding, "CHECKPOINT_TOTAL_BYTES", sum(map(len, objects.values())))

    tokenizer = tmp_path / "paligemma_tokenizer.model"
    tokenizer_payload = b"small exact tokenizer"
    tokenizer.write_bytes(tokenizer_payload)
    monkeypatch.setattr(binding, "TOKENIZER_SIZE", len(tokenizer_payload))
    monkeypatch.setattr(binding, "TOKENIZER_MD5_BASE64", _md5(tokenizer_payload))
    monkeypatch.setattr(
        binding, "TOKENIZER_SHA256", hashlib.sha256(tokenizer_payload).hexdigest()
    )

    source = tmp_path / "source"
    required = ("scripts/eval.py", "src/polaris/runtime.py", "openpi/client.py")
    for index, relative in enumerate(required):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source {index}\n", encoding="utf-8")
    monkeypatch.setattr(binding, "_SOURCE_REQUIRED_PATHS", required)
    source_digest = binding.source_tree_sha256(source)
    return {
        "checkpoint": checkpoint,
        "manifest": manifest,
        "tokenizer": tokenizer,
        "source": source,
        "source_digest": source_digest,
        "objects": objects,
    }


def _open(inputs: dict[str, object]) -> binding.ConsumerBinding:
    return binding.open_consumer_binding(
        checkpoint=inputs["checkpoint"],
        manifest=inputs["manifest"],
        tokenizer=inputs["tokenizer"],
        source=inputs["source"],
        expected_source_tree_sha256=inputs["source_digest"],
    )


def _write_source_approval(
    path: Path, *, source: Path, tree_sha256: str, implementation: str = "a" * 40
) -> dict[str, object]:
    value = {
        "schema_version": 1,
        "profile": binding.SOURCE_APPROVAL_PROFILE,
        "snapshot_path": str(source.resolve(strict=True)),
        "source_tree_sha256": tree_sha256,
        "implementation_commit": implementation,
        "polaris_base_commit": binding.POLARIS_COMMIT,
        "polaris_base_tree": binding.POLARIS_TREE,
        "openpi_commit": binding.OPENPI_COMMIT,
        "trusted_hasher_sha256": binding.SOURCE_APPROVAL_TRUSTED_HASHER_SHA256,
    }
    path.write_bytes(binding.canonical_json_bytes(value) + b"\n")
    path.chmod(0o444)
    return binding.validate_persisted_source_approval(path)


def test_three_phase_binding_is_identical_and_persisted(
    consumer_inputs: dict[str, object], tmp_path: Path
) -> None:
    with _open(consumer_inputs) as opened:
        reports = [opened.snapshot(stage) for stage in binding.CONSUMER_BINDING_STAGES]
    assert [report["stage"] for report in reports] == list(
        binding.CONSUMER_BINDING_STAGES
    )
    assert len({report["binding_sha256"] for report in reports}) == 1
    artifacts = []
    for report in reports:
        artifact = binding.publish_consumer_binding(
            tmp_path / f"{report['stage']}.json", report
        )
        assert artifact["mode"] == "0444"
        assert artifact["nlink"] == 1
        artifacts.append(artifact)
    compared = binding.compare_consumer_binding_reports(
        *(artifact["value"] for artifact in artifacts)
    )
    assert compared["stages"] == list(binding.CONSUMER_BINDING_STAGES)


def test_evidence_validates_real_three_phase_consumer_artifacts(
    consumer_inputs: dict[str, object], tmp_path: Path
) -> None:
    from polaris import pi05_droid_jointpos_evidence as evidence

    paths = {
        "consumer_binding_preload": tmp_path / "pi05_consumer_binding_preload.json",
        "consumer_binding_postload": tmp_path / "pi05_consumer_binding_postload.json",
        "consumer_binding_postrun": tmp_path / "pi05_consumer_binding_postrun.json",
        "run_tokenizer": tmp_path / "consumer_inputs/paligemma_tokenizer.model",
        "source_approval": tmp_path / "polaris_source_approval.json",
    }
    source_tokenizer = consumer_inputs["tokenizer"]
    assert isinstance(source_tokenizer, Path)
    binding.prepare_run_tokenizer(source_tokenizer, paths["run_tokenizer"])
    artifacts = []
    with _open({**consumer_inputs, "tokenizer": paths["run_tokenizer"]}) as opened:
        for stage, key in zip(
            binding.CONSUMER_BINDING_STAGES,
            (
                "consumer_binding_preload",
                "consumer_binding_postload",
                "consumer_binding_postrun",
            ),
            strict=True,
        ):
            artifacts.append(
                binding.publish_consumer_binding(paths[key], opened.snapshot(stage))
            )
    summary = binding.compare_consumer_binding_reports(
        *(artifact["value"] for artifact in artifacts)
    )
    source = consumer_inputs["source"]
    assert isinstance(source, Path)
    source_approval = _write_source_approval(
        paths["source_approval"],
        source=source,
        tree_sha256=str(consumer_inputs["source_digest"]),
    )
    model_consumer = {
        **summary,
        "preload_filename": paths["consumer_binding_preload"].name,
        "preload_artifact_sha256": artifacts[0]["sha256"],
        "postload_filename": paths["consumer_binding_postload"].name,
        "postload_artifact_sha256": artifacts[1]["sha256"],
        "postrun_filename": paths["consumer_binding_postrun"].name,
        "source_approval_filename": paths["source_approval"].name,
        "source_approval_artifact_sha256": source_approval["sha256"],
        "implementation_commit": "a" * 40,
        "trusted_source_hasher_sha256": (
            binding.SOURCE_APPROVAL_TRUSTED_HASHER_SHA256
        ),
        "model_parameter_readiness": {
            "ready_leaf_count": 1,
            "total_elements": 1,
        },
    }
    incomplete_paths = {**paths}
    incomplete_paths.pop("source_approval")
    with pytest.raises(ValueError, match="evidence paths are incomplete"):
        evidence._validate_consumer_binding_contracts(
            incomplete_paths, {"consumer_binding": model_consumer}
        )
    validated = evidence._validate_consumer_binding_contracts(
        paths, {"consumer_binding": model_consumer}
    )
    assert validated["consumer_binding_sha256"] == summary["binding_sha256"]
    assert validated["source_approval_artifact_sha256"] == source_approval["sha256"]
    swapped_path = tmp_path / "swapped-source-approval.json"
    swapped_approval = _write_source_approval(
        swapped_path,
        source=source,
        tree_sha256="9" * 64,
    )
    paths["source_approval"] = swapped_path
    model_consumer["source_approval_filename"] = swapped_path.name
    model_consumer["source_approval_artifact_sha256"] = swapped_approval["sha256"]
    with pytest.raises(ValueError, match="source approval differs"):
        evidence._validate_consumer_binding_contracts(
            paths, {"consumer_binding": model_consumer}
        )
    paths["source_approval"] = tmp_path / "polaris_source_approval.json"
    model_consumer["source_approval_filename"] = paths["source_approval"].name
    model_consumer["source_approval_artifact_sha256"] = source_approval["sha256"]
    model_consumer["source_tree_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from lifecycle artifacts"):
        evidence._validate_consumer_binding_contracts(
            paths, {"consumer_binding": model_consumer}
        )


def test_checkpoint_path_replacement_is_rejected(
    consumer_inputs: dict[str, object]
) -> None:
    checkpoint = consumer_inputs["checkpoint"]
    assert isinstance(checkpoint, Path)
    target = checkpoint / "params/d/one"
    with _open(consumer_inputs) as opened:
        original = target.read_bytes()
        target.rename(target.with_suffix(".old"))
        target.write_bytes(original)
        with pytest.raises(
            binding.ConsumerBindingError,
            match="inode changed|object changed|closure changed",
        ):
            opened.snapshot("post_load")


def test_checkpoint_in_place_mutation_is_rejected(
    consumer_inputs: dict[str, object]
) -> None:
    checkpoint = consumer_inputs["checkpoint"]
    assert isinstance(checkpoint, Path)
    target = checkpoint / "params/d/two"
    with _open(consumer_inputs) as opened:
        target.write_bytes(b"X" * target.stat().st_size)
        with pytest.raises(binding.ConsumerBindingError, match="checkpoint object changed"):
            opened.snapshot("post_load")


def test_post_open_extra_checkpoint_or_source_entry_is_rejected(
    consumer_inputs: dict[str, object]
) -> None:
    checkpoint = consumer_inputs["checkpoint"]
    source = consumer_inputs["source"]
    assert isinstance(checkpoint, Path)
    assert isinstance(source, Path)
    with _open(consumer_inputs) as opened:
        (checkpoint / "params/new-object").write_bytes(b"unexpected")
        with pytest.raises(binding.ConsumerBindingError, match="checkpoint closure changed"):
            opened.snapshot("post_load")
    (checkpoint / "params/new-object").unlink()
    with _open(consumer_inputs) as opened:
        (source / "unexpected.py").write_text("unexpected\n")
        with pytest.raises(
            binding.ConsumerBindingError,
            match="source (root inode|closure) changed",
        ):
            opened.snapshot("postrun")


def test_checkpoint_root_swap_is_rejected(
    consumer_inputs: dict[str, object]
) -> None:
    checkpoint = consumer_inputs["checkpoint"]
    assert isinstance(checkpoint, Path)
    moved = checkpoint.with_name("checkpoint-old")
    with _open(consumer_inputs) as opened:
        checkpoint.rename(moved)
        checkpoint.mkdir()
        with pytest.raises(binding.ConsumerBindingError, match="root inode changed"):
            opened.snapshot("post_load")


def test_root_with_symlinked_ancestor_is_rejected(
    consumer_inputs: dict[str, object], tmp_path: Path
) -> None:
    checkpoint = consumer_inputs["checkpoint"]
    assert isinstance(checkpoint, Path)
    ancestor_alias = tmp_path / "ancestor-alias"
    ancestor_alias.symlink_to(checkpoint.parent, target_is_directory=True)
    aliased_checkpoint = ancestor_alias / checkpoint.name
    with pytest.raises(binding.ConsumerBindingError, match="unsafe.*path component"):
        binding._open_root(aliased_checkpoint)


def test_resolved_physical_path_is_accepted_after_alias_rejection(
    consumer_inputs: dict[str, object], tmp_path: Path
) -> None:
    checkpoint = consumer_inputs["checkpoint"]
    assert isinstance(checkpoint, Path)
    first_alias = tmp_path / "fsw"
    physical = tmp_path / "fs11/projects/nvr/users/lzha"
    physical.mkdir(parents=True)
    aliased_checkpoint = physical / "checkpoint"
    checkpoint.rename(aliased_checkpoint)
    first_alias.symlink_to(tmp_path / "fs11", target_is_directory=True)
    governed_alias = first_alias / "projects/nvr/users/lzha/checkpoint"
    with pytest.raises(binding.ConsumerBindingError, match="unsafe.*path component"):
        binding._open_root(governed_alias)
    canonical = governed_alias.resolve(strict=True)
    opened_path, descriptor, _identity = binding._open_root(canonical)
    try:
        assert opened_path == aliased_checkpoint
    finally:
        os.close(descriptor)


@pytest.mark.skipif(
    not Path("/lustre/fsw/portfolios/nvr/users/lzha").exists(),
    reason="live l401 Lustre aliases are unavailable",
)
def test_live_l401_alias_resolves_to_strict_physical_root() -> None:
    alias = Path("/lustre/fsw/portfolios/nvr/users/lzha")
    canonical = alias.resolve(strict=True)
    assert canonical == Path(
        "/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha"
    )
    with pytest.raises(binding.ConsumerBindingError, match="unsafe.*path component"):
        binding._open_root(alias)
    opened_path, descriptor, _identity = binding._open_root(canonical)
    try:
        assert opened_path == canonical
    finally:
        os.close(descriptor)


def test_source_approval_parser_rejects_schema_content_and_filesystem_drift(
    consumer_inputs: dict[str, object], tmp_path: Path
) -> None:
    source = consumer_inputs["source"]
    assert isinstance(source, Path)
    approval = tmp_path / "approval.json"
    artifact = _write_source_approval(
        approval,
        source=source,
        tree_sha256=str(consumer_inputs["source_digest"]),
    )
    assert artifact["value"]["snapshot_path"] == str(source.resolve())

    approval.chmod(0o644)
    with pytest.raises(binding.ConsumerBindingError, match="mode-0444"):
        binding.validate_persisted_source_approval(approval)
    approval.chmod(0o444)
    hardlink = tmp_path / "approval-hardlink.json"
    os.link(approval, hardlink)
    with pytest.raises(binding.ConsumerBindingError, match="mode-0444 link"):
        binding.validate_persisted_source_approval(approval)
    hardlink.unlink()
    symlink = tmp_path / "approval-symlink.json"
    symlink.symlink_to(approval)
    with pytest.raises(binding.ConsumerBindingError, match="must not be a symlink"):
        binding.validate_persisted_source_approval(symlink)

    value = artifact["value"]
    for field, replacement in (
        ("source_tree_sha256", "0" * 63),
        ("implementation_commit", "0" * 39),
        ("polaris_base_commit", "0" * 40),
        ("polaris_base_tree", "0" * 40),
        ("openpi_commit", "0" * 40),
        ("trusted_hasher_sha256", "0" * 64),
    ):
        drifted = {**value, field: replacement}
        candidate = tmp_path / f"drift-{field}.json"
        candidate.write_bytes(binding.canonical_json_bytes(drifted) + b"\n")
        candidate.chmod(0o444)
        with pytest.raises(binding.ConsumerBindingError, match="identity mismatch"):
            binding.validate_persisted_source_approval(candidate)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(
        b'{"schema_version":1,"schema_version":1}\n'
    )
    duplicate.chmod(0o444)
    with pytest.raises(binding.ConsumerBindingError, match="strict JSON"):
        binding.validate_persisted_source_approval(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_bytes(b'{"schema_version":NaN}\n')
    nonfinite.chmod(0o444)
    with pytest.raises(binding.ConsumerBindingError, match="strict JSON"):
        binding.validate_persisted_source_approval(nonfinite)
    extra = tmp_path / "extra.json"
    extra.write_bytes(
        binding.canonical_json_bytes({**value, "unexpected": True}) + b"\n"
    )
    extra.chmod(0o444)
    with pytest.raises(binding.ConsumerBindingError, match="schema mismatch"):
        binding.validate_persisted_source_approval(extra)
    source_alias_parent = tmp_path / "source-alias-parent"
    source_alias_parent.symlink_to(source.parent, target_is_directory=True)
    alias_value = {
        **value,
        "snapshot_path": str(source_alias_parent / source.name),
    }
    alias_approval = tmp_path / "alias-approval.json"
    alias_approval.write_bytes(binding.canonical_json_bytes(alias_value) + b"\n")
    alias_approval.chmod(0o444)
    with pytest.raises(binding.ConsumerBindingError, match="not canonical"):
        binding.validate_persisted_source_approval(alias_approval)
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_text(json.dumps(value, indent=2) + "\n")
    noncanonical.chmod(0o444)
    with pytest.raises(binding.ConsumerBindingError, match="canonical JSON"):
        binding.validate_persisted_source_approval(noncanonical)


def test_checkpoint_extra_file_and_symlink_fail_closed(
    consumer_inputs: dict[str, object]
) -> None:
    checkpoint = consumer_inputs["checkpoint"]
    assert isinstance(checkpoint, Path)
    extra = checkpoint / "params/extra"
    extra.write_bytes(b"extra")
    with pytest.raises(binding.ConsumerBindingError, match="checkpoint closure mismatch"):
        _open(consumer_inputs)
    extra.unlink()
    (checkpoint / "params/link").symlink_to("d")
    with pytest.raises(binding.ConsumerBindingError, match="symlink|non-directory"):
        _open(consumer_inputs)


def test_tokenizer_replacement_is_rejected(
    consumer_inputs: dict[str, object]
) -> None:
    tokenizer = consumer_inputs["tokenizer"]
    assert isinstance(tokenizer, Path)
    with _open(consumer_inputs) as opened:
        payload = tokenizer.read_bytes()
        tokenizer.rename(tokenizer.with_suffix(".old"))
        tokenizer.write_bytes(payload)
        with pytest.raises(binding.ConsumerBindingError, match="tokenizer identity changed"):
            opened.snapshot("post_load")


def test_source_replacement_is_rejected(
    consumer_inputs: dict[str, object]
) -> None:
    source = consumer_inputs["source"]
    assert isinstance(source, Path)
    target = source / "scripts/eval.py"
    with _open(consumer_inputs) as opened:
        payload = target.read_bytes()
        target.rename(target.with_suffix(".old"))
        target.write_bytes(payload)
        with pytest.raises(
            binding.ConsumerBindingError,
            match="source (file|closure) changed",
        ):
            opened.snapshot("postrun")


def test_run_local_tokenizer_copy_is_exact_and_single_link(
    consumer_inputs: dict[str, object], tmp_path: Path
) -> None:
    tokenizer = consumer_inputs["tokenizer"]
    assert isinstance(tokenizer, Path)
    destination = tmp_path / "run/consumer_inputs/paligemma_tokenizer.model"
    report = binding.prepare_run_tokenizer(tokenizer, destination)
    assert report["sha256"] == binding.TOKENIZER_SHA256
    identity = os.stat(destination, follow_symlinks=False)
    assert identity.st_nlink == 1
    assert identity.st_mode & 0o777 == 0o444


def test_stage_or_artifact_drift_is_rejected(
    consumer_inputs: dict[str, object], tmp_path: Path
) -> None:
    with _open(consumer_inputs) as opened:
        preload = opened.snapshot("pre_load")
        postload = opened.snapshot("post_load")
    postload["identity"]["checkpoint"]["root"] += "-drift"
    with pytest.raises(binding.ConsumerBindingError, match="identity mismatch"):
        binding.validate_consumer_binding_report(postload)
    artifact = binding.publish_consumer_binding(tmp_path / "preload.json", preload)
    Path(artifact["path"]).chmod(0o644)
    with pytest.raises(binding.ConsumerBindingError, match="mode-0444"):
        binding.validate_persisted_consumer_binding(Path(artifact["path"]))


def test_report_rejects_recomputed_forged_closure_and_out_of_order_stages(
    consumer_inputs: dict[str, object],
) -> None:
    with _open(consumer_inputs) as opened:
        preload = opened.snapshot("pre_load")
        postload = opened.snapshot("post_load")
    forged = binding.json.loads(binding.json.dumps(preload))
    forged["identity"]["checkpoint"]["objects"][0]["sha256"] = "f" * 64
    forged["binding_sha256"] = binding._sha256(forged["identity"])
    with pytest.raises(binding.ConsumerBindingError, match="closure mismatch"):
        binding.validate_consumer_binding_report(forged)
    with pytest.raises(binding.ConsumerBindingError, match="out of order"):
        binding.compare_consumer_binding_reports(postload, preload)


def test_failed_binding_constructors_close_descriptors(
    consumer_inputs: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor_root = Path("/proc/self/fd")
    if not descriptor_root.is_dir():
        pytest.skip("descriptor inventory requires procfs")
    checkpoint = consumer_inputs["checkpoint"]
    manifest = consumer_inputs["manifest"]
    tokenizer = consumer_inputs["tokenizer"]
    source = consumer_inputs["source"]
    assert isinstance(checkpoint, Path)
    assert isinstance(manifest, Path)
    assert isinstance(tokenizer, Path)
    assert isinstance(source, Path)
    before = len(list(descriptor_root.iterdir()))
    monkeypatch.setattr(binding, "CHECKPOINT_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(binding.ConsumerBindingError, match="manifest SHA-256"):
        binding.CheckpointBinding(checkpoint, manifest)
    missing = tokenizer.with_name("missing-tokenizer.model")
    with pytest.raises(FileNotFoundError):
        binding.TokenizerBinding(missing)
    monkeypatch.setattr(binding, "_SOURCE_REQUIRED_PATHS", ("missing.py",))
    with pytest.raises(binding.ConsumerBindingError, match="lacks runtime paths"):
        binding.SourceBinding(source, str(consumer_inputs["source_digest"]))
    assert len(list(descriptor_root.iterdir())) == before


def test_checkpoint_is_not_copied_by_binding_implementation() -> None:
    source = Path(binding.__file__).read_text(encoding="utf-8")
    checkpoint_class = source[
        source.index("class CheckpointBinding") : source.index("class TokenizerBinding")
    ]
    assert "copyfile" not in checkpoint_class
    assert "copytree" not in checkpoint_class
    assert "shutil" not in checkpoint_class


def test_server_orders_binding_around_unchanged_official_restore() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/polaris/serve_pi05_droid_jointpos_attested.py"
    ).read_text(encoding="utf-8")
    preload = source.index('consumer_binding.snapshot("pre_load")')
    approval = source.index("validate_persisted_source_approval(")
    restore = source.index("policy = policy_config.create_trained_policy(")
    ready = source.index("_block_loaded_model_until_ready(policy, jax)")
    postload = source.index('consumer_binding.snapshot("post_load")')
    publication = source.index("publish_pi05_droid_jointpos_serving_contract(")
    listener = source.index("server = websocket_policy_server.WebsocketPolicyServer(")
    postrun = source.index('consumer_binding.snapshot("postrun")')
    rng_read = source.index("final_key_data = jax.random.key_data(policy._rng)")
    assert approval < preload < restore < ready < postload < publication < listener
    assert postrun < rng_read
    exact_restore = """policy = policy_config.create_trained_policy(
        train_config, args.checkpoint_dir.resolve()
    )"""
    assert exact_restore in source


def test_wrapper_preserves_scientific_eval_vector_and_ro_source_mount() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh"
    ).read_text(encoding="utf-8")
    expected = """eval_args=(
  scripts/eval.py
  --environment "${POLARIS_ENVIRONMENT}"
  --control-mode joint-position
  --policy.client DroidJointPos
  --policy.host 127.0.0.1
  --policy.port "${PORT}"
  --policy.open-loop-horizon "${OPEN_LOOP_HORIZON}"
  --policy.frame-description "robot base frame"
  --policy.action-frame robot_base
  --policy.dataset-name droid
  --policy.no-rotate-wrist-180
  --policy.no-render-every-step
  --policy.state-type joint_position
  --policy.expected-action-horizon "${EXPECTED_ACTION_HORIZON}"
  --policy.expected-action-dim "${EXPECTED_ACTION_DIM}"
  --policy.trace-path "${TRACE_PATH}"
  --run-folder "${TASK_DIR}"
  --rollouts "${ROLLOUTS}"
  --environment-seed "${ENVIRONMENT_SEED}"
  --runtime-contract-path "${RUNTIME_CONTRACT_FILE}"
  --headless
)"""
    assert expected in source
    assert "${POLARIS_DIR}:${POLARIS_CONTAINER_SOURCE}:ro" in source
    assert '"--container-workdir=${POLARIS_CONTAINER_SOURCE}"' in source
    assert source.index("pi05_consumer_binding_postload.json") < source.index(
        '"${eval_command[@]}"'
    )


def test_model_readiness_waits_for_every_array_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    from types import SimpleNamespace

    root = Path(__file__).resolve().parents[1]
    server_path = root / "scripts/polaris/serve_pi05_droid_jointpos_attested.py"
    specification = importlib.util.spec_from_file_location(
        "consumer_binding_server_test", server_path
    )
    assert specification is not None and specification.loader is not None
    server = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(server)

    class Leaf:
        size = 7

        def __init__(self) -> None:
            self.calls = 0

        def block_until_ready(self) -> None:
            self.calls += 1

    leaves = [Leaf(), Leaf()]
    from flax import nnx

    monkeypatch.setattr(nnx, "state", lambda _model: object())
    fake_jax = SimpleNamespace(
        tree_util=SimpleNamespace(tree_leaves=lambda _state: leaves)
    )
    result = server._block_loaded_model_until_ready(
        SimpleNamespace(_model=object()), fake_jax
    )
    assert result == {"ready_leaf_count": 2, "total_elements": 14}
    assert [leaf.calls for leaf in leaves] == [1, 1]
