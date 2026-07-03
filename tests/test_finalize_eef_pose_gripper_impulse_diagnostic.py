from __future__ import annotations

from argparse import Namespace
import copy
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from scripts import finalize_eef_pose_gripper_impulse_diagnostic as finalizer
from scripts import smoke_eef_pose_gripper_impulse_diagnostic as diagnostic_module


finalizer.diagnostic = diagnostic_module


def _write_immutable(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    path.chmod(0o444)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_submodule_status() -> str:
    return "\n".join(
        f"-{commit} {path}"
        for path, commit in finalizer.EXPECTED_UNINITIALIZED_SUBMODULE_GITLINKS
    )


def test_secure_reader_accepts_exact_immutable_single_link(tmp_path: Path):
    path = _write_immutable(tmp_path / "evidence", b"evidence\n")
    identity, data = finalizer._secure_read_immutable(path, field="evidence")
    assert data == b"evidence\n"
    assert identity == {
        "path": str(path.absolute()),
        "size_bytes": 9,
        "sha256": hashlib.sha256(b"evidence\n").hexdigest(),
        "mode": "0444",
        "nlink": 1,
    }


def test_secure_reader_rejects_missing_writable_symlink_and_multilink(tmp_path: Path):
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="missing"):
        finalizer._secure_read_immutable(tmp_path / "missing", field="evidence")

    writable = tmp_path / "writable"
    writable.write_bytes(b"x")
    writable.chmod(0o644)
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="mode"):
        finalizer._secure_read_immutable(writable, field="evidence")

    target = _write_immutable(tmp_path / "target", b"x")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(target)
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="regular"):
        finalizer._secure_read_immutable(symlink, field="evidence")

    first = _write_immutable(tmp_path / "first", b"x")
    second = tmp_path / "second"
    os.link(first, second)
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="link count"):
        finalizer._secure_read_immutable(first, field="evidence")
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="link count"):
        finalizer._secure_read_immutable(second, field="evidence")


@pytest.mark.parametrize("data", [b"1\n", b"0", b"0\n\n", b" 0\n", b""])
def test_zero_status_requires_exact_bytes(tmp_path: Path, data: bytes):
    path = _write_immutable(tmp_path / "status", data)
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="exact bytes"):
        finalizer._require_zero_status(path, field="status")


def _slurm_args(tmp_path: Path) -> Namespace:
    args = Namespace(
        expected_slurm_job_id=123,
        expected_slurm_node_list="pool0-00001",
        expected_slurm_node_name="pool0-00001",
        expected_slurm_account="nvr_lpr_rvp",
        expected_slurm_partition="batch",
        expected_slurm_job_name="impulse",
        expected_slurm_num_nodes=1,
        expected_slurm_num_cpus=8,
        expected_slurm_num_tasks=1,
        expected_slurm_req_tres="cpu=8,mem=64G,node=1,billing=1,gres/gpu=1",
        expected_slurm_alloc_tres="cpu=8,mem=64G,node=1,billing=1,gres/gpu=1",
        expected_slurm_tres_per_node="gres/gpu:1",
        expected_slurm_gpus_on_node="1",
        expected_slurm_cpus_per_task=8,
        expected_slurm_output=tmp_path / "slurm.out",
        expected_slurm_command=tmp_path / "submitted-saved.sh",
        expected_slurm_work_dir=tmp_path,
        submitted_saved_wrapper=tmp_path / "submitted-saved.sh",
        slurm_job_oneliner_snapshot=tmp_path / "scontrol-show-job.txt",
    )
    _write_slurm_snapshot(args)
    return args


def _slurm_record(args: Namespace) -> dict[str, str]:
    return {
        "JobId": str(args.expected_slurm_job_id),
        "NodeList": args.expected_slurm_node_list,
        "Account": args.expected_slurm_account,
        "Partition": args.expected_slurm_partition,
        "JobName": args.expected_slurm_job_name,
        "NumNodes": str(args.expected_slurm_num_nodes),
        "NumCPUs": str(args.expected_slurm_num_cpus),
        "NumTasks": str(args.expected_slurm_num_tasks),
        "ReqTRES": args.expected_slurm_req_tres,
        "AllocTRES": args.expected_slurm_alloc_tres,
        "TresPerNode": args.expected_slurm_tres_per_node,
        "CPUs/Task": str(args.expected_slurm_cpus_per_task),
        "StdOut": str(args.expected_slurm_output.absolute()),
        "Command": str(args.expected_slurm_command.absolute()),
        "WorkDir": str(args.expected_slurm_work_dir.absolute()),
        "BatchHost": args.expected_slurm_node_name,
        "JobState": "RUNNING",
        "BatchFlag": "1",
    }


def _write_slurm_snapshot(
    args: Namespace, *, overrides: dict[str, str] | None = None, suffix: str = ""
) -> Path:
    record = _slurm_record(args)
    record.update(overrides or {})
    if suffix:
        args.slurm_job_oneliner_snapshot = (
            args.slurm_job_oneliner_snapshot.parent / f"scontrol-show-job-{suffix}.txt"
        )
    data = (
        " ".join(f"{key}={value}" for key, value in record.items()) + " \n"
    ).encode()
    return _write_immutable(args.slurm_job_oneliner_snapshot, data)


def _set_slurm_environment(monkeypatch, args: Namespace) -> None:
    values = {
        "SLURM_JOB_ID": str(args.expected_slurm_job_id),
        "SLURM_JOB_NODELIST": args.expected_slurm_node_list,
        "SLURMD_NODENAME": args.expected_slurm_node_name,
        "SLURM_JOB_ACCOUNT": args.expected_slurm_account,
        "SLURM_JOB_PARTITION": args.expected_slurm_partition,
        "SLURM_JOB_NAME": args.expected_slurm_job_name,
        "SLURM_NTASKS": str(args.expected_slurm_num_tasks),
        "SLURM_GPUS_ON_NODE": args.expected_slurm_gpus_on_node,
        "SLURM_CPUS_PER_TASK": str(args.expected_slurm_cpus_per_task),
    }
    for field, value in values.items():
        monkeypatch.setenv(field, value)


def test_slurm_provenance_binds_immutable_snapshot_and_environment(
    tmp_path, monkeypatch
):
    args = _slurm_args(tmp_path)
    _set_slurm_environment(monkeypatch, args)
    observed = finalizer._capture_slurm_provenance(args)
    assert observed["job_id"] == 123
    assert observed["node_name"] == "pool0-00001"
    assert observed["snapshot"]["mode"] == "0444"
    assert observed["requested_tres"]["entries"][0] == {
        "name": "billing",
        "kind": "integer",
        "value": 1,
    }
    assert next(
        entry
        for entry in observed["requested_tres"]["entries"]
        if entry["name"] == "mem"
    ) == {
        "name": "mem",
        "kind": "bytes_binary",
        "value": 64 * 1024**3,
    }
    assert observed["tres_per_node"]["entries"] == [
        {"resource": "gres/gpu", "type": None, "count": 1}
    ]
    assert observed["stdout_path"] == str((tmp_path / "slurm.out").absolute())
    assert not hasattr(finalizer, "_scontrol_job")


@pytest.mark.parametrize("snapshot_field", ["Account", "JobName"])
def test_step_local_account_and_job_name_are_snapshot_authoritative(
    tmp_path, monkeypatch, snapshot_field
):
    args = _slurm_args(tmp_path)
    _set_slurm_environment(monkeypatch, args)
    monkeypatch.delenv("SLURM_JOB_ACCOUNT", raising=False)
    monkeypatch.setenv("SLURM_JOB_NAME", "<bash>")
    observed = finalizer._capture_slurm_provenance(args)
    assert observed["account"] == args.expected_slurm_account
    assert observed["job_name"] == args.expected_slurm_job_name

    args.slurm_job_oneliner_snapshot.unlink()
    _write_slurm_snapshot(args, overrides={snapshot_field: "tampered"})
    with pytest.raises(
        finalizer.GripperImpulseFinalizationError,
        match=rf"snapshot {snapshot_field} mismatch",
    ):
        finalizer._capture_slurm_provenance(args)


def test_slurm_tres_comparison_is_order_independent(tmp_path, monkeypatch):
    args = _slurm_args(tmp_path)
    args.slurm_job_oneliner_snapshot.unlink()
    observed_req = "gres/gpu=1,billing=1,node=1,mem=64G,cpu=8"
    observed_alloc = "node=1,gres/gpu=1,cpu=8,billing=1,mem=64G"
    _write_slurm_snapshot(
        args,
        overrides={"ReqTRES": observed_req, "AllocTRES": observed_alloc},
    )
    _set_slurm_environment(monkeypatch, args)
    result = finalizer._capture_slurm_provenance(args)
    assert result["requested_tres"]["raw"] == observed_req
    assert result["allocated_tres"]["raw"] == observed_alloc


def test_tres_assignments_canonicalize_equivalent_binary_memory_units():
    left = finalizer._canonical_tres_assignments("mem=64G,cpu=8", field="left")
    right = finalizer._canonical_tres_assignments("cpu=8,mem=65536M", field="right")
    assert left["entries"] == right["entries"]


def test_tres_per_node_canonicalizes_typed_and_untyped_gpu_forms():
    assert finalizer._canonical_tres_per_node("gres/gpu:1", field="generic")[
        "entries"
    ] == [{"resource": "gres/gpu", "type": None, "count": 1}]
    assert finalizer._canonical_tres_per_node("gres/gpu:l40s:1", field="typed")[
        "entries"
    ] == [{"resource": "gres/gpu", "type": "l40s", "count": 1}]


@pytest.mark.parametrize(
    "field",
    [
        "JobId",
        "NodeList",
        "Account",
        "Partition",
        "JobName",
        "NumNodes",
        "NumCPUs",
        "NumTasks",
        "ReqTRES",
        "AllocTRES",
        "TresPerNode",
        "CPUs/Task",
        "StdOut",
        "Command",
        "WorkDir",
        "BatchHost",
        "JobState",
    ],
)
def test_slurm_provenance_rejects_every_snapshot_field(tmp_path, monkeypatch, field):
    args = _slurm_args(tmp_path)
    args.slurm_job_oneliner_snapshot.unlink()
    _set_slurm_environment(monkeypatch, args)
    _write_slurm_snapshot(args, overrides={field: "tampered"})
    with pytest.raises(finalizer.GripperImpulseFinalizationError):
        finalizer._capture_slurm_provenance(args)


@pytest.mark.parametrize(
    "field",
    [
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURMD_NODENAME",
        "SLURM_JOB_PARTITION",
        "SLURM_NTASKS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK",
    ],
)
def test_slurm_provenance_rejects_every_live_environment_field(
    tmp_path, monkeypatch, field
):
    args = _slurm_args(tmp_path)
    _set_slurm_environment(monkeypatch, args)
    monkeypatch.setenv(field, "tampered")
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match=field):
        finalizer._capture_slurm_provenance(args)


@pytest.mark.parametrize(
    "field",
    [
        "expected_slurm_job_id",
        "expected_slurm_num_nodes",
        "expected_slurm_num_cpus",
        "expected_slurm_num_tasks",
        "expected_slurm_cpus_per_task",
    ],
)
def test_slurm_integer_fields_reject_boolean_impersonation(tmp_path, field):
    args = _slurm_args(tmp_path)
    setattr(args, field, True)
    with pytest.raises(finalizer.GripperImpulseFinalizationError):
        finalizer._capture_slurm_provenance(args)


@pytest.mark.parametrize(
    ("value", "parser"),
    [
        ("cpu=8,cpu=9", finalizer._canonical_tres_assignments),
        ("gres/gpu:01", finalizer._canonical_tres_per_node),
        (True, finalizer._canonical_tres_assignments),
    ],
)
def test_tres_parsers_reject_duplicate_noncanonical_or_boolean(value, parser):
    with pytest.raises(finalizer.GripperImpulseFinalizationError):
        parser(value, field="probe")


@pytest.mark.parametrize(
    "data",
    [b"JobId=1", b"JobId=1\n\n", b"JobId=1\r\n", b"garbage JobId=1\n"],
)
def test_slurm_snapshot_parser_requires_one_exact_record(data):
    with pytest.raises(finalizer.GripperImpulseFinalizationError):
        finalizer._parse_scontrol_oneliner(data)


def test_submodule_status_attests_exact_pinned_uninitialized_gitlinks():
    validated = finalizer._validate_submodule_status(_expected_submodule_status())
    assert validated["profile"] == finalizer.SUBMODULE_STATUS_PROFILE
    assert [
        (entry["path"], entry["gitlink_commit"], entry["state"])
        for entry in validated["entries"]
    ] == [
        (path, commit, "uninitialized")
        for path, commit in finalizer.EXPECTED_UNINITIALIZED_SUBMODULE_GITLINKS
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda lines: [],
        lambda lines: [*lines, lines[0]],
        lambda lines: [f"+{lines[0][1:]}", *lines[1:]],
        lambda lines: [f"U{lines[0][1:]}", *lines[1:]],
        lambda lines: [f" {lines[0][1:]}", *lines[1:]],
        lambda lines: [f"-{'0' * 40} {lines[0].split(' ', 1)[1]}", *lines[1:]],
        lambda lines: [
            f"-{lines[0][1:41]} unknown/submodule",
            *lines[1:],
        ],
        lambda lines: [f"{lines[0]} (description)", *lines[1:]],
    ],
)
def test_submodule_status_rejects_state_commit_path_and_grammar_drift(mutation):
    lines = _expected_submodule_status().splitlines()
    with pytest.raises(finalizer.GripperImpulseFinalizationError):
        finalizer._validate_submodule_status("\n".join(mutation(lines)))


def test_standalone_repository_layout_requires_real_inroot_git_directory(
    monkeypatch, tmp_path
):
    repo = tmp_path / "polaris"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)

    def fake_git(_repo, *arguments):
        if arguments == ("rev-parse", "--absolute-git-dir"):
            return str(git_dir)
        if arguments == ("rev-parse", "--git-common-dir"):
            return ".git"
        raise AssertionError(arguments)

    monkeypatch.setattr(finalizer, "_git", fake_git)
    assert finalizer._validate_standalone_repository_layout(repo) == {  # noqa: SLF001
        "layout_profile": finalizer.REPOSITORY_LAYOUT_PROFILE,
        "git_dir": str(git_dir),
        "git_common_dir": str(git_dir),
    }


@pytest.mark.parametrize("kind", ["gitdir_file", "symlink"])
def test_standalone_repository_layout_rejects_linked_worktree_or_symlink(
    monkeypatch, tmp_path, kind
):
    repo = tmp_path / "polaris"
    repo.mkdir()
    external = tmp_path / "external.git"
    external.mkdir()
    if kind == "gitdir_file":
        (repo / ".git").write_text(f"gitdir: {external}\n")
    else:
        (repo / ".git").symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(
        finalizer,
        "_git",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("Git must not run for a rejected .git entry")
        ),
    )
    with pytest.raises(
        finalizer.GripperImpulseFinalizationError, match="real in-root directory"
    ):
        finalizer._validate_standalone_repository_layout(repo)  # noqa: SLF001


@pytest.mark.parametrize("external_field", ["git_dir", "git_common_dir"])
def test_standalone_repository_layout_rejects_external_git_metadata(
    monkeypatch, tmp_path, external_field
):
    repo = tmp_path / "polaris"
    git_dir = repo / ".git"
    git_dir.mkdir(parents=True)
    external = tmp_path / "external.git"

    def fake_git(_repo, *arguments):
        if arguments == ("rev-parse", "--absolute-git-dir"):
            return str(external if external_field == "git_dir" else git_dir)
        if arguments == ("rev-parse", "--git-common-dir"):
            return str(external) if external_field == "git_common_dir" else ".git"
        raise AssertionError(arguments)

    monkeypatch.setattr(finalizer, "_git", fake_git)
    with pytest.raises(
        finalizer.GripperImpulseFinalizationError, match="Git directory"
    ):
        finalizer._validate_standalone_repository_layout(repo)  # noqa: SLF001


def test_repository_contract_rejects_attached_branch_head(monkeypatch, tmp_path):
    repo = tmp_path / "polaris"
    repo.mkdir()
    commit = "a" * 40
    monkeypatch.setattr(
        finalizer,
        "_validate_standalone_repository_layout",
        lambda _repo: {
            "layout_profile": finalizer.REPOSITORY_LAYOUT_PROFILE,
            "git_dir": str(repo / ".git"),
            "git_common_dir": str(repo / ".git"),
        },
    )

    def fake_git(_repo, *arguments):
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repo)
        if arguments == ("rev-parse", "HEAD"):
            return commit
        if arguments == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "main"
        raise AssertionError(arguments)

    monkeypatch.setattr(finalizer, "_git", fake_git)
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="not detached"):
        finalizer._validate_repository(  # noqa: SLF001
            Namespace(polaris_repo=repo, expected_polaris_commit=commit)
        )


def test_source_preflight_cli_uses_same_standalone_repository_contract(
    monkeypatch, tmp_path, capsys
):
    repo = tmp_path / "polaris"
    commit = "a" * 40
    expected = {
        "path": str(repo),
        "top_level": str(repo),
        "commit": commit,
        "clean": True,
        "head_state": "detached",
        "layout_profile": finalizer.REPOSITORY_LAYOUT_PROFILE,
        "git_dir": str(repo / ".git"),
        "git_common_dir": str(repo / ".git"),
        "submodule_status": _expected_submodule_status(),
        "submodules": finalizer._validate_submodule_status(
            _expected_submodule_status()
        ),
    }
    monkeypatch.setattr(finalizer, "_validate_repository", lambda _args: expected)
    monkeypatch.setattr(
        finalizer,
        "_bootstrap_trusted_sources",
        lambda _args: (_ for _ in ()).throw(
            AssertionError("source preflight must run before diagnostic bootstrap")
        ),
    )
    assert (
        finalizer.main(
            [
                "--source-preflight",
                "--polaris-repo",
                str(repo),
                "--expected-polaris-commit",
                commit,
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == (
        "POLARIS_GRIPPER_IMPULSE_SOURCE_PREFLIGHT="
        f"profile={finalizer.REPOSITORY_LAYOUT_PROFILE};repo={repo};"
        f"git_dir={repo / '.git'};commit={commit}\n"
    )


def test_source_provenance_binds_import_repo_and_all_hashes(monkeypatch):
    diagnostic = diagnostic_module
    repo = finalizer.DIAGNOSTIC_PATH.parent.parent
    diagnostic_identity = diagnostic._file_identity(  # noqa: SLF001
        finalizer.DIAGNOSTIC_PATH
    )
    boundary_identity = diagnostic._file_identity(  # noqa: SLF001
        diagnostic.BOUNDARY_HELPER_PATH
    )
    fixture_identity, _ = diagnostic.boundary.load_replay_fixture()
    commit = "a" * 40
    args = Namespace(
        polaris_repo=repo,
        expected_polaris_commit=commit,
        expected_diagnostic_sha256=diagnostic_identity["sha256"],
        expected_finalizer_sha256=_sha256(finalizer.SCRIPT_PATH),
        expected_boundary_sha256=boundary_identity["sha256"],
        expected_fixture_sha256=fixture_identity["sha256"],
    )
    layout = {
        "layout_profile": finalizer.REPOSITORY_LAYOUT_PROFILE,
        "git_dir": str(repo / ".git"),
        "git_common_dir": str(repo / ".git"),
    }
    monkeypatch.setattr(
        finalizer,
        "_validate_standalone_repository_layout",
        lambda _repo: dict(layout),
    )

    def fake_git(_repo, *arguments):
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(repo)
        if arguments == ("rev-parse", "HEAD"):
            return commit
        if arguments == ("rev-parse", "--abbrev-ref", "HEAD"):
            return "HEAD"
        if arguments == ("submodule", "status", "--recursive"):
            return _expected_submodule_status()
        return ""

    monkeypatch.setattr(finalizer, "_git", fake_git)
    capture = {
        "diagnostic_source": {"actual": diagnostic_identity},
        "boundary_helper_source": boundary_identity,
        "fixture": fixture_identity,
    }
    sources, repository = finalizer._capture_source_provenance(args, capture)
    assert sources["imported_diagnostic_file"] == str(finalizer.DIAGNOSTIC_PATH)
    assert sources["boundary_helper"]["sha256"] == (
        diagnostic.EXPECTED_BOUNDARY_HELPER_SHA256
    )
    assert repository == {
        "path": str(repo),
        "top_level": str(repo),
        "commit": commit,
        "clean": True,
        "head_state": "detached",
        **layout,
        "submodule_status": _expected_submodule_status(),
        "submodules": {
            "profile": finalizer.SUBMODULE_STATUS_PROFILE,
            "entries": [
                {
                    "path": path,
                    "gitlink_commit": gitlink_commit,
                    "state": "uninitialized",
                }
                for path, gitlink_commit in (
                    finalizer.EXPECTED_UNINITIALIZED_SUBMODULE_GITLINKS
                )
            ],
        },
    }

    monkeypatch.setattr(
        finalizer,
        "_git",
        lambda _repo, *arguments: (
            commit
            if arguments == ("rev-parse", "HEAD")
            else (
                str(repo)
                if arguments == ("rev-parse", "--show-toplevel")
                else (
                    "HEAD"
                    if arguments == ("rev-parse", "--abbrev-ref", "HEAD")
                    else (
                        _expected_submodule_status()
                        if arguments == ("submodule", "status", "--recursive")
                        else ("dirty" if "status" in arguments else "")
                    )
                )
            )
        ),
    )
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="dirty"):
        finalizer._capture_source_provenance(args, capture)


def test_bootstrap_hashes_every_source_before_diagnostic_import(monkeypatch):
    repo = finalizer.DIAGNOSTIC_PATH.parent.parent
    args = Namespace(
        polaris_repo=repo,
        expected_polaris_commit="a" * 40,
        expected_diagnostic_sha256=_sha256(finalizer.DIAGNOSTIC_PATH),
        expected_finalizer_sha256=_sha256(finalizer.SCRIPT_PATH),
        expected_boundary_sha256=_sha256(finalizer.BOUNDARY_PATH),
        expected_fixture_sha256=_sha256(finalizer.FIXTURE_PATH),
    )
    monkeypatch.setattr(
        finalizer,
        "_validate_repository",
        lambda _args: {
            "path": str(repo),
            "top_level": str(repo),
            "commit": "a" * 40,
            "clean": True,
            "head_state": "detached",
            "layout_profile": finalizer.REPOSITORY_LAYOUT_PROFILE,
            "git_dir": str(repo / ".git"),
            "git_common_dir": str(repo / ".git"),
            "submodule_status": _expected_submodule_status(),
            "submodules": finalizer._validate_submodule_status(
                _expected_submodule_status()
            ),
        },
    )
    imported = finalizer._bootstrap_trusted_sources(args)
    assert Path(imported.__file__).resolve() == finalizer.DIAGNOSTIC_PATH
    assert tuple(imported.GRIPPER_DRIVE_PROFILES) == finalizer.GRIPPER_DRIVE_PROFILES

    args.expected_diagnostic_sha256 = "0" * 64
    monkeypatch.setattr(
        finalizer.importlib.util,
        "spec_from_file_location",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("untrusted diagnostic was imported")
        ),
    )
    with pytest.raises(
        finalizer.GripperImpulseFinalizationError, match="bootstrap digest"
    ):
        finalizer._bootstrap_trusted_sources(args)


def test_secure_asset_rehash_rejects_multilink_scene(tmp_path):
    revision = "8c7e4103e266ef83d8b1ad2e9a63116edd5f155b"
    data_root = tmp_path / "data"
    food_root = data_root / "food_bussing"
    scene = food_root / "scene.usda"
    conditions = food_root / "initial_conditions.json"
    scene.parent.mkdir(parents=True)
    scene.write_bytes(b"scene")
    conditions.write_bytes(b"conditions")
    metadata_root = data_root / ".cache" / "huggingface" / "download" / "food_bussing"
    metadata_root.mkdir(parents=True)
    food_metadata = {}
    for filename in ("initial_conditions.json", "scene.usda"):
        path = metadata_root / f"{filename}.metadata"
        path.write_text(f"{revision}\n")
        identity, _ = finalizer._secure_file(path, field=filename)
        food_metadata[filename] = {**identity, "revision": revision}

    robot_root = tmp_path / "robot_data"
    robot = robot_root / "nvidia_droid" / "noninstanceable.usd"
    robot.parent.mkdir(parents=True)
    robot.write_bytes(b"robot")
    robot_identity, _ = finalizer._secure_file(robot, field="robot")
    robot_metadata_path = (
        robot_root
        / ".cache"
        / "huggingface"
        / "download"
        / "nvidia_droid"
        / "noninstanceable.usd.metadata"
    )
    robot_metadata_path.parent.mkdir(parents=True)
    robot_metadata_path.write_text(f"{revision}\n{robot_identity['sha256']}\n")
    robot_metadata_identity, _ = finalizer._secure_file(
        robot_metadata_path, field="robot metadata"
    )
    scene_identity, _ = finalizer._secure_file(scene, field="scene")
    conditions_identity, _ = finalizer._secure_file(conditions, field="conditions")
    capture = {
        "assets": {
            "foodbussing": {
                "scene": scene_identity,
                "initial_conditions": conditions_identity,
                "polaris_hub_revision": revision,
                "revision_metadata": food_metadata,
            },
            "robot_usd": robot_identity,
            "robot_usd_revision_metadata": {
                "identity": robot_metadata_identity,
                "revision": revision,
                "recorded_sha256": robot_identity["sha256"],
            },
        }
    }
    finalizer._secure_validate_live_assets(capture)
    os.link(scene, tmp_path / "scene-hardlink")
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="link count"):
        finalizer._secure_validate_live_assets(capture)


def test_execution_provenance_binds_three_distinct_exact_byte_wrapper_roles(
    tmp_path, monkeypatch
):
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"image")
    wrapper_bytes = (
        "#!/usr/bin/env bash\n"
        f"IMAGE={image.resolve()}\n"
        'srun --container-image="$IMAGE" cmd\n'
    ).encode()
    submitted = _write_immutable(tmp_path / "submitted.sh", wrapper_bytes)
    runtime_zero = _write_immutable(tmp_path / "runtime-dollar-zero.sh", wrapper_bytes)
    scontrol_batch = _write_immutable(tmp_path / "scontrol-batch.sh", wrapper_bytes)
    args = Namespace(
        container_image=image,
        submitted_saved_wrapper=submitted,
        runtime_dollar_zero_snapshot=runtime_zero,
        scontrol_batch_script_snapshot=scontrol_batch,
        expected_container_image_sha256=_sha256(image),
        expected_submitted_saved_wrapper_sha256=_sha256(submitted),
        expected_runtime_dollar_zero_sha256=_sha256(runtime_zero),
        expected_scontrol_batch_script_sha256=_sha256(scontrol_batch),
    )
    monkeypatch.setattr(
        finalizer, "_capture_slurm_provenance", lambda _args: {"slurm": True}
    )
    execution = finalizer._capture_execution_provenance(args)
    assert (
        execution["submitted_saved_wrapper"]["sha256"]
        == execution["runtime_dollar_zero_snapshot"]["sha256"]
        == execution["scontrol_batch_script_snapshot"]["sha256"]
    )
    runtime_zero.chmod(0o644)
    runtime_zero.write_bytes(b"different")
    runtime_zero.chmod(0o444)
    args.expected_runtime_dollar_zero_sha256 = _sha256(runtime_zero)
    with pytest.raises(
        finalizer.GripperImpulseFinalizationError, match="wrapper bytes"
    ):
        finalizer._capture_execution_provenance(args)


def test_execution_provenance_requires_pinned_container_path_literal(
    tmp_path, monkeypatch
):
    image = tmp_path / "image.sqsh"
    image.write_bytes(b"image")
    wrapper_bytes = b"#!/usr/bin/env bash\nsrun --container-image=$UNBOUND_IMAGE cmd\n"
    submitted = _write_immutable(tmp_path / "submitted.sh", wrapper_bytes)
    runtime_zero = _write_immutable(tmp_path / "runtime-dollar-zero.sh", wrapper_bytes)
    scontrol_batch = _write_immutable(tmp_path / "scontrol-batch.sh", wrapper_bytes)
    args = Namespace(
        container_image=image,
        submitted_saved_wrapper=submitted,
        runtime_dollar_zero_snapshot=runtime_zero,
        scontrol_batch_script_snapshot=scontrol_batch,
        expected_container_image_sha256=_sha256(image),
        expected_submitted_saved_wrapper_sha256=_sha256(submitted),
        expected_runtime_dollar_zero_sha256=_sha256(runtime_zero),
        expected_scontrol_batch_script_sha256=_sha256(scontrol_batch),
    )
    monkeypatch.setattr(
        finalizer, "_capture_slurm_provenance", lambda _args: {"slurm": True}
    )
    with pytest.raises(
        finalizer.GripperImpulseFinalizationError, match="container path literal"
    ):
        finalizer._capture_execution_provenance(args)


def _fake_capture() -> dict:
    return {
        "outcome": {"kind": "diagnostic_horizon_reached"},
        "gripper_drive_contract": {
            "profile": finalizer.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
        },
        "runtime_protocol": {
            "reset_seed": 0,
            "initial_condition_index": 0,
        },
        "runtime_frame": {"eef_frame": "panda_link8"},
        "runtime_exit_contract": {"profile": "runtime"},
        "assets": {"all": "assets"},
    }


def test_host_validation_forces_exact_stdlib_video_probe(tmp_path):
    calls = []

    def stdlib_probe(_path):
        return {"frame_count": 1, "height": 1, "width": 1}

    class Module:
        _probe_video_stdlib = staticmethod(stdlib_probe)

        @staticmethod
        def validate_capture_artifacts(
            capture,
            video,
            *,
            expected_mode,
            expected_gripper_drive_profile,
            probe,
        ):
            calls.append(
                (
                    capture,
                    video,
                    expected_mode,
                    expected_gripper_drive_profile,
                    probe,
                )
            )
            return {"validated": True}

    args = Namespace(
        validate_capture=tmp_path / "capture.json",
        video=tmp_path / "video.mp4",
        expected_mode="exact",
        expected_gripper_drive_profile=(
            finalizer.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
        ),
    )
    assert finalizer._validate_capture_with_stdlib_probe(Module, args) == {
        "validated": True
    }
    assert calls == [
        (
            args.validate_capture,
            args.video,
            "exact",
            finalizer.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE,
            Module._probe_video_stdlib,
        )
    ]


def test_finalizer_requires_closed_independent_gripper_drive_profile():
    parser = finalizer.build_parser()
    action = next(
        candidate
        for candidate in parser._actions  # noqa: SLF001
        if candidate.dest == "expected_gripper_drive_profile"
    )

    assert action.required is True
    assert tuple(action.choices) == finalizer.GRIPPER_DRIVE_PROFILES
    assert set(finalizer.GRIPPER_DRIVE_PROFILES) == {
        finalizer.GRIPPER_DRIVE_PROFILE,
        finalizer.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE,
    }


def _fake_context(tmp_path: Path) -> dict:
    def identity(name):
        return {
            "path": str(tmp_path / name),
            "size_bytes": 1,
            "sha256": "0" * 64,
            "mode": "0444",
            "nlink": 1,
            "mtime_ns": 0,
            "ctime_ns": 0,
            "publication_time_ns": 0,
        }

    return {
        "identities": {
            "capture": identity("capture"),
            "video": identity("video"),
            "ready_marker": identity("ready"),
            "runtime_exit": identity("runtime.exit"),
            "outer_srun_status": identity("srun.exit"),
        },
        "sources": {"source": True},
        "repository": {"commit": "a" * 40, "clean": True},
        "execution": {"slurm": {"snapshot": {"publication_time_ns": 0}}},
    }


def _action_args(tmp_path: Path, *, action: str) -> Namespace:
    validator = tmp_path / "validator.exit"
    if action != "validate" and not validator.exists():
        _write_immutable(validator, b"0\n")
    return Namespace(
        action=action,
        validate_capture=tmp_path / "capture.json",
        video=tmp_path / "video.mp4",
        ready_marker=tmp_path / "ready.json",
        runtime_exit=tmp_path / "runtime.exit",
        outer_srun_status=tmp_path / "srun.exit",
        validator_status=None if action == "validate" else validator,
        intended_attestation_path=tmp_path / "final.attestation.json",
        attestation_staging_output=(
            tmp_path / "isolated" / "staging.attestation.json"
            if action == "finalize"
            else None
        ),
        attestation_input=(
            tmp_path / "final.attestation.json" if action == "verify" else None
        ),
        expected_mode="exact",
        expected_gripper_drive_profile=(
            finalizer.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
        ),
        container_image=tmp_path / "image",
        submitted_saved_wrapper=tmp_path / "submitted-saved-wrapper",
        runtime_dollar_zero_snapshot=tmp_path / "runtime-dollar-zero-snapshot",
        scontrol_batch_script_snapshot=tmp_path / "scontrol-batch-script-snapshot",
        slurm_job_oneliner_snapshot=tmp_path / "scontrol-show-job-oneliner",
    )


def test_attestation_rejects_capture_expected_profile_swap(tmp_path):
    capture = _fake_capture()
    capture["gripper_drive_contract"]["profile"] = finalizer.GRIPPER_DRIVE_PROFILE
    context = _fake_context(tmp_path)
    validator_status = next(iter(context["identities"].values()))

    with pytest.raises(
        finalizer.GripperImpulseFinalizationError,
        match="attestation/capture gripper drive profile mismatch",
    ):
        finalizer._attestation_payload(  # noqa: SLF001
            capture=capture,
            context=context,
            validator_status=validator_status,
            mode="exact",
            gripper_drive_profile=(
                finalizer.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
            ),
            intended_attestation_path=tmp_path / "final.attestation.json",
        )


class _FixedParser:
    def __init__(self, args):
        self.args = args

    def parse_args(self, _argv):
        return self.args


def test_staging_parent_symlink_cannot_alias_intended_final(tmp_path):
    args = _action_args(tmp_path, action="finalize")
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    args.attestation_staging_output = alias / args.intended_attestation_path.name
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="collision"):
        finalizer._require_action_paths(args)


def test_action_paths_enforce_ten_distinct_ceremony_inputs(tmp_path):
    args = _action_args(tmp_path, action="validate")
    finalizer._require_action_paths(args)
    args.slurm_job_oneliner_snapshot = args.runtime_dollar_zero_snapshot
    with pytest.raises(finalizer.GripperImpulseFinalizationError, match="collide"):
        finalizer._require_action_paths(args)


def test_validate_finalize_verify_use_isolated_staging_and_exact_bytes(
    tmp_path, monkeypatch
):
    capture = _fake_capture()
    context = _fake_context(tmp_path)
    monkeypatch.setattr(finalizer, "_validate_common", lambda _args: (capture, context))
    monkeypatch.setattr(
        finalizer, "_bootstrap_trusted_sources", lambda _args: diagnostic_module
    )

    validate_args = _action_args(tmp_path, action="validate")
    monkeypatch.setattr(finalizer, "build_parser", lambda: _FixedParser(validate_args))
    assert finalizer.main([]) == 0
    assert not validate_args.intended_attestation_path.exists()

    finalize_args = _action_args(tmp_path, action="finalize")
    monkeypatch.setattr(finalizer, "build_parser", lambda: _FixedParser(finalize_args))
    assert finalizer.main([]) == 0
    staging = finalize_args.attestation_staging_output
    assert staging.is_file()
    assert stat.S_IMODE(staging.stat().st_mode) == 0o444
    assert not finalize_args.intended_attestation_path.exists()
    staged = json.loads(staging.read_text())
    assert staged["intended_attestation_path"] == str(
        finalize_args.intended_attestation_path.absolute()
    )
    assert (
        staged["gripper_drive_profile"]
        == finalizer.GRIPPER_VELOCITY_LIMIT_CANDIDATE_DRIVE_PROFILE
    )

    final_path = finalize_args.intended_attestation_path
    _write_immutable(final_path, staging.read_bytes())
    verify_args = _action_args(tmp_path, action="verify")
    monkeypatch.setattr(finalizer, "build_parser", lambda: _FixedParser(verify_args))
    assert finalizer.main([]) == 0

    tampered = copy.deepcopy(staged)
    tampered["schema_version"] = True
    final_path.chmod(0o644)
    final_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
    final_path.chmod(0o444)
    with pytest.raises(
        finalizer.GripperImpulseFinalizationError, match="byte verification"
    ):
        finalizer.main([])
