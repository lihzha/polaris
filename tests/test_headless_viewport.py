from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from polaris import headless_viewport as viewport


@dataclass
class _Prim:
    valid: bool

    def IsValid(self) -> bool:
        return self.valid


class _Camera:
    def __init__(self, prim: _Prim):
        self._prim = prim

    def GetPrim(self) -> _Prim:
        return self._prim


class _Stage:
    def __init__(self, valid: bool):
        self.prim = _Prim(valid)

    def GetPrimAtPath(self, path: str) -> _Prim:
        assert path == viewport.DEFAULT_VIEWPORT_CAMERA_PRIM_PATH
        return self.prim


def _context_type(stage: _Stage):
    class Context:
        calls: list[tuple[object, object, str]] = []

        def set_camera_view(
            self,
            eye,
            target,
            camera_prim_path=viewport.DEFAULT_VIEWPORT_CAMERA_PRIM_PATH,
        ):
            if camera_prim_path == viewport.DEFAULT_VIEWPORT_CAMERA_PRIM_PATH:
                assert stage.prim.IsValid()
            self.calls.append((eye, target, camera_prim_path))
            return "moved"

    return Context


def test_guard_defines_only_a_missing_default_camera_before_first_move() -> None:
    stage = _Stage(valid=False)
    context_type = _context_type(stage)
    definitions: list[str] = []
    messages: list[str] = []

    def define(_stage: _Stage, path: str) -> _Camera:
        definitions.append(path)
        _stage.prim = _Prim(True)
        return _Camera(_stage.prim)

    assert viewport.install_viewport_camera_guard(
        context_type,
        stage_getter=lambda: stage,
        camera_definer=define,
        emit=messages.append,
    )

    assert context_type().set_camera_view((1, 2, 3), (0, 0, 0)) == "moved"
    assert definitions == [viewport.DEFAULT_VIEWPORT_CAMERA_PRIM_PATH]
    assert context_type.calls == [
        ((1, 2, 3), (0, 0, 0), viewport.DEFAULT_VIEWPORT_CAMERA_PRIM_PATH)
    ]
    assert messages == [
        "POLARIS_HEADLESS_VIEWPORT_CAMERA_RECOVERY="
        f"profile={viewport.HEADLESS_VIEWPORT_RECOVERY_PROFILE};"
        f"prim_path={viewport.DEFAULT_VIEWPORT_CAMERA_PRIM_PATH}"
    ]


def test_guard_leaves_an_existing_default_camera_untouched() -> None:
    stage = _Stage(valid=True)
    context_type = _context_type(stage)

    assert viewport.install_viewport_camera_guard(
        context_type,
        stage_getter=lambda: stage,
        camera_definer=lambda *_: pytest.fail("camera must not be redefined"),
        emit=lambda _: pytest.fail("recovery must not be emitted"),
    )

    assert context_type().set_camera_view((1, 1, 1), (0, 0, 0)) == "moved"


def test_guard_does_not_inspect_or_define_nondefault_camera_paths() -> None:
    stage = _Stage(valid=False)
    context_type = _context_type(stage)

    assert viewport.install_viewport_camera_guard(
        context_type,
        stage_getter=lambda: pytest.fail("stage must not be inspected"),
        camera_definer=lambda *_: pytest.fail("camera must not be defined"),
    )

    assert (
        context_type().set_camera_view(
            (1, 1, 1), (0, 0, 0), camera_prim_path="/World/ExternalCamera"
        )
        == "moved"
    )


def test_guard_installation_is_idempotent() -> None:
    stage = _Stage(valid=True)
    context_type = _context_type(stage)
    kwargs = {
        "stage_getter": lambda: stage,
        "camera_definer": lambda *_: pytest.fail("camera must not be defined"),
    }

    assert viewport.install_viewport_camera_guard(context_type, **kwargs)
    assert not viewport.install_viewport_camera_guard(context_type, **kwargs)
    assert context_type().set_camera_view((1, 1, 1), (0, 0, 0)) == "moved"
    assert len(context_type.calls) == 1


def test_guard_fails_closed_when_camera_definition_is_invalid() -> None:
    stage = _Stage(valid=False)
    context_type = _context_type(stage)

    viewport.install_viewport_camera_guard(
        context_type,
        stage_getter=lambda: stage,
        camera_definer=lambda *_: _Camera(_Prim(False)),
    )

    with pytest.raises(RuntimeError, match="Failed to define"):
        context_type().set_camera_view((1, 1, 1), (0, 0, 0))
    assert context_type.calls == []


def test_guard_preserves_the_original_failure_when_no_stage_exists() -> None:
    class Context:
        def set_camera_view(self, eye, target, camera_prim_path):
            raise RuntimeError("original missing-stage failure")

    viewport.install_viewport_camera_guard(
        Context,
        stage_getter=lambda: None,
        camera_definer=lambda *_: pytest.fail("camera must not be defined"),
    )

    with pytest.raises(RuntimeError, match="original missing-stage failure"):
        Context().set_camera_view((1, 1, 1), (0, 0, 0))


def test_cluster_smoke_forces_one_removal_and_requires_one_recovery() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "smoke_headless_viewport_camera_recovery.py"
    ).read_text()
    tree = ast.parse(source)

    remove_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "RemovePrim"
    ]
    guard_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "install_viewport_camera_guard"
    ]
    assert len(remove_calls) == 1
    assert len(guard_calls) == 1
    assert '"camera_valid_after_recovery": camera_valid' in source
    assert "len(recovery_messages) != 1" in source


def test_cluster_smoke_seals_ready_marker_before_simulation_close() -> None:
    source = (
        Path(__file__).parents[1]
        / "scripts"
        / "smoke_headless_viewport_camera_recovery.py"
    ).read_text()

    success_stage = '"stage": "simulation_app_close_pending"'
    raw_publish = "raw_identity = _publish(output_json, payload)"
    ready_publish = "_publish(ready_marker, ready_payload)"
    simulation_close = "simulation_app.close()"
    failure_exit = "os._exit(1)"

    assert source.count(success_stage) == 2
    assert source.count(raw_publish) == 1
    assert source.count(ready_publish) == 1
    assert source.count(simulation_close) == 1
    assert source.count(failure_exit) == 2
    assert source.index(raw_publish) < source.index(ready_publish)
    assert source.index(ready_publish) < source.index(simulation_close)
    assert 'output_json.name + ".ready.json"' in source
    assert '"raw_result": raw_identity' in source
    assert "os.O_EXCL | os.O_NOFOLLOW" in source
    assert "os.fchmod(descriptor, 0o444)" in source
    assert "os.fsync(directory_descriptor)" in source


def test_cluster_smoke_publication_is_immutable_and_bound(tmp_path: Path) -> None:
    source_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "smoke_headless_viewport_camera_recovery.py"
    )
    tree = ast.parse(source_path.read_text())
    publish = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_publish"
    )
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "stat": stat,
    }
    exec(
        compile(ast.Module(body=[publish], type_ignores=[]), source_path, "exec"),
        namespace,
    )

    path = tmp_path / "sealed.json"
    identity = namespace["_publish"](path, {"schema_version": 1, "passed": True})
    payload = path.read_bytes()

    assert identity == {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "mode": "0444",
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert path.stat().st_nlink == 1
    with pytest.raises(FileExistsError):
        namespace["_publish"](path, {"schema_version": 1, "passed": False})


def test_cluster_smoke_ready_marker_exists_before_hard_close(tmp_path: Path) -> None:
    source_path = (
        Path(__file__).parents[1]
        / "scripts"
        / "smoke_headless_viewport_camera_recovery.py"
    )
    tree = ast.parse(source_path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_publish", "_publish_success_and_close"}
    ]
    namespace = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "stat": stat,
        "sys": __import__("sys"),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), source_path, "exec"),
        namespace,
    )

    output = tmp_path / "raw.json"

    class HardClose:
        called = False

        def close(self):
            assert output.is_file()
            assert output.with_name("raw.json.ready.json").is_file()
            self.called = True

    simulation_app = HardClose()
    namespace["_publish_success_and_close"](
        output,
        {"schema_version": 1, "stage": "simulation_app_close_pending"},
        simulation_app,
    )

    marker = json.loads(output.with_name("raw.json.ready.json").read_bytes())
    raw = output.read_bytes()
    assert simulation_app.called
    assert marker == {
        "schema_version": 1,
        "stage": "simulation_app_close_pending",
        "raw_result": {
            "path": str(output),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mode": "0444",
        },
    }


def test_environment_installs_guard_immediately_before_base_constructor() -> None:
    source = (
        Path(__file__).parents[1]
        / "src"
        / "polaris"
        / "environments"
        / "manager_based_rl_splat_environment.py"
    ).read_text()
    tree = ast.parse(source)
    environment = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ManagerBasedRLSplatEnv"
    )
    constructor = next(
        node
        for node in environment.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    guard_index = next(
        index
        for index, node in enumerate(constructor.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "install_isaaclab_headless_viewport_camera_guard"
    )
    super_index = next(
        index
        for index, node in enumerate(constructor.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "__init__"
    )
    assert guard_index < super_index
