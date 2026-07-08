"""Capture deterministic in-process graphics environment mutations on l401.

This diagnostic intentionally follows the import and AppLauncher order used by
``scripts/eval.py``.  It does not construct a task, connect to a policy server,
or execute a policy action.  The output is written durably before Kit teardown.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


_ENVIRONMENT_NAMES = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_PROFILE",
    "LD_DEBUG",
    "LD_DEBUG_OUTPUT",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "VK_DRIVER_FILES",
    "VK_ADD_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "VK_SDK_PATH",
    "VULKAN_SDK",
    "VK_LAYER_PATH",
    "VK_ADD_LAYER_PATH",
    "VK_INSTANCE_LAYERS",
    "VK_DEVICE_LAYERS",
    "VK_LOADER_DRIVERS_SELECT",
    "VK_LOADER_DRIVERS_DISABLE",
    "VK_LOADER_LAYERS_ENABLE",
    "VK_LOADER_LAYERS_DISABLE",
    "VK_LOADER_LAYERS_ALLOW",
    "VULKAN_HEADERS_INSTALL_DIR",
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "QT_QPA_FONTDIR",
    "__GLX_VENDOR_LIBRARY_NAME",
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "LIBGL_DRIVERS_PATH",
)


def _environment() -> dict[str, Any]:
    # The job launches through /usr/bin/env -i with an explicit non-secret
    # allowlist, so recording the complete in-process environment is safe and
    # prevents a diagnostic blind spot from another previously unknown import
    # mutation.  The selected view keeps loader-sensitive values easy to audit.
    return {
        "selected": {name: os.environ.get(name) for name in _ENVIRONMENT_NAMES},
        "full": dict(sorted(os.environ.items())),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Expected a regular file: {path}")
    return {
        "path": str(path),
        "size": metadata.st_size,
        "sha256": _sha256(path),
    }


def _proc_initial_environment() -> dict[str, str]:
    payload = Path("/proc/self/environ").read_bytes()
    if payload and not payload.endswith(b"\0"):
        raise ValueError("/proc/self/environ is truncated")
    result = {}
    for raw in payload.rstrip(b"\0").split(b"\0") if payload else ():
        name, value = raw.split(b"=", 1)
        decoded_name = name.decode("utf-8", errors="strict")
        if decoded_name in result:
            raise ValueError("/proc/self/environ has duplicate keys")
        result[decoded_name] = value.decode("utf-8", errors="strict")
    return dict(sorted(result.items()))


def _mapped_identity(path: Path) -> dict[str, Any]:
    identities = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5] != str(path):
            continue
        identities.add((fields[3].lower(), int(fields[4], 10)))
    if len(identities) != 1:
        raise ValueError(f"Expected one mapped identity for {path}")
    device, inode = next(iter(identities))
    return {"device": device, "inode": inode}


def _opencv_identity() -> dict[str, Any]:
    import cv2

    package_dir = Path(cv2.__file__).resolve().parent
    native = cv2._native
    version = cv2.version
    higher_priority_config = package_dir / "config-3.11.py"
    selected = []
    for relative in (
        "__init__.py",
        "config.py",
        "config-3.py",
        "load_config_py2.py",
        "load_config_py3.py",
        "version.py",
        "cv2.abi3.so",
    ):
        candidate = package_dir / relative
        if candidate.is_file():
            selected.append(_file_identity(candidate))
    distributions = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "")
        if "opencv" in name.lower():
            distributions.append({"name": name, "version": distribution.version})
    return {
        "module_file": str(Path(cv2.__file__).resolve()),
        "module_spec_origin": str(Path(cv2.__spec__.origin).resolve()),
        "native_module_file": str(Path(native.__file__).resolve()),
        "native_module_spec_origin": str(Path(native.__spec__.origin).resolve()),
        "native_module_maps_identity": _mapped_identity(
            Path(native.__file__).resolve()
        ),
        "load_config_module_file": str(
            Path(sys.modules["cv2.load_config_py3"].__file__).resolve()
        ),
        "version_module_file": str(Path(version.__file__).resolve()),
        "version": cv2.__version__,
        "package_version": version.opencv_version,
        "ci_build": version.ci_build,
        "headless": version.headless,
        "contrib": version.contrib,
        "rolling": version.rolling,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_lexical": str(Path(sys.executable).absolute()),
        "python_implementation": sys.implementation.name,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_major": sys.version_info.major,
        "python_minor": sys.version_info.minor,
        "higher_priority_config_path": str(higher_priority_config),
        "higher_priority_config_exists": higher_priority_config.exists(),
        "selected_config_path": str(package_dir / "config-3.py"),
        "build_information_sha256": hashlib.sha256(
            cv2.getBuildInformation().encode("utf-8")
        ).hexdigest(),
        "distributions": sorted(distributions, key=lambda item: item["name"]),
        "selected_files": selected,
        "loaded_cv2_modules": sorted(
            name for name in sys.modules if name.startswith("cv2")
        ),
    }


def _loader_search_safety() -> dict[str, Any]:
    working_directory = Path.cwd().resolve()
    repository_root = Path(__file__).resolve().parents[2]
    loader_value = os.environ.get("LD_LIBRARY_PATH")
    first_element = loader_value.split(":", 1)[0] if loader_value is not None else None
    normalized_target = (
        str(Path(first_element).resolve(strict=False))
        if first_element is not None
        else None
    )
    root_elf_files = []
    for entry in sorted(os.scandir(working_directory), key=lambda item: item.name):
        if not entry.is_file(follow_symlinks=False):
            continue
        candidate = Path(entry.path)
        with candidate.open("rb") as stream:
            if stream.read(4) == b"\x7fELF":
                root_elf_files.append(_file_identity(candidate))
    return {
        "profile": "cv2_empty_loader_element_readonly_workdir_v1",
        "working_directory": str(working_directory),
        "repository_root": str(repository_root),
        "working_directory_equals_repository_root": (
            working_directory == repository_root
        ),
        "working_directory_read_only": bool(
            os.statvfs(working_directory).f_flag & os.ST_RDONLY
        ),
        "ld_library_path": loader_value,
        "normalized_first_element": normalized_target,
        "normalized_first_element_exists": (
            Path(normalized_target).exists() if normalized_target is not None else None
        ),
        "working_directory_elf_files": root_elf_files,
    }


def _publish(path: Path, value: dict[str, Any]) -> None:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o444,
    )
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if path.read_bytes() != payload:
        raise RuntimeError("Published graphics-environment probe changed on reread")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args, _ = parser.parse_known_args()

    report: dict[str, Any] = {
        "schema_version": 1,
        "profile": "pi05_eval_import_order_graphics_environment_probe_v1",
        "python": sys.version,
        "stages": {"process_start": _environment()},
        "proc_initial_environment": _proc_initial_environment(),
    }

    # Match the global import order of scripts/eval.py before AppLauncher.
    import tyro  # noqa: F401
    import mediapy  # noqa: F401
    import tqdm  # noqa: F401
    import gymnasium  # noqa: F401
    import torch  # noqa: F401
    import pandas  # noqa: F401
    from isaaclab.app import AppLauncher

    report["stages"]["eval_global_imports"] = _environment()
    launcher_args = argparse.Namespace(enable_cameras=True, headless=True)
    simulation_app = AppLauncher(launcher_args).app
    report["stages"]["app_started"] = _environment()

    # This is the first eval.py import that loads the PolaRiS renderer and cv2.
    from polaris.environments import manager_based_rl_splat_environment  # noqa: F401
    from polaris import pi05_droid_jointpos_runtime as runtime

    report["stages"]["manager_and_cv2_imported"] = _environment()
    report["runtime_graphics_environment"] = runtime._graphics_environment()
    report["runtime_profile"] = runtime.PI05_DROID_JOINTPOS_GRAPHICS_RUNTIME_PROFILE
    report["runtime_expected_ld_library_path"] = (
        runtime.PI05_DROID_JOINTPOS_GRAPHICS_EXPECTED_LD_LIBRARY_PATH
    )
    report["runtime_module"] = _file_identity(Path(runtime.__file__).resolve())
    report["opencv"] = _opencv_identity()
    report["loader_search_safety"] = _loader_search_safety()
    report["proc_initial_environment_unchanged"] = (
        report["proc_initial_environment"] == _proc_initial_environment()
    )
    report["sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _publish(args.output, report)
    print(
        "POLARIS_PI05_GRAPHICS_ENVIRONMENT_PROBE="
        + json.dumps(
            {
                "output": str(args.output),
                "sha256": _sha256(args.output),
                "content_sha256": report["sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    simulation_app.close()


if __name__ == "__main__":
    main()
