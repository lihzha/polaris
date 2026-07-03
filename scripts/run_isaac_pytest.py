#!/usr/bin/env python3
"""Launch Isaac Sim before collecting tests that import Isaac Lab extensions."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import stat
import sys
import traceback


EXIT_CODE_FILE_ENV = "ISAAC_PYTEST_EXIT_CODE_FILE"


def _report_exception(error: BaseException) -> None:
    """Report a bootstrap failure without risking its nonzero exit status."""

    try:
        traceback.print_exception(type(error), error, error.__traceback__)
    except BaseException:
        pass


def _flush_streams(exit_code: int, *, streams=None) -> int:
    """Best-effort flush and return the resulting fail-closed exit code."""

    if streams is None:
        streams = (sys.stdout, sys.stderr)
    final_exit_code = exit_code
    for stream in streams:
        try:
            stream.flush()
        except BaseException:
            final_exit_code = 1
    return final_exit_code


def _publish_exit_code(path: Path, exit_code: int) -> None:
    """Atomically publish one immutable, non-overwriting exit-code file."""

    if type(exit_code) is not int or not 0 <= exit_code <= 255:
        raise ValueError(f"invalid exit code: {exit_code!r}")
    if not path.is_absolute():
        raise ValueError(f"exit-code file must be absolute: {path}")
    if not path.parent.is_dir():
        raise ValueError(f"exit-code parent directory does not exist: {path.parent}")

    payload = f"{exit_code}\n".encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp")
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short write while publishing exit-code file")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None

        metadata = temporary.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"temporary exit-code file is not regular: {temporary}")
        if stat.S_IMODE(metadata.st_mode) != 0o444:
            raise ValueError(f"temporary exit-code file mode is not 0444: {temporary}")
        if temporary.read_bytes() != payload:
            raise ValueError(f"temporary exit-code file reread mismatch: {temporary}")
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

        # This hard link is the final critical operation: it publishes the
        # already-fsynced and validated inode without replacing an existing
        # target. The wrapper performs the final-directory sync and reread.
        os.link(temporary, path, follow_symlinks=False)
        try:
            temporary.unlink()
        except BaseException:
            # A leftover temporary makes the wrapper reject the run, while the
            # correctly published final file remains authoritative.
            pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except BaseException:
            # The wrapper rejects a leftover temporary or missing final file.
            pass


def _flush_and_exit(exit_code: int, *, streams=None, exit_code_file=None) -> None:
    """Flush, optionally publish the final code, then bypass Kit hooks."""

    final_exit_code = _flush_streams(exit_code, streams=streams)
    try:
        if exit_code_file is not None:
            _publish_exit_code(Path(exit_code_file), final_exit_code)
    except BaseException as error:
        final_exit_code = 1
        _report_exception(error)
        _flush_streams(final_exit_code)
    finally:
        os._exit(final_exit_code)


def main() -> None:
    # Kit teardown can mutate process-global state. Capture the host-provided
    # publication path before AppLauncher or pytest can run.
    exit_code_file = os.environ.get(EXIT_CODE_FILE_ENV)
    pytest_args = sys.argv[1:]
    sys.argv = [sys.argv[0]]

    launcher = None
    exit_code = 1
    try:
        app_launcher_module = importlib.import_module("isaaclab.app")
        app_launcher_type = app_launcher_module.AppLauncher
        launcher = app_launcher_type({"headless": True, "enable_cameras": False})
        controller_module = importlib.import_module(
            "isaaclab.controllers.differential_ik"
        )
        importlib.import_module("omni.log")
        pytest = importlib.import_module("pytest")

        print(f"ISAAC_PYTEST_ARGS={pytest_args!r}", flush=True)
        print(f"ISAAC_PYTEST_EXIT_CODE_FILE={exit_code_file!r}", flush=True)
        print(
            f"ISAAC_CONTROLLER={controller_module.DifferentialIKController.__module__}",
            flush=True,
        )
        print("ISAAC_PYTEST_BOOTSTRAP_OK", flush=True)
        exit_code = int(pytest.main(pytest_args))
        print(f"ISAAC_PYTEST_EXIT_CODE={exit_code}", flush=True)
    except BaseException as error:
        exit_code = 1
        _report_exception(error)
    finally:
        if launcher is not None:
            try:
                launcher.app.close()
            except BaseException as error:
                exit_code = 1
                _report_exception(error)
        # Kit teardown can otherwise mask pytest's nonzero status. Bypass
        # interpreter/atexit hooks after the app has closed so Slurm sees the
        # exact test result. This helper reaches os._exit even if flushing
        # itself fails.
        _flush_and_exit(
            exit_code,
            exit_code_file=exit_code_file,
        )


if __name__ == "__main__":
    main()
