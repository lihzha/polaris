#!/usr/bin/env python3
"""Launch Isaac Sim before collecting tests that import Isaac Lab extensions."""

from __future__ import annotations

import fcntl
import importlib
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import traceback


EXIT_CODE_FILE_ENV = "ISAAC_PYTEST_EXIT_CODE_FILE"
CHILD_PROCESS_ENV = "ISAAC_PYTEST_CHILD_PROCESS"
CHILD_RESULT_FD_ENV = "ISAAC_PYTEST_CHILD_RESULT_FD"
CHILD_TIMEOUT_SECONDS = 300
CHILD_TERMINATE_GRACE_SECONDS = 10
PYTEST_EXIT_CODES = frozenset(range(6))


class _ParentSignalError(RuntimeError):
    """Raised so the parent can reap Kit before honoring an external signal."""


def _raise_parent_signal(signum, _frame) -> None:
    raise _ParentSignalError(f"received signal {signum} while Kit child was active")


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


def _prepare_child_result_descriptor() -> int:
    """Validate the inherited pipe and block inheritance by Kit descendants."""

    descriptor_text = os.environ.pop(CHILD_RESULT_FD_ENV, None)
    if descriptor_text is None:
        raise ValueError(f"missing {CHILD_RESULT_FD_ENV}")
    descriptor = int(descriptor_text)
    if descriptor < 3:
        raise ValueError(f"unsafe child result descriptor: {descriptor}")
    metadata = os.fstat(descriptor)
    if not stat.S_ISFIFO(metadata.st_mode):
        raise ValueError(f"child result descriptor is not a pipe: {descriptor}")
    access_mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
    if access_mode != os.O_WRONLY:
        raise ValueError(f"child result descriptor is not write-only: {descriptor}")
    os.set_inheritable(descriptor, False)
    return descriptor


def _write_child_exit_code(exit_code: int, descriptor: int) -> None:
    """Send one exact result byte to the non-Kit parent before teardown."""

    if type(exit_code) is not int or exit_code not in PYTEST_EXIT_CODES:
        raise ValueError(f"invalid child exit code: {exit_code!r}")
    try:
        if os.write(descriptor, bytes((exit_code,))) != 1:
            raise OSError("short write to Isaac pytest parent result pipe")
    except BaseException:
        try:
            os.close(descriptor)
        except BaseException:
            pass
        raise
    # The one-byte write is the report commit point. A close error cannot
    # revoke a byte already visible to the parent, so cleanup is best-effort.
    try:
        os.close(descriptor)
    except BaseException:
        pass


def _child_main(pytest_args: list[str]) -> None:
    """Run pytest under Kit and report its result before native teardown."""

    os.environ.pop(CHILD_PROCESS_ENV, None)
    sys.argv = [sys.argv[0]]

    launcher = None
    exit_code = 1
    result_descriptor = None
    try:
        result_descriptor = _prepare_child_result_descriptor()
        app_launcher_module = importlib.import_module("isaaclab.app")
        app_launcher_type = app_launcher_module.AppLauncher
        launcher = app_launcher_type({"headless": True, "enable_cameras": False})
        controller_module = importlib.import_module(
            "isaaclab.controllers.differential_ik"
        )
        importlib.import_module("omni.log")
        pytest = importlib.import_module("pytest")

        print(f"ISAAC_PYTEST_ARGS={pytest_args!r}", flush=True)
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
        exit_code = _flush_streams(exit_code)
        try:
            if result_descriptor is None:
                raise ValueError("Isaac pytest child result pipe was not prepared")
            print(f"ISAAC_PYTEST_CHILD_REPORTED_EXIT_CODE={exit_code}", flush=True)
            # Commit the result only after every fallible diagnostic flush.
            # No Python I/O occurs between a successful write and Kit close.
            _write_child_exit_code(exit_code, result_descriptor)
        except BaseException as error:
            exit_code = 1
            _report_exception(error)
            _flush_streams(exit_code)
        if launcher is not None:
            try:
                launcher.app.close()
            except BaseException as error:
                exit_code = 1
                _report_exception(error)
        # This is reached only when Kit close returns to Python. On the pinned
        # runtime it terminates the child natively; either outcome closes the
        # result pipe so the non-Kit parent can finish the transaction.
        _flush_and_exit(exit_code)


def _kill_lingering_process_group(process_group: int) -> bool:
    """Kill and drain an unexpected surviving child process group."""

    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    raise RuntimeError(
        f"Isaac pytest child process group survived SIGKILL: {process_group}"
    )


def _force_kill_and_reap(process) -> None:
    """Unconditionally kill the child group, reap its leader, and drain survivors."""

    kill_error = None
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except BaseException as error:
        kill_error = error

    reaped = False
    wait_error = None
    for _ in range(3):
        try:
            process.wait(timeout=10)
            reaped = True
            break
        except subprocess.TimeoutExpired:
            continue
        except ChildProcessError:
            reaped = True
            break
        except BaseException as error:
            wait_error = error
            break
    if not reaped:
        try:
            reaped = process.poll() is not None
        except BaseException as error:
            wait_error = error

    # This runs even when every bounded wait above failed.
    lingering_error = None
    try:
        _kill_lingering_process_group(process.pid)
    except BaseException as error:
        lingering_error = error
    if kill_error is not None:
        raise RuntimeError(
            f"Could not signal Isaac pytest child group: {process.pid}"
        ) from kill_error
    if lingering_error is not None:
        raise RuntimeError(
            f"Could not drain Isaac pytest child group: {process.pid}"
        ) from lingering_error
    if not reaped:
        raise RuntimeError(
            f"Isaac pytest child could not be reaped: {process.pid}"
        ) from wait_error


def _run_isaac_child(pytest_args: list[str]) -> tuple[int | None, int]:
    """Run the Kit child and collect its pre-teardown result plus process code."""

    read_descriptor, write_descriptor = os.pipe()
    environment = os.environ.copy()
    environment[CHILD_PROCESS_ENV] = "1"
    environment[CHILD_RESULT_FD_ENV] = str(write_descriptor)
    environment.pop(EXIT_CODE_FILE_ENV, None)
    process = None
    previous_signal_handlers = None
    try:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *pytest_args],
            env=environment,
            pass_fds=(write_descriptor,),
            start_new_session=True,
        )
        os.close(write_descriptor)
        write_descriptor = None

        previous_signal_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGTERM, signal.SIGINT)
        }
        for signum in previous_signal_handlers:
            signal.signal(signum, _raise_parent_signal)

        try:
            process_return_code = int(process.wait(timeout=CHILD_TIMEOUT_SECONDS))
        except subprocess.TimeoutExpired as timeout_error:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=CHILD_TERMINATE_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
            raise timeout_error
        if _kill_lingering_process_group(process.pid):
            raise RuntimeError("Isaac pytest child left a surviving process group")
        os.set_blocking(read_descriptor, False)
        payload = b""
        reached_eof = False
        while True:
            try:
                chunk = os.read(read_descriptor, 2)
            except BlockingIOError:
                break
            if not chunk:
                reached_eof = True
                break
            payload += chunk
        reported_exit_code = (
            payload[0]
            if reached_eof and len(payload) == 1 and payload[0] in PYTEST_EXIT_CODES
            else None
        )
        return reported_exit_code, process_return_code
    except BaseException as error:
        if process is not None:
            try:
                _force_kill_and_reap(process)
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "Isaac pytest child cleanup failed after bootstrap error"
                ) from cleanup_error
        raise error
    finally:
        if previous_signal_handlers is not None:
            for signum, handler in previous_signal_handlers.items():
                signal.signal(signum, handler)
        if write_descriptor is not None:
            try:
                os.close(write_descriptor)
            except BaseException:
                pass
        try:
            os.close(read_descriptor)
        except BaseException:
            pass


def _resolve_child_exit_code(
    reported_exit_code: int | None,
    process_return_code: int,
) -> int:
    """Fail closed unless the pre-teardown report and child exit are coherent."""

    if reported_exit_code is None:
        return 1
    if reported_exit_code not in PYTEST_EXIT_CODES:
        return 1
    # Kit may terminate the child with zero during close, masking pytest's
    # nonzero result. A normal Python return preserves the reported code.
    if process_return_code not in (0, reported_exit_code):
        return 1
    return reported_exit_code


def _parent_main(pytest_args: list[str]) -> None:
    """Own the final sidecar outside Kit so native teardown cannot skip it."""

    exit_code_file = os.environ.get(EXIT_CODE_FILE_ENV)
    exit_code = 1
    try:
        print(f"ISAAC_PYTEST_PARENT_ARGS={pytest_args!r}", flush=True)
        print(f"ISAAC_PYTEST_EXIT_CODE_FILE={exit_code_file!r}", flush=True)
        reported_exit_code, process_return_code = _run_isaac_child(pytest_args)
        exit_code = _resolve_child_exit_code(
            reported_exit_code,
            process_return_code,
        )
        print(
            "ISAAC_PYTEST_CHILD_RESULT="
            f"reported={reported_exit_code!r};process={process_return_code};"
            f"resolved={exit_code}",
            flush=True,
        )
    except BaseException as error:
        exit_code = 1
        _report_exception(error)
    _flush_and_exit(exit_code, exit_code_file=exit_code_file)


def main() -> None:
    pytest_args = sys.argv[1:]
    if os.environ.get(CHILD_PROCESS_ENV) == "1":
        _child_main(pytest_args)
    else:
        _parent_main(pytest_args)


if __name__ == "__main__":
    main()
