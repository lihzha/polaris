"""Exactly-once evaluator cleanup for the native pi0.5 child process."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any


class NativeEvaluatorLifecycle:
    """Close env, publish close-ready evidence, then close Kit in that order."""

    def __init__(self, simulation_app: Any) -> None:
        self._simulation_app = simulation_app
        self._env: Any | None = None
        self._close_ready: (
            tuple[
                Callable[[Path, dict[str, Any]], dict[str, Any]],
                Path,
                dict[str, Any],
            ]
            | None
        ) = None
        self._closed = False

    def bind_environment(self, env: Any) -> None:
        if self._env is not None or self._closed:
            raise RuntimeError(
                "Native evaluator environment lifecycle is already bound"
            )
        self._env = env

    def prepare_close_ready(
        self,
        publisher: Callable[[Path, dict[str, Any]], dict[str, Any]],
        path: Path,
        payload: dict[str, Any],
    ) -> None:
        if self._env is None or self._close_ready is not None or self._closed:
            raise RuntimeError("Native evaluator close-ready lifecycle is invalid")
        self._close_ready = (publisher, Path(path), payload)

    def close(self) -> None:
        if self._closed:
            raise RuntimeError("Native evaluator lifecycle closed more than once")
        self._closed = True
        try:
            if self._env is not None:
                self._env.close()
            if self._close_ready is not None:
                publisher, path, payload = self._close_ready
                publisher(path, payload)
        except BaseException as error:
            # SimulationApp.close() may terminate the process with status zero.
            # Do not let it mask an env-close or immutable-ready publication
            # failure.  Only a fully prepared success reaches Kit teardown.
            raise RuntimeError("Native evaluator pre-Kit cleanup failed") from error
        self._simulation_app.close()
