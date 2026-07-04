"""Deterministic recovery for Isaac Lab's headless viewport camera prim.

Isaac Lab 2.3 constructs a viewport camera controller whenever camera rendering
is enabled, including headless evaluation.  The controller immediately moves
``/OmniverseKit_Persp``.  Isaac Sim normally creates that prim, but a missing
prim otherwise aborts environment construction before policy evaluation.

The guard below is installed after ``AppLauncher`` starts and before the
environment creates ``SimulationContext``.  It only defines the default
viewport camera when that exact prim is absent; existing cameras and all
non-default camera paths are untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


DEFAULT_VIEWPORT_CAMERA_PRIM_PATH = "/OmniverseKit_Persp"
HEADLESS_VIEWPORT_RECOVERY_PROFILE = (
    "isaaclab2p3_define_missing_default_viewport_camera_v1"
)
_GUARD_MARKER = "_polaris_missing_viewport_camera_guard_v1"


def install_viewport_camera_guard(
    simulation_context_type: type[Any],
    *,
    stage_getter: Callable[[], Any],
    camera_definer: Callable[[Any, str], Any],
    emit: Callable[[str], None] = print,
) -> bool:
    """Install the missing-default-camera guard on a simulation context type.

    Returns ``True`` for the first installation and ``False`` when the same
    guard is already installed.  Dependencies are injected so the fail-closed
    behavior can be tested without importing Isaac Sim on the host.
    """

    original = simulation_context_type.set_camera_view
    if getattr(original, _GUARD_MARKER, False) is True:
        return False

    @wraps(original)
    def guarded_set_camera_view(
        self: Any,
        eye: Any,
        target: Any,
        camera_prim_path: str = DEFAULT_VIEWPORT_CAMERA_PRIM_PATH,
    ) -> Any:
        if camera_prim_path == DEFAULT_VIEWPORT_CAMERA_PRIM_PATH:
            stage = stage_getter()
            if stage is not None:
                prim = stage.GetPrimAtPath(camera_prim_path)
                if not prim.IsValid():
                    camera = camera_definer(stage, camera_prim_path)
                    defined_prim = camera.GetPrim()
                    if not defined_prim.IsValid():
                        raise RuntimeError(
                            "Failed to define the missing default viewport camera "
                            f"at {camera_prim_path!r}"
                        )
                    emit(
                        "POLARIS_HEADLESS_VIEWPORT_CAMERA_RECOVERY="
                        f"profile={HEADLESS_VIEWPORT_RECOVERY_PROFILE};"
                        f"prim_path={camera_prim_path}"
                    )
        return original(
            self,
            eye=eye,
            target=target,
            camera_prim_path=camera_prim_path,
        )

    setattr(guarded_set_camera_view, _GUARD_MARKER, True)
    simulation_context_type.set_camera_view = guarded_set_camera_view
    return True


def install_isaaclab_headless_viewport_camera_guard() -> bool:
    """Install the production guard using the active Isaac Sim stage."""

    import omni.usd
    from isaaclab.sim import SimulationContext
    from pxr import UsdGeom

    return install_viewport_camera_guard(
        SimulationContext,
        stage_getter=lambda: omni.usd.get_context().get_stage(),
        camera_definer=lambda stage, path: UsdGeom.Camera.Define(stage, path),
    )
