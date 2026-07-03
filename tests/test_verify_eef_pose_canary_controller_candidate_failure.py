from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load_script():
    path = SCRIPTS / "verify_eef_pose_canary_controller_candidate_failure.py"
    spec = importlib.util.spec_from_file_location("candidate_failure_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


verifier = _load_script()


class _Parser:
    def parse_args(self):
        return SimpleNamespace()


def test_failure_verifier_reports_independent_pass(monkeypatch, capsys):
    monkeypatch.setattr(verifier, "_parser", lambda: _Parser())
    monkeypatch.setattr(
        verifier.validator,
        "validate_failure",
        lambda _args: {
            "variant": "official_lap3b",
            "job_id": 123,
            "raw_result": {"sha256": "a" * 64},
            "joint_name": "panda_joint5",
            "policy_step": 117,
            "physics_substep": 6,
        },
    )
    assert verifier.main() == 0
    output = capsys.readouterr()
    assert "POLARIS_CONTROLLER_CANDIDATE_FAILURE_VERIFY_PASS=" in output.out
    assert output.err == ""


def test_failure_verifier_fails_closed_on_tampered_ring(monkeypatch, capsys):
    monkeypatch.setattr(verifier, "_parser", lambda: _Parser())

    def reject(_args):
        raise verifier.validator.CandidateArtifactValidationError(
            "failure trace apply-index ordering drift"
        )

    monkeypatch.setattr(verifier.validator, "validate_failure", reject)
    assert verifier.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "POLARIS_CONTROLLER_CANDIDATE_FAILURE_VERIFY_FAIL=" in output.err
    assert "failure trace apply-index ordering drift" in output.err
