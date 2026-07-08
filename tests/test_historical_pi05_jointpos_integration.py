from pathlib import Path

from polaris.config import EvalArgs


ROOT = Path(__file__).resolve().parents[1]


def test_historical_evaluator_ports_only_attested_jointpos_lifecycle():
    source = (ROOT / "scripts/eval.py").read_text(encoding="utf-8")

    assert "DroidJointPositionActionCfg" in source
    assert "DroidJointPositionObservationCfg" in source
    assert "configure_jointpos_timeout(env_cfg)" in source
    assert "capture_jointpos_runtime(env, obs)" in source
    assert "publish_jointpos_runtime(" in source
    assert "policy_client.bind_jointpos_runtime(live_jointpos_runtime)" in source
    assert "policy_client.begin_rollout(env)" in source
    assert "PI05_DROID_JOINTPOS_OUTER_STEPS" in source
    assert "eval_args.policy.render_every_step or policy_client.rerender" in source
    assert "policy_client.record_execution(" in source
    assert "policy_client.final_terminal_visualization()" in source
    assert "episode_{episode}_terminal.png" in source

    # This control intentionally retains its pre-native-evaluator lifecycle.
    assert "NativeEvaluatorLifecycle" not in source
    assert "DroidJointVelocity" not in source
    assert "env.close()" in source
    assert "simulation_app.close()" in source


def test_historical_eval_config_exposes_jointpos_runtime_artifact_only():
    fields = EvalArgs.__dataclass_fields__
    assert fields["runtime_contract_path"].default is None
    assert "lifecycle_ready_path" not in fields
    assert "expected_gripper_drive_profile" not in fields
