"""
Lightweight config dataclasses for evaluation.
No heavy dependencies - safe to import anywhere.
"""

from dataclasses import dataclass
from typing import Literal


LAP_EEF_FRAME = "panda_link8"


@dataclass
class PolicyServer:
    """
    Configuration for a policy server to co-launch.

    Use {port} placeholder in command - it will be replaced with an auto-assigned free port.
    Jobs using this server will automatically have their policy.port updated.

    Example:
        PolicyServer(
            name="pi0",
            command="CUDA_VISIBLE_DEVICES=0 python serve_policy.py --port {port}",
        )
    """

    name: str  # Friendly name for logging (also used to match jobs to servers)
    command: str  # Shell command with {port} placeholder
    ready_message: str = (
        "Application startup complete"  # Message indicating server is ready
    )

    # Runtime-assigned (don't set manually)
    _assigned_port: int | None = None


@dataclass
class PolicyArgs:
    """Policy configuration."""

    # name: str                              # Policy name (pi05_droid_jointpos, pi0_fast_droid_jointpos, etc.)
    client: str = "DroidJointPos"  # Client name (DroidJointPos, Fake, etc.)
    host: str = "0.0.0.0"
    port: int = 8000
    # Ego-LAP values below are assertions against authoritative server metadata.
    # When omitted, the client derives them from the validated serving contract.
    open_loop_horizon: int | None = 8
    frame_description: str | None = None
    action_frame: Literal["robot_base"] | None = None
    dataset_name: str = "droid"
    state_type: str = "eef_pose"
    eef_frame: Literal["panda_link8"] = LAP_EEF_FRAME
    rotate_wrist_180: bool | None = None
    render_every_step: bool = True
    trace_dir: str | None = None
    trace_path: str | None = None
    ar_interpolation_steps: int = 8
    contract_output: str | None = None
    checkpoint_profile: str | None = None
    checkpoint_path: str | None = None
    policy_type: Literal["flow", "ar"] | None = None
    normalization_scope: Literal["global", "category"] | None = None
    normalization_stats_sha256: str | None = None
    normalization_profile: str | None = None
    normalization_input_formula: str | None = None
    normalization_output_formula: str | None = None


@dataclass
class EvalArgs:
    """Evaluation configuration."""

    policy: PolicyArgs  # Policy arguments
    environment: str  # Which IsaacLab environment to use
    run_folder: str  # Path to run folder
    headless: bool = True  # Whether to run in headless mode
    initial_conditions_file: str | None = None  # Path to initial conditions file
    instruction: str | None = None  # Override language instruction
    rollouts: int | None = None  # Number of rollouts to evaluate
    control_mode: Literal["joint-position", "eef-pose"] = "joint-position"
    runtime_contract_output: str | None = None


def validate_policy_control_mode(eval_args: EvalArgs) -> None:
    """Fail before Isaac launch when a policy and controller are incompatible."""

    if (
        eval_args.policy.client == "EgoLAPEefPose"
        and eval_args.control_mode != "eef-pose"
    ):
        raise ValueError("EgoLAPEefPose requires --control-mode eef-pose")
    if (
        eval_args.policy.client == "DroidJointPos"
        and eval_args.control_mode != "joint-position"
    ):
        raise ValueError("DroidJointPos requires --control-mode joint-position")


@dataclass
class JobCfg:
    """A single evaluation job in a batch."""

    eval_args: EvalArgs
    server: PolicyServer | None = None  # Server to co-launch for this job


@dataclass
class BatchConfig:
    """Batch evaluation configuration."""

    jobs: list[JobCfg]

    # @staticmethod # let users do this on their own if they want
    # def sweep(**kwargs: list[Any]) -> list[dict[str, Any]]:
    #     """
    #     Helper to generate grid of configs from lists of values.

    #     Example:
    #         BatchConfig.sweep(
    #             usd=["env1.usd", "env2.usd"],
    #             policy=["pi0", "pi05"],
    #         )
    #         # Returns 4 dicts: all combinations
    #     """
    #     keys = list(kwargs.keys())
    #     values = [v if isinstance(v, list) else [v] for v in kwargs.values()]
    #     return [dict(zip(keys, combo)) for combo in itertools.product(*values)]
