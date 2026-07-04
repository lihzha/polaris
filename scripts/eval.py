import tyro
import mediapy

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
import os
import pandas as pd


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs


def main(eval_args: EvalArgs):
    from polaris.pi05_droid_jointvelocity_contract import (
        NATIVE_GRIPPER_DRIVE_PROFILE,
    )

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
    if (
        eval_args.policy.client == "DroidJointVelocity"
        and eval_args.control_mode != "joint-velocity"
    ):
        raise ValueError("DroidJointVelocity requires --control-mode joint-velocity")
    if (
        eval_args.control_mode == "joint-velocity"
        and eval_args.policy.client != "DroidJointVelocity"
    ):
        raise ValueError(
            "joint-velocity control mode is reserved for DroidJointVelocity"
        )
    if eval_args.policy.client == "DroidJointVelocity" and (
        not eval_args.runtime_contract_path
        or not eval_args.lifecycle_ready_path
        or not eval_args.policy.trace_path
    ):
        raise ValueError(
            "DroidJointVelocity requires --runtime-contract-path and "
            "--lifecycle-ready-path and --policy.trace-path"
        )
    if eval_args.control_mode == "joint-velocity":
        if eval_args.expected_gripper_drive_profile != NATIVE_GRIPPER_DRIVE_PROFILE:
            raise ValueError("joint-velocity expected gripper drive profile mismatch")
    elif eval_args.expected_gripper_drive_profile is not None:
        raise ValueError(
            "expected gripper drive profile is valid only for joint-velocity control"
        )

    # This must be done before importing anything from IsaacLab
    # Inside main function to avoid launching IsaacLab in global scope
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    from polaris.pi05_droid_native_lifecycle import NativeEvaluatorLifecycle

    lifecycle = NativeEvaluatorLifecycle(simulation_app)
    try:
        return _run_evaluation(eval_args, lifecycle)
    finally:
        lifecycle.close()


def _run_evaluation(eval_args: EvalArgs, lifecycle):
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.environments.droid_cfg import (
        DroidJointVelocityActionCfg,
        DroidJointVelocityEventCfg,
        DroidJointVelocityObservationCfg,
        EefPoseActionCfg,
    )
    from polaris.environments.robot_cfg import NVIDIA_DROID_JOINT_VELOCITY
    from polaris.joint_velocity_runtime import (
        print_joint_velocity_runtime,
        validate_joint_velocity_runtime,
    )
    from polaris.pi05_droid_native_eval_contract import (
        PI05_DROID_NATIVE_EPISODE_STEPS,
        configure_native_environment_timeout,
        make_environment_runtime_contract,
        make_close_ready_artifact,
        make_episode_sidecar,
        make_runtime_artifact,
        publish_immutable_file_from_temporary,
        publish_immutable_json,
        should_render_expensive,
    )
    from polaris.utils import load_eval_initial_conditions
    from polaris.policy import InferenceClient
    from polaris.policy.droid_jointpos_client import (
        JointPositionObservationNumericalError,
    )
    from polaris.policy.droid_jointvelocity_client import (
        JointVelocityObservationNumericalError,
    )
    from polaris.robust_differential_ik import DifferentialIKNumericalError
    from polaris.native_gripper_runtime import NativeAllJointVelocityLimitError
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    configured_episode_length_seconds = None
    if eval_args.control_mode == "eef-pose":
        # Action managers are constructed by gym.make, so select the controller
        # on the config before creating the environment.
        env_cfg.actions = EefPoseActionCfg()
    elif eval_args.control_mode == "joint-velocity":
        env_cfg.scene.robot = NVIDIA_DROID_JOINT_VELOCITY.copy()
        env_cfg.actions = DroidJointVelocityActionCfg()
        env_cfg.events = DroidJointVelocityEventCfg()
        env_cfg.observations = DroidJointVelocityObservationCfg()
        configured_episode_length_seconds = configure_native_environment_timeout(
            env_cfg
        )
    elif eval_args.control_mode != "joint-position":
        raise ValueError(f"Unsupported control mode: {eval_args.control_mode}")
    env: ManagerBasedRLSplatEnv = gym.make(  # type: ignore[assignment]
        eval_args.environment, cfg=env_cfg
    )
    lifecycle.bind_environment(env)
    runtime_artifact = None
    environment_runtime_contract = None
    if eval_args.control_mode == "joint-velocity":
        # Gym construction initializes physics but does not execute reset-mode
        # events.  Perform one native reset before accepting any live drive
        # contract; the rollout reset below reapplies and recounts the cap.
        env.reset(expensive=False)
        environment_runtime_contract = make_environment_runtime_contract(
            configured_episode_length_seconds=configured_episode_length_seconds,
            live_max_episode_length=env.max_episode_length,
        )
        runtime_contract = validate_joint_velocity_runtime(
            env,
            expected_gripper_drive_profile=eval_args.expected_gripper_drive_profile,
        )
        print_joint_velocity_runtime(runtime_contract)
        runtime_artifact = publish_immutable_json(
            Path(eval_args.runtime_contract_path),
            make_runtime_artifact(runtime_contract, environment_runtime_contract),
        )

    default_instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=eval_args.initial_conditions_file,
        rollouts=eval_args.rollouts,
    )
    language_instruction = (
        eval_args.instruction
        if eval_args.instruction is not None
        else default_instruction
    )
    rollouts = len(initial_conditions)
    # Resume CSV logging
    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / "eval_results.csv"
    incident_path = run_folder / "native_failures" / "episode_000000.json"
    sidecar_path = run_folder / "native_runtime" / "episode_000000.json"
    if csv_path.exists():
        episode_df = pd.read_csv(csv_path)
    else:
        episode_df = pd.DataFrame(
            {
                "episode": pd.Series(dtype="int"),
                "episode_length": pd.Series(dtype="int"),
                "success": pd.Series(dtype="bool"),
                "progress": pd.Series(dtype="float"),
                "numerical_failure": pd.Series(dtype="bool"),
                "numerical_failure_reason": pd.Series(dtype="str"),
            }
        )
    if "numerical_failure" not in episode_df:
        episode_df["numerical_failure"] = False
    if "numerical_failure_reason" not in episode_df:
        episode_df["numerical_failure_reason"] = ""
    episode = len(episode_df)
    if episode >= rollouts:
        print("All rollouts have been evaluated. Exiting.")
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)
    if eval_args.control_mode == "joint-velocity":
        policy_client.bind_evaluation_runtime(environment_runtime_contract)

    horizon = (
        PI05_DROID_NATIVE_EPISODE_STEPS
        if eval_args.control_mode == "joint-velocity"
        else env.max_episode_length
    )
    terminal_outcome = None
    episode_sidecar_artifact = None
    while episode < rollouts:
        # Index the initial condition with the episode being started. The old
        # loop reset before incrementing ``episode``, repeating condition zero.
        obs, info = env.reset(object_positions=initial_conditions[episode])
        policy_client.reset()
        native_arm_term = None
        if eval_args.control_mode == "joint-velocity":
            policy_client.begin_rollout(env)
            native_arm_term = getattr(env, "unwrapped", env).action_manager._terms[
                "arm"
            ]
            native_arm_term.bind_native_all_joint_failure_path(incident_path)
        video = []
        numerical_failure_reason = ""
        incident_artifact = None
        dynamic_report = None
        bar = tqdm.tqdm(total=horizon)
        print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")

        def finalize_native_velocity_failure(error):
            if native_arm_term is None:
                raise RuntimeError("Native velocity failure has no audited action term")
            report = native_arm_term.native_all_joint_dynamic_report(
                include_samples=False
            )
            terminal = policy_client.record_execution_failure(error, env, report)
            return (
                f"{type(error).__name__}: {error}",
                report,
                terminal,
                error.incident_artifact,
            )

        while bar.n < horizon:
            try:
                action, viz = policy_client.infer(
                    obs, language_instruction, return_viz=True
                )
            except (
                JointPositionObservationNumericalError,
                JointVelocityObservationNumericalError,
            ) as error:
                if eval_args.control_mode == "joint-velocity" and isinstance(
                    error, JointVelocityObservationNumericalError
                ):
                    # The native canary has one typed monitor terminal form.
                    # Observation/schema failures remain fatal contract errors.
                    raise
                numerical_failure_reason = f"{type(error).__name__}: {error}"
                print(
                    f"Numerical failure in episode {episode} before action "
                    f"{bar.n}: {numerical_failure_reason}"
                )
                break
            if viz is not None:
                # Request visualization every step so saved videos are complete,
                # even when policy inference itself is open-loop chunked.
                video.append(viz)
            try:
                obs, rew, term, trunc, info = env.step(
                    torch.as_tensor(action, device=env.device).reshape(1, -1),
                    expensive=should_render_expensive(
                        policy_client_name=eval_args.policy.client,
                        render_every_step=eval_args.policy.render_every_step,
                        needs_next_policy_render=policy_client.rerender,
                    ),
                )
            except NativeAllJointVelocityLimitError as error:
                if eval_args.control_mode != "joint-velocity":
                    raise
                (
                    numerical_failure_reason,
                    dynamic_report,
                    terminal_outcome,
                    incident_artifact,
                ) = finalize_native_velocity_failure(error)
                print(
                    f"Numerical failure in episode {episode} at action "
                    f"{bar.n}: {numerical_failure_reason}"
                )
                bar.update(1)
                break
            except (DifferentialIKNumericalError, torch.linalg.LinAlgError) as error:
                numerical_failure_reason = f"{type(error).__name__}: {error}"
                print(
                    f"Numerical failure in episode {episode} at action "
                    f"{bar.n}: {numerical_failure_reason}"
                )
                # The policy visualization was already recorded for this
                # attempted action, so include it in the episode length.
                bar.update(1)
                break
            if eval_args.control_mode == "joint-velocity":
                try:
                    native_arm_term.record_native_all_joint_post_policy_step()
                except NativeAllJointVelocityLimitError as error:
                    # env.step completed all eight physics substeps. Persist its
                    # execution record before closing the failing boundary sample.
                    policy_client.record_execution(
                        obs,
                        env,
                        terminated=term,
                        truncated=trunc,
                    )
                    (
                        numerical_failure_reason,
                        dynamic_report,
                        terminal_outcome,
                        incident_artifact,
                    ) = finalize_native_velocity_failure(error)
                    print(
                        f"Numerical failure in episode {episode} at completed action "
                        f"{bar.n}: {numerical_failure_reason}"
                    )
                    bar.update(1)
                    break
                policy_client.record_execution(
                    obs,
                    env,
                    terminated=term,
                    truncated=trunc,
                )
            bar.update(1)
            if eval_args.control_mode != "joint-velocity" and (term[0] or trunc[0]):
                break

        episode_length = bar.n
        if (
            eval_args.control_mode == "joint-velocity"
            and not numerical_failure_reason
            and episode_length == PI05_DROID_NATIVE_EPISODE_STEPS
        ):
            terminal_outcome = policy_client.finish_rollout(env, info["rubric"])
            dynamic_report = (
                getattr(env, "unwrapped", env)
                .action_manager._terms["arm"]
                .native_all_joint_dynamic_report(include_samples=False)
            )
        bar.close()

        video_artifact = None
        if video:
            filename = run_folder / f"episode_{episode}.mp4"
            if eval_args.control_mode == "joint-velocity":
                temporary_video = filename.with_name(
                    f".{filename.stem}.partial-{os.getpid()}.mp4"
                )
                try:
                    mediapy.write_video(temporary_video, video, fps=15)
                    video_artifact = publish_immutable_file_from_temporary(
                        temporary_video, filename
                    )
                finally:
                    if temporary_video.exists() and not temporary_video.is_symlink():
                        temporary_video.unlink()
            else:
                mediapy.write_video(filename, video, fps=15)
        else:
            print(
                f"Warning: policy returned no visualization for episode {episode}; "
                "no video was written."
            )

        episode_data = {
            "episode": episode,
            "episode_length": episode_length,
            "success": (
                False
                if numerical_failure_reason
                else (
                    terminal_outcome["rubric"]["success"]
                    if eval_args.control_mode == "joint-velocity"
                    else info["rubric"]["success"]
                )
            ),
            "progress": (
                0.0
                if numerical_failure_reason
                else (
                    terminal_outcome["rubric"]["progress"]
                    if eval_args.control_mode == "joint-velocity"
                    else info["rubric"]["progress"]
                )
            ),
            "numerical_failure": bool(numerical_failure_reason),
            "numerical_failure_reason": numerical_failure_reason,
        }
        if eval_args.control_mode == "joint-velocity":
            trace_artifact = policy_client.finalized_trace_artifact
            if (
                terminal_outcome is None
                or dynamic_report is None
                or trace_artifact is None
                or video_artifact is None
            ):
                raise RuntimeError(
                    "Native joint-velocity episode transaction is incomplete"
                )
            sidecar_artifact = publish_immutable_json(
                sidecar_path,
                make_episode_sidecar(
                    episode_result=episode_data,
                    terminal_outcome=terminal_outcome,
                    environment_runtime_contract=environment_runtime_contract,
                    dynamic_report=dynamic_report,
                    trace_artifact=trace_artifact,
                    video_artifact=video_artifact,
                    incident_artifact=incident_artifact,
                ),
            )
            episode_sidecar_artifact = {
                key: sidecar_artifact[key]
                for key in ("path", "size", "sha256", "mode", "nlink")
            }
        episode_df = pd.concat(
            [episode_df, pd.DataFrame([episode_data])], ignore_index=True
        )
        if eval_args.control_mode == "joint-velocity":
            temporary_csv = csv_path.with_name(
                f".{csv_path.name}.partial-{os.getpid()}"
            )
            try:
                episode_df.to_csv(temporary_csv, index=False)
                publish_immutable_file_from_temporary(temporary_csv, csv_path)
            finally:
                if temporary_csv.exists() and not temporary_csv.is_symlink():
                    temporary_csv.unlink()
        else:
            episode_df.to_csv(csv_path, index=False)
        print(f"Episode {episode} finished. Episode length: {episode_length}")
        episode += 1

    if eval_args.control_mode == "joint-velocity":
        if (
            runtime_artifact is None
            or environment_runtime_contract is None
            or terminal_outcome is None
            or episode_sidecar_artifact is None
            or not eval_args.policy.trace_path
        ):
            raise RuntimeError("Native joint-velocity close evidence is incomplete")
        lifecycle.prepare_close_ready(
            publish_immutable_json,
            Path(eval_args.lifecycle_ready_path),
            make_close_ready_artifact(
                runtime_artifact=runtime_artifact,
                runtime_path=Path(eval_args.runtime_contract_path),
                metrics_path=csv_path,
                trace_path=Path(eval_args.policy.trace_path),
                video_path=run_folder / "episode_0.mp4",
                environment_runtime_contract=environment_runtime_contract,
                terminal_outcome=terminal_outcome,
                episode_sidecar=episode_sidecar_artifact,
            ),
        )


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
