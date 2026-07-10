import tyro
import mediapy

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
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

    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.environments.droid_cfg import (
        DroidJointVelocityActionCfg,
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
        make_runtime_artifact,
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
        env_cfg.observations = DroidJointVelocityObservationCfg()
        configured_episode_length_seconds = configure_native_environment_timeout(
            env_cfg
        )
    elif eval_args.control_mode != "joint-position":
        raise ValueError(f"Unsupported control mode: {eval_args.control_mode}")
    env: ManagerBasedRLSplatEnv = gym.make(  # type: ignore[assignment]
        eval_args.environment, cfg=env_cfg
    )
    runtime_artifact = None
    environment_runtime_contract = None
    if eval_args.control_mode == "joint-velocity":
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
        env.close()
        simulation_app.close()
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)
    if eval_args.control_mode == "joint-velocity":
        policy_client.bind_evaluation_runtime(environment_runtime_contract)

    horizon = (
        PI05_DROID_NATIVE_EPISODE_STEPS
        if eval_args.control_mode == "joint-velocity"
        else env.max_episode_length
    )
    terminal_rollout = None
    while episode < rollouts:
        # Index the initial condition with the episode being started. The old
        # loop reset before incrementing ``episode``, repeating condition zero.
        obs, info = env.reset(object_positions=initial_conditions[episode])
        policy_client.reset()
        if eval_args.control_mode == "joint-velocity":
            policy_client.begin_rollout(env)
        video = []
        numerical_failure_reason = ""
        bar = tqdm.tqdm(total=horizon)
        print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")

        while bar.n < horizon:
            try:
                action, viz = policy_client.infer(
                    obs, language_instruction, return_viz=True
                )
            except (
                JointPositionObservationNumericalError,
                JointVelocityObservationNumericalError,
            ) as error:
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
                        render_every_step=eval_args.policy.render_every_step,
                        needs_next_policy_render=policy_client.rerender,
                    ),
                )
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
            terminal_rollout = policy_client.finish_rollout(env, info["rubric"])
        bar.close()

        if video:
            filename = run_folder / f"episode_{episode}.mp4"
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
                    terminal_rollout["rubric"]["success"]
                    if eval_args.control_mode == "joint-velocity"
                    else info["rubric"]["success"]
                )
            ),
            "progress": (
                0.0
                if numerical_failure_reason
                else (
                    terminal_rollout["rubric"]["progress"]
                    if eval_args.control_mode == "joint-velocity"
                    else info["rubric"]["progress"]
                )
            ),
            "numerical_failure": bool(numerical_failure_reason),
            "numerical_failure_reason": numerical_failure_reason,
        }
        episode_df = pd.concat(
            [episode_df, pd.DataFrame([episode_data])], ignore_index=True
        )
        episode_df.to_csv(csv_path, index=False)
        print(f"Episode {episode} finished. Episode length: {episode_length}")
        episode += 1

    env.close()
    if eval_args.control_mode == "joint-velocity":
        if (
            runtime_artifact is None
            or environment_runtime_contract is None
            or terminal_rollout is None
            or not eval_args.policy.trace_path
        ):
            raise RuntimeError("Native joint-velocity close evidence is incomplete")
        publish_immutable_json(
            Path(eval_args.lifecycle_ready_path),
            make_close_ready_artifact(
                runtime_artifact=runtime_artifact,
                runtime_path=Path(eval_args.runtime_contract_path),
                metrics_path=csv_path,
                trace_path=Path(eval_args.policy.trace_path),
                video_path=run_folder / "episode_0.mp4",
                environment_runtime_contract=environment_runtime_contract,
                terminal_rollout=terminal_rollout,
            ),
        )
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
