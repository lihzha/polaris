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
    from polaris.environments.droid_cfg import EefPoseActionCfg
    from polaris.utils import load_eval_initial_conditions
    from polaris.policy import InferenceClient
    from polaris.policy.droid_jointpos_client import (
        JointPositionObservationNumericalError,
    )
    from polaris.robust_differential_ik import DifferentialIKNumericalError
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    if eval_args.control_mode == "eef-pose":
        # Action managers are constructed by gym.make, so select the controller
        # on the config before creating the environment.
        env_cfg.actions = EefPoseActionCfg()
    elif eval_args.control_mode != "joint-position":
        raise ValueError(f"Unsupported control mode: {eval_args.control_mode}")
    env: ManagerBasedRLSplatEnv = gym.make(  # type: ignore[assignment]
        eval_args.environment, cfg=env_cfg
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

    horizon = env.max_episode_length
    while episode < rollouts:
        # Index the initial condition with the episode being started. The old
        # loop reset before incrementing ``episode``, repeating condition zero.
        obs, info = env.reset(object_positions=initial_conditions[episode])
        policy_client.reset()
        video = []
        numerical_failure_reason = ""
        bar = tqdm.tqdm(total=horizon)
        print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")

        while bar.n < horizon:
            try:
                action, viz = policy_client.infer(
                    obs, language_instruction, return_viz=True
                )
            except JointPositionObservationNumericalError as error:
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
                    expensive=policy_client.rerender,
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
            bar.update(1)
            if term[0] or trunc[0]:
                break

        episode_length = bar.n
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
                False if numerical_failure_reason else info["rubric"]["success"]
            ),
            "progress": (
                0.0 if numerical_failure_reason else info["rubric"]["progress"]
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
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
