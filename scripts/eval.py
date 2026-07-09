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
from polaris.evaluation_seed import (
    episode_environment_seed,
    validate_episode_seed_range,
)


def _print_eval_phase(phase: str) -> None:
    """Publish one flushed evaluator phase boundary for startup diagnosis."""

    print(f"POLARIS_EVAL_PHASE={phase}", flush=True)


def main(eval_args: EvalArgs):
    if eval_args.policy.client == "DroidJointPos":
        if eval_args.environment_seed is None:
            raise ValueError("DroidJointPos requires --environment-seed")
        if eval_args.rollouts is not None:
            validate_episode_seed_range(eval_args.environment_seed, eval_args.rollouts)
        if not eval_args.runtime_contract_path or not eval_args.policy.trace_path:
            raise ValueError(
                "DroidJointPos requires --runtime-contract-path and --policy.trace-path"
            )
    elif eval_args.environment_seed is not None:
        episode_environment_seed(eval_args.environment_seed, 0)
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
    _print_eval_phase("before_app_launcher")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    _print_eval_phase("after_app_launcher")
    # >>>> Isaac Sim App Launcher <<<<

    _print_eval_phase("before_evaluation_imports")
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.environments.droid_cfg import EefPoseActionCfg
    from polaris.environments.pi05_droid_jointpos_cfg import (
        DroidJointPositionActionCfg,
        DroidJointPositionObservationCfg,
    )
    from polaris.evaluation_seed import (
        bind_environment_seed,
        format_environment_seed_contract,
        make_live_environment_seed_contract,
    )
    from polaris.pi05_droid_jointpos_runtime import (
        PI05_DROID_JOINTPOS_OUTER_STEPS,
        capture_jointpos_runtime,
        configure_jointpos_timeout,
        format_jointpos_runtime,
        publish_jointpos_runtime,
    )
    from polaris.pi05_droid_jointpos_image_contract import (
        install_jointpos_image_instrumentation,
    )
    from polaris.utils import load_eval_initial_conditions
    from polaris.policy import InferenceClient
    from polaris.policy.droid_jointpos_client import (
        JointPositionObservationNumericalError,
    )
    from polaris.robust_differential_ik import DifferentialIKNumericalError
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    _print_eval_phase("after_evaluation_imports")
    audited_jointpos = eval_args.policy.client == "DroidJointPos"
    _print_eval_phase("before_parse_env_cfg")
    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    _print_eval_phase("after_parse_env_cfg")
    if eval_args.environment_seed is not None:
        bind_environment_seed(env_cfg, eval_args.environment_seed)
    if eval_args.control_mode == "eef-pose":
        # Action managers are constructed by gym.make, so select the controller
        # on the config before creating the environment.
        env_cfg.actions = EefPoseActionCfg()
    elif audited_jointpos:
        env_cfg.actions = DroidJointPositionActionCfg()
        env_cfg.observations = DroidJointPositionObservationCfg()
        configure_jointpos_timeout(env_cfg)
    elif eval_args.control_mode != "joint-position":
        raise ValueError(f"Unsupported control mode: {eval_args.control_mode}")
    _print_eval_phase("before_gym_make")
    env: ManagerBasedRLSplatEnv = gym.make(  # type: ignore[assignment]
        eval_args.environment, cfg=env_cfg
    )
    _print_eval_phase("after_gym_make")
    if audited_jointpos:
        _print_eval_phase("before_jointpos_image_instrumentation")
        install_jointpos_image_instrumentation(env)
        _print_eval_phase("after_jointpos_image_instrumentation")
    environment_seed_contract = None
    if eval_args.environment_seed is not None:
        environment_seed_contract = make_live_environment_seed_contract(
            env, eval_args.environment_seed
        )
        print(
            format_environment_seed_contract(environment_seed_contract),
            flush=True,
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
    if audited_jointpos:
        validate_episode_seed_range(eval_args.environment_seed, rollouts)
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

    _print_eval_phase(f"before_policy_client_load:{eval_args.policy.client}")
    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)
    _print_eval_phase(f"after_policy_client_load:{eval_args.policy.client}")
    if audited_jointpos:
        if environment_seed_contract is None:
            raise RuntimeError("DroidJointPos environment seed contract is missing")
        policy_client.bind_environment_seed_contract(environment_seed_contract)

    horizon = (
        PI05_DROID_JOINTPOS_OUTER_STEPS if audited_jointpos else env.max_episode_length
    )
    jointpos_runtime_sha256 = None
    while episode < rollouts:
        # Index the initial condition with the episode being started. The old
        # loop reset before incrementing ``episode``, repeating condition zero.
        episode_seed = None
        if eval_args.environment_seed is not None:
            episode_seed = episode_environment_seed(eval_args.environment_seed, episode)
        _print_eval_phase(f"before_rollout_reset:{episode}")
        obs, info = env.reset(
            seed=episode_seed,
            object_positions=initial_conditions[episode],
        )
        _print_eval_phase(f"after_rollout_reset:{episode}")
        if audited_jointpos:
            _print_eval_phase(f"before_jointpos_runtime_capture:{episode}")
            live_jointpos_runtime = capture_jointpos_runtime(env, obs)
            _print_eval_phase(f"after_jointpos_runtime_capture:{episode}")
            if jointpos_runtime_sha256 is None:
                print(format_jointpos_runtime(live_jointpos_runtime), flush=True)
                publish_jointpos_runtime(
                    Path(eval_args.runtime_contract_path), live_jointpos_runtime
                )
                jointpos_runtime_sha256 = live_jointpos_runtime["runtime_sha256"]
            elif live_jointpos_runtime["runtime_sha256"] != jointpos_runtime_sha256:
                raise ValueError(
                    "live joint-position runtime drifted across episode reset"
                )
            policy_client.bind_jointpos_runtime(live_jointpos_runtime)
            policy_client.reset(
                episode_index=episode,
                episode_seed=episode_seed,
            )
            policy_client.begin_rollout(env)
        else:
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
                    expensive=(
                        eval_args.policy.render_every_step or policy_client.rerender
                        if audited_jointpos
                        else policy_client.rerender
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
            if audited_jointpos:
                policy_client.record_execution(
                    obs,
                    env,
                    terminated=term,
                    truncated=trunc,
                    terminal_rubric=(
                        info["rubric"]
                        if bar.n + 1 == PI05_DROID_JOINTPOS_OUTER_STEPS
                        else None
                    ),
                )
            bar.update(1)
            if not audited_jointpos and (term[0] or trunc[0]):
                break

        episode_length = bar.n
        if audited_jointpos and episode_length != PI05_DROID_JOINTPOS_OUTER_STEPS:
            raise RuntimeError(
                "DroidJointPos did not complete the explicit 450-step horizon"
            )
        if audited_jointpos:
            from PIL import Image

            terminal_visualization = policy_client.final_terminal_visualization()
            terminal_path = run_folder / f"episode_{episode}_terminal.png"
            terminal_temporary = terminal_path.with_name(
                f".{terminal_path.stem}.partial-{os.getpid()}.png"
            )
            if (
                terminal_path.exists()
                or terminal_path.is_symlink()
                or terminal_temporary.exists()
                or terminal_temporary.is_symlink()
            ):
                raise RuntimeError(
                    "DroidJointPos terminal visualization already exists"
                )
            try:
                with terminal_temporary.open("xb") as terminal_file:
                    Image.fromarray(terminal_visualization).save(
                        terminal_file, format="PNG"
                    )
                    terminal_file.flush()
                    os.fsync(terminal_file.fileno())
                os.link(terminal_temporary, terminal_path, follow_symlinks=False)
                terminal_temporary.unlink()
                directory = os.open(run_folder, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if terminal_temporary.exists() and not terminal_temporary.is_symlink():
                    terminal_temporary.unlink()
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
