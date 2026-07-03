import tyro

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
import pandas as pd


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import LAP_EEF_FRAME, EvalArgs, validate_policy_control_mode
from polaris.eef_runtime_contract import atomic_write_runtime_contract
from polaris.eef_runtime_contract import atomic_write_episode_safety
from polaris.eef_runtime_contract import aggregate_episode_safety
from polaris.eef_runtime_contract import begin_eef_safety_episode
from polaris.eef_runtime_contract import build_terminal_rollout_evidence
from polaris.eef_runtime_contract import capture_eef_environment_state
from polaris.eef_runtime_contract import configure_ego_lap_environment_timeout
from polaris.eef_runtime_contract import eef_episode_safety_report
from polaris.eef_runtime_contract import load_episode_safety_sidecars
from polaris.eef_runtime_contract import reconcile_episode_safety_transactions
from polaris.eef_runtime_contract import validate_eef_runtime_frame
from polaris.eef_runtime_contract import validate_eef_runtime_safety
from polaris.eef_runtime_contract import validate_ego_lap_runtime_protocol
from polaris.eef_runtime_contract import validate_eef_outer_step_transition
from polaris.eval_artifacts import atomic_write_episode_video
from polaris.eval_artifacts import atomic_write_results
from polaris.eval_artifacts import build_episode_artifact_identity
from polaris.eval_artifacts import load_resume_results
from polaris.eef_gripper_runtime import install_eef_gripper_runtime
from polaris.eef_gripper_runtime import record_eef_gripper_post_policy_step
from polaris.eef_gripper_runtime import validate_eef_gripper_post_reset


def main(eval_args: EvalArgs):
    validate_policy_control_mode(eval_args)
    is_ego_lap = eval_args.policy.client == "EgoLAPEefPose"

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
    from polaris.environments.robot_cfg import configure_eef_pose_joint_safety
    from polaris.utils import load_eval_initial_conditions
    from polaris.policy import InferenceClient
    from polaris.robust_differential_ik import DifferentialIKNumericalError
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    if is_ego_lap:
        configure_ego_lap_environment_timeout(env_cfg)
    if eval_args.control_mode == "eef-pose":
        # Action managers are constructed by gym.make, so select the controller
        # on the config before creating the environment.
        env_cfg.actions = EefPoseActionCfg()
        configure_eef_pose_joint_safety(
            env_cfg.scene.robot,
            physx_cfg=env_cfg.sim.physx,
            enable_gripper_velocity_limit=is_ego_lap,
        )
        frame_cfg = env_cfg.scene.lap_ee_frame
        target_cfg = frame_cfg.target_frames[0]
        observed_source = frame_cfg.prim_path
        observed_prim = target_cfg.prim_path
        controlled_body = env_cfg.actions.arm.body_name
        source_offset_is_identity = tuple(frame_cfg.source_frame_offset.pos) == (
            0.0,
            0.0,
            0.0,
        ) and tuple(frame_cfg.source_frame_offset.rot) == (1.0, 0.0, 0.0, 0.0)
        target_offset_is_identity = tuple(target_cfg.offset.pos) == (
            0.0,
            0.0,
            0.0,
        ) and tuple(target_cfg.offset.rot) == (1.0, 0.0, 0.0, 0.0)
        control_offset = env_cfg.actions.arm.body_offset
        control_offset_is_identity = control_offset is not None and (
            tuple(control_offset.pos) == (0.0, 0.0, 0.0)
            and tuple(control_offset.rot) == (1.0, 0.0, 0.0, 0.0)
        )
        if not observed_source.endswith("/robot/panda_link0"):
            raise ValueError(
                "EEF observation source does not match the DROID/LAP contract: "
                f"{observed_source!r}"
            )
        if not observed_prim.endswith(f"/robot/{LAP_EEF_FRAME}"):
            raise ValueError(
                "EEF observation frame does not match the DROID/LAP contract: "
                f"{observed_prim!r}"
            )
        if controlled_body != LAP_EEF_FRAME:
            raise ValueError(
                "EEF controller frame does not match the DROID/LAP contract: "
                f"{controlled_body!r}"
            )
        if not source_offset_is_identity or not target_offset_is_identity:
            raise ValueError("EEF observation source/target offsets must be identity")
        if not control_offset_is_identity:
            raise ValueError("EEF controller body offset must be identity")
        print(
            "POLARIS_LAP_EEF_FRAME="
            f"{LAP_EEF_FRAME};reference={observed_source};"
            f"observation={observed_prim};frame_offsets=identity;"
            f"control={controlled_body};control_offset=identity",
            flush=True,
        )
    elif eval_args.control_mode != "joint-position":
        raise ValueError(f"Unsupported control mode: {eval_args.control_mode}")
    robot_usd_path = Path(env_cfg.scene.robot.spawn.usd_path)
    env: ManagerBasedRLSplatEnv = gym.make(  # type: ignore[assignment]
        eval_args.environment, cfg=env_cfg
    )
    runtime_protocol = None
    if is_ego_lap:
        runtime_protocol = validate_ego_lap_runtime_protocol(env)
        print(
            "POLARIS_LAP_RUNTIME_PROTOCOL="
            f"profile={runtime_protocol['profile']};"
            f"outer_steps={runtime_protocol['episode_steps']};"
            f"internal_steps={runtime_protocol['live_max_episode_length']};"
            f"autoreset_margin={runtime_protocol['autoreset_margin_steps']};"
            f"policy_hz={runtime_protocol['policy_hz']};"
            f"step_dt={runtime_protocol['step_dt']};"
            f"physics_hz={runtime_protocol['physics_hz']};"
            f"physics_dt={runtime_protocol['physics_dt']};"
            f"decimation={runtime_protocol['decimation']}",
            flush=True,
        )
    evaluation_horizon = (
        int(runtime_protocol["episode_steps"])
        if runtime_protocol is not None
        else int(env.max_episode_length)
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
    if is_ego_lap and eval_args.runtime_contract_output is None:
        eval_args.runtime_contract_output = str(
            run_folder / "polaris_runtime_contract.json"
        )
    if is_ego_lap and eval_args.policy.contract_output is None:
        eval_args.policy.contract_output = str(
            run_folder / "ego_lap_serving_contract.json"
        )
    if (
        is_ego_lap
        and eval_args.policy.trace_dir is None
        and eval_args.policy.trace_path is None
    ):
        eval_args.policy.trace_dir = str(run_folder / "policy_traces")
    if is_ego_lap and (
        eval_args.policy.trace_dir is None or eval_args.policy.trace_path is not None
    ):
        raise ValueError(
            "Transactional Ego-LAP evaluation requires trace_dir with one "
            "immutable finalized trace per episode"
        )
    if (
        is_ego_lap
        and Path(eval_args.policy.trace_dir).resolve()
        != (run_folder / "policy_traces").resolve()
    ):
        raise ValueError(
            "Transactional Ego-LAP evaluation requires trace_dir inside the "
            "run folder at policy_traces/"
        )
    csv_path = run_folder / "eval_results.csv"
    safety_dir = run_folder / "ik_safety"
    episode_df = load_resume_results(
        csv_path,
        run_folder=run_folder,
        expected_rollouts=rollouts,
        expected_horizon=evaluation_horizon,
        require_episode_artifacts=is_ego_lap,
        trace_dir=(
            Path(eval_args.policy.trace_dir)
            if eval_args.policy.trace_dir is not None
            else None
        ),
        trace_path=(
            Path(eval_args.policy.trace_path)
            if eval_args.policy.trace_path is not None
            else None
        ),
    )
    if is_ego_lap:
        trace_dir = Path(eval_args.policy.trace_dir)
        episode_df, recovered_row = reconcile_episode_safety_transactions(
            episode_df,
            directory=safety_dir,
            run_folder=run_folder,
            trace_dir=trace_dir,
            expected_rollouts=rollouts,
            expected_horizon=evaluation_horizon,
        )
        if recovered_row:
            atomic_write_results(episode_df, csv_path)
            print(
                "Recovered the next CSV row from its immutable episode "
                "safety transaction.",
                flush=True,
            )
        episode_df = load_resume_results(
            csv_path,
            run_folder=run_folder,
            expected_rollouts=rollouts,
            expected_horizon=evaluation_horizon,
            require_episode_artifacts=True,
            trace_dir=trace_dir,
        )
    episode = len(episode_df)
    policy_client: InferenceClient | None = None
    if is_ego_lap:
        # Validate and persist the live serving contract even when a resumed task
        # already contains every rollout artifact.
        policy_client = InferenceClient.get_client(eval_args.policy)
        policy_client.bind_runtime_contract(runtime_protocol)

    runtime_frame_evidence = None
    gripper_runtime_contract = None

    def install_or_validate_gripper_runtime():
        nonlocal gripper_runtime_contract
        if gripper_runtime_contract is None:
            gripper_runtime_contract = install_eef_gripper_runtime(
                env, robot_usd_path=robot_usd_path
            )
        else:
            validate_eef_gripper_post_reset(env, gripper_runtime_contract)

    def persist_runtime_contract():
        if (
            runtime_protocol is None
            or runtime_frame_evidence is None
            or eval_args.runtime_contract_output is None
        ):
            raise RuntimeError("Ego-LAP runtime contract evidence is incomplete")
        runtime_safety = validate_eef_runtime_safety(env, require_gripper_runtime=True)
        committed_episode_indices = (
            [int(value) for value in episode_df["episode"].tolist()]
            if "episode" in episode_df
            else []
        )
        sidecars = load_episode_safety_sidecars(safety_dir, committed_episode_indices)
        aggregate_safety = aggregate_episode_safety(runtime_safety, sidecars)
        atomic_write_runtime_contract(
            Path(eval_args.runtime_contract_output),
            protocol=runtime_protocol,
            frame=runtime_frame_evidence,
            ik_safety=aggregate_safety,
        )
        return runtime_safety

    def validate_runtime_frame_and_persist(observation):
        nonlocal runtime_frame_evidence
        if runtime_protocol is None or eval_args.runtime_contract_output is None:
            raise RuntimeError("Ego-LAP runtime contract output was not configured")
        runtime_frame_evidence = validate_eef_runtime_frame(env, observation)
        runtime_safety = persist_runtime_contract()
        print(
            "POLARIS_LAP_RUNTIME_EEF="
            f"frame={runtime_frame_evidence['eef_frame']};"
            f"reference={runtime_frame_evidence['reference_frame']};"
            f"position_error_m={runtime_frame_evidence['position_error_m']};"
            f"rotation_error_rad={runtime_frame_evidence['rotation_error_rad']}",
            flush=True,
        )
        print(
            "POLARIS_LAP_IK_SAFETY="
            f"profile={runtime_safety['profile']};"
            f"cadence={runtime_safety['apply_actions_cadence']};"
            f"physics_dt={runtime_safety['physics_dt']};"
            "target_guard_band="
            f"{runtime_safety['target_soft_limit_guard_band_profile']};"
            "target_limit_sha256="
            f"{runtime_safety['target_joint_pos_limits_float32_sha256']};"
            "max_delta_joint_pos_rad="
            + ",".join(
                str(value) for value in runtime_safety["max_delta_joint_pos_rad"]
            ),
            flush=True,
        )

    if episode >= rollouts:
        if is_ego_lap:
            resumed_observation, _ = env.reset(object_positions=initial_conditions[0])
            install_or_validate_gripper_runtime()
            validate_runtime_frame_and_persist(resumed_observation)
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    if policy_client is None:
        policy_client = InferenceClient.get_client(eval_args.policy)
        if is_ego_lap:
            policy_client.bind_runtime_contract(runtime_protocol)

    horizon = evaluation_horizon
    runtime_frame_validated = False
    while episode < rollouts:
        # Index the initial condition with the episode being started. The old
        # loop reset before incrementing ``episode``, repeating condition zero.
        obs, info = env.reset(object_positions=initial_conditions[episode])
        if is_ego_lap:
            install_or_validate_gripper_runtime()
        if is_ego_lap and not runtime_frame_validated:
            validate_runtime_frame_and_persist(obs)
            runtime_frame_validated = True
        if is_ego_lap:
            policy_client.reset(episode_index=episode)
            begin_eef_safety_episode(env, episode)
            rollout_environment_before = capture_eef_environment_state(env)
            policy_client.begin_rollout(rollout_environment_before)
            rollout_environment_after = rollout_environment_before
            terminated_false_count = 0
            truncated_false_count = 0
        else:
            policy_client.reset()
        video = []
        numerical_failure_reason = ""
        bar = tqdm.tqdm(total=horizon)
        print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")

        while bar.n < horizon:
            action, viz = policy_client.infer(
                obs, language_instruction, return_viz=True
            )
            if viz is not None:
                # Request visualization every step so saved videos are complete,
                # even when policy inference itself is open-loop chunked.
                video.append(viz)
            try:
                if is_ego_lap:
                    environment_before = capture_eef_environment_state(env)
                obs, rew, term, trunc, info = env.step(
                    torch.as_tensor(action, device=env.device).reshape(1, -1),
                    expensive=policy_client.rerender,
                )
                if is_ego_lap:
                    record_eef_gripper_post_policy_step(env)
                    environment_after = capture_eef_environment_state(env)
                    transition = validate_eef_outer_step_transition(
                        step_index=bar.n,
                        environment_before=environment_before,
                        environment_after=environment_after,
                        terminated=term,
                        truncated=trunc,
                    )
                    policy_client.record_execution(transition)
                    rollout_environment_after = environment_after
                    terminated_false_count += 1
                    truncated_false_count += 1
            except (DifferentialIKNumericalError, torch.linalg.LinAlgError) as error:
                numerical_failure_reason = f"{type(error).__name__}: {error}"
                if is_ego_lap:
                    # Isaac increments the sim counter before apply_action. Preserve
                    # that one-to-eight-substep failed tail as the actual terminal
                    # state; episode/common/camera counters must not advance.
                    rollout_environment_after = capture_eef_environment_state(env)
                    policy_client.record_execution_failure(numerical_failure_reason)
                print(
                    f"Numerical failure in episode {episode} at action "
                    f"{bar.n}: {numerical_failure_reason}"
                )
                # The policy visualization was already recorded for this
                # attempted action, so include it in the episode length.
                bar.update(1)
                break
            bar.update(1)
            if not is_ego_lap and (term[0] or trunc[0]):
                break

        episode_length = bar.n
        bar.close()

        if video:
            filename = run_folder / f"episode_{episode}.mp4"
            atomic_write_episode_video(filename, video, fps=15)
        else:
            if is_ego_lap:
                raise RuntimeError(
                    f"Ego-LAP returned no visualization for episode {episode}"
                )
            print(f"Warning: policy returned no visualization for episode {episode}")

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
        if is_ego_lap:
            terminal_rollout = build_terminal_rollout_evidence(
                episode_result=episode_data,
                environment_before=rollout_environment_before,
                environment_after=rollout_environment_after,
                terminated_false_count=terminated_false_count,
                truncated_false_count=truncated_false_count,
            )
            finalized_trace = policy_client.finalize_episode(
                episode_length=episode_length,
                success=bool(episode_data["success"]),
                progress=float(episode_data["progress"]),
                terminal_rollout=terminal_rollout,
                numerical_failure_reason=numerical_failure_reason,
            )
            if finalized_trace is None:
                raise RuntimeError(
                    "Transactional Ego-LAP evaluation did not finalize a trace"
                )
            episode_safety = eef_episode_safety_report(env, episode)
            artifact_identity = build_episode_artifact_identity(
                run_folder=run_folder,
                trace_path=finalized_trace,
                episode_result=episode_data,
            )
            atomic_write_episode_safety(
                safety_dir / f"episode_{episode:06d}.json",
                episode_index=episode,
                episode_result=episode_data,
                safety=episode_safety,
                artifact_identity=artifact_identity,
                terminal_rollout=terminal_rollout,
            )
        episode_df = pd.concat(
            [episode_df, pd.DataFrame([episode_data])], ignore_index=True
        )
        atomic_write_results(episode_df, csv_path)
        if is_ego_lap:
            persist_runtime_contract()
        print(f"Episode {episode} finished. Episode length: {episode_length}")
        episode += 1

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
