import tyro
import mediapy

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
import os
import pandas as pd
import sys
import traceback


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
    position_adapter = eval_args.policy.client == "DroidDeltaJointPosition"
    native_drive_contract = (
        eval_args.control_mode == "joint-velocity" or position_adapter
    )
    if native_drive_contract:
        from polaris.pi05_droid_jointvelocity_contract import (
            NATIVE_GRIPPER_DRIVE_PROFILE,
        )

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
    if (
        eval_args.policy.client == "DroidJointVelocity"
        and eval_args.control_mode != "joint-velocity"
    ):
        raise ValueError("DroidJointVelocity requires --control-mode joint-velocity")
    if position_adapter and eval_args.control_mode != "joint-position":
        raise ValueError(
            "DroidDeltaJointPosition requires --control-mode joint-position"
        )
    if position_adapter and eval_args.rollouts != 1:
        raise ValueError("DroidDeltaJointPosition requires exactly one rollout")
    if (
        eval_args.control_mode == "joint-velocity"
        and eval_args.policy.client != "DroidJointVelocity"
    ):
        raise ValueError(
            "joint-velocity control mode is reserved for DroidJointVelocity"
        )
    if (eval_args.policy.client == "DroidJointVelocity" or position_adapter) and (
        not eval_args.runtime_contract_path
        or not eval_args.lifecycle_ready_path
        or not eval_args.policy.trace_path
    ):
        raise ValueError(
            "Audited pi0.5-DROID clients require --runtime-contract-path and "
            "--lifecycle-ready-path and --policy.trace-path"
        )
    if eval_args.control_mode == "joint-velocity":
        if eval_args.expected_gripper_drive_profile != NATIVE_GRIPPER_DRIVE_PROFILE:
            raise ValueError("joint-velocity expected gripper drive profile mismatch")
    elif position_adapter:
        if eval_args.expected_gripper_drive_profile != NATIVE_GRIPPER_DRIVE_PROFILE:
            raise ValueError(
                "DroidDeltaJointPosition expected gripper drive profile mismatch"
            )
    elif eval_args.expected_gripper_drive_profile is not None:
        raise ValueError(
            "expected gripper drive profile is valid only for joint-velocity control "
            "or DroidDeltaJointPosition"
        )

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
    from polaris.pi05_droid_native_lifecycle import NativeEvaluatorLifecycle

    lifecycle = NativeEvaluatorLifecycle(simulation_app)
    try:
        result = _run_evaluation(eval_args, lifecycle)
    except BaseException as error:
        # Pinned Kit teardown may hard-exit zero.  Never enter it with an active
        # evaluator failure, because that would erase the original traceback
        # and make the srun look successful.  Process exit releases the failed
        # simulator; only a complete transaction is allowed through close().
        traceback.print_exception(error, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        raise SystemExit(1) from error
    lifecycle.close()
    return result


def _run_evaluation(eval_args: EvalArgs, lifecycle):
    position_adapter = eval_args.policy.client == "DroidDeltaJointPosition"
    audited_jointpos = eval_args.policy.client == "DroidJointPos"
    audited_droid = eval_args.control_mode == "joint-velocity" or position_adapter
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
    from polaris.environments.pi05_droid_jointpos_cfg import (
        DroidJointPositionActionCfg,
        DroidJointPositionObservationCfg,
    )
    from polaris.environments.pi05_droid_position_cfg import (
        DroidPositionAdapterActionCfg,
        DroidPositionAdapterEventCfg,
        DroidPositionAdapterObservationCfg,
    )
    from polaris.joint_velocity_runtime import (
        print_joint_velocity_runtime,
        validate_joint_velocity_runtime,
    )
    from polaris.evaluation_seed import (
        bind_environment_seed,
        episode_environment_seed,
        format_environment_seed_contract,
        make_live_environment_seed_contract,
        validate_episode_seed_range,
    )
    from polaris.pi05_droid_jointpos_runtime import (
        PI05_DROID_JOINTPOS_OUTER_STEPS,
        capture_jointpos_actuator_intent,
        capture_jointpos_runtime,
        configure_jointpos_timeout,
        format_jointpos_runtime,
        publish_jointpos_runtime,
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
    from polaris.pi05_droid_position_runtime import (
        capture_position_adapter_runtime,
        make_position_close_ready,
        make_position_episode_sidecar,
        make_position_failure_close_ready,
        make_position_failure_sidecar,
        make_position_safety_report,
        print_position_adapter_runtime,
    )
    from polaris.utils import load_eval_initial_conditions
    from polaris.policy import InferenceClient
    from polaris.robust_differential_ik import DifferentialIKNumericalError
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    selected_client_class = InferenceClient.load_client_class(eval_args.policy.client)
    selected_client_module = sys.modules[selected_client_class.__module__]

    def selected_exception_type(name: str) -> type[BaseException]:
        exception_type = getattr(selected_client_module, name, None)
        if not isinstance(exception_type, type) or not issubclass(
            exception_type, BaseException
        ):
            raise RuntimeError(
                f"Selected client module {selected_client_class.__module__} lacks "
                f"exception type {name}"
            )
        return exception_type

    observation_numerical_errors = ()
    position_target_limit_errors = ()
    native_velocity_limit_errors = ()
    if audited_jointpos:
        observation_numerical_errors = (
            selected_exception_type("JointPositionObservationNumericalError"),
        )
    elif eval_args.control_mode == "joint-velocity":
        from polaris.native_gripper_runtime import NativeAllJointVelocityLimitError

        observation_numerical_errors = (
            selected_exception_type("JointVelocityObservationNumericalError"),
        )
        native_velocity_limit_errors = (NativeAllJointVelocityLimitError,)
    elif position_adapter:
        from polaris.native_gripper_runtime import NativeAllJointVelocityLimitError

        observation_numerical_errors = (
            selected_exception_type("JointPositionObservationNumericalError"),
        )
        position_target_limit_errors = (
            selected_exception_type("PositionTargetLimitError"),
        )
        native_velocity_limit_errors = (NativeAllJointVelocityLimitError,)

    _print_eval_phase("after_evaluation_imports")
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
    configured_episode_length_seconds = None
    jointpos_actuator_intent = None
    if eval_args.control_mode == "eef-pose":
        # Action managers are constructed by gym.make, so select the controller
        # on the config before creating the environment.
        env_cfg.actions = EefPoseActionCfg()
    elif eval_args.control_mode == "joint-velocity":
        from polaris.environments.robot_cfg import (
            make_nvidia_droid_joint_velocity_cfg,
        )

        env_cfg.scene.robot = make_nvidia_droid_joint_velocity_cfg()
        env_cfg.actions = DroidJointVelocityActionCfg()
        env_cfg.events = DroidJointVelocityEventCfg()
        env_cfg.observations = DroidJointVelocityObservationCfg()
        configured_episode_length_seconds = configure_native_environment_timeout(
            env_cfg
        )
    elif position_adapter:
        from polaris.environments.pi05_droid_position_robot_cfg import (
            make_nvidia_droid_position_adapter_cfg,
        )

        env_cfg.scene.robot = make_nvidia_droid_position_adapter_cfg()
        env_cfg.actions = DroidPositionAdapterActionCfg()
        env_cfg.events = DroidPositionAdapterEventCfg()
        env_cfg.observations = DroidPositionAdapterObservationCfg()
        configured_episode_length_seconds = configure_native_environment_timeout(
            env_cfg
        )
    elif audited_jointpos:
        env_cfg.actions = DroidJointPositionActionCfg()
        env_cfg.observations = DroidJointPositionObservationCfg()
        configured_episode_length_seconds = configure_jointpos_timeout(env_cfg)
        _print_eval_phase("before_jointpos_actuator_intent_capture")
        jointpos_actuator_intent = capture_jointpos_actuator_intent(env_cfg.scene.robot)
        _print_eval_phase("after_jointpos_actuator_intent_capture")
    elif eval_args.control_mode != "joint-position":
        raise ValueError(f"Unsupported control mode: {eval_args.control_mode}")
    _print_eval_phase("before_gym_make")
    env: ManagerBasedRLSplatEnv = gym.make(  # type: ignore[assignment]
        eval_args.environment, cfg=env_cfg
    )
    _print_eval_phase("after_gym_make")
    lifecycle.bind_environment(env)
    environment_seed_contract = None
    if eval_args.environment_seed is not None:
        environment_seed_contract = make_live_environment_seed_contract(
            env, eval_args.environment_seed
        )
        print(
            format_environment_seed_contract(environment_seed_contract),
            flush=True,
        )
    runtime_artifact = None
    environment_runtime_contract = None
    if eval_args.control_mode == "joint-velocity":
        # Gym construction initializes physics but does not execute reset-mode
        # events.  Perform one native reset before accepting any live drive
        # contract; the rollout reset below reapplies and recounts the cap.
        _print_eval_phase("before_native_preflight_reset")
        env.reset(expensive=False)
        _print_eval_phase("after_native_preflight_reset")
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
    elif position_adapter:
        # Reset-mode events establish the all-six gripper limits before the
        # live controller report is accepted. The rollout reset repeats them.
        _print_eval_phase("before_position_preflight_reset")
        env.reset(expensive=False)
        _print_eval_phase("after_position_preflight_reset")
        environment_runtime_contract = make_environment_runtime_contract(
            configured_episode_length_seconds=configured_episode_length_seconds,
            live_max_episode_length=env.max_episode_length,
        )
        runtime_contract = capture_position_adapter_runtime(env)
        print_position_adapter_runtime(runtime_contract)
        runtime_artifact = publish_immutable_json(
            Path(eval_args.runtime_contract_path), runtime_contract
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

    _print_eval_phase(f"before_policy_client_load:{eval_args.policy.client}")
    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)
    _print_eval_phase(f"after_policy_client_load:{eval_args.policy.client}")
    if eval_args.policy.client == "DroidJointPos":
        if environment_seed_contract is None:
            raise RuntimeError("DroidJointPos environment seed contract is missing")
        policy_client.bind_environment_seed_contract(environment_seed_contract)
    if eval_args.control_mode == "joint-velocity":
        policy_client.bind_evaluation_runtime(environment_runtime_contract)
    elif position_adapter:
        policy_client.bind_evaluation_runtime(environment_runtime_contract)

    horizon = (
        PI05_DROID_NATIVE_EPISODE_STEPS
        if audited_droid
        else (
            PI05_DROID_JOINTPOS_OUTER_STEPS
            if audited_jointpos
            else env.max_episode_length
        )
    )
    jointpos_runtime_sha256 = None
    terminal_outcome = None
    episode_sidecar_artifact = None
    position_safety_report = None
    trace_artifact = None
    metrics_artifact = None
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
            if jointpos_actuator_intent is None:
                raise RuntimeError("joint-position pre-gym actuator intent is missing")
            _print_eval_phase(f"before_jointpos_runtime_capture:{episode}")
            live_jointpos_runtime = capture_jointpos_runtime(
                env,
                obs,
                actuator_intent=jointpos_actuator_intent,
            )
            _print_eval_phase(f"after_jointpos_runtime_capture:{episode}")
            if jointpos_runtime_sha256 is None:
                print(format_jointpos_runtime(live_jointpos_runtime), flush=True)
                runtime_artifact = publish_jointpos_runtime(
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
        native_arm_term = None
        if audited_droid:
            native_arm_term = getattr(env, "unwrapped", env).action_manager._terms[
                "arm"
            ]
            native_arm_term.bind_native_all_joint_failure_path(incident_path)
        if position_adapter:
            policy_client.begin_rollout(env)
        if eval_args.control_mode == "joint-velocity":
            policy_client.begin_rollout(env)
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
            except position_target_limit_errors as error:
                if not position_adapter:
                    raise
                video.append(policy_client.visualize(obs))
                dynamic_report = native_arm_term.native_all_joint_dynamic_report(
                    include_samples=False
                )
                terminal_outcome = policy_client.record_target_limit_failure(
                    error, env, dynamic_report
                )
                numerical_failure_reason = terminal_outcome["reason"]
                incident_artifact = error.incident_artifact
                bar.update(1)
                break
            except observation_numerical_errors as error:
                if position_adapter:
                    # State/image/schema corruption is a contract-fatal error,
                    # not a controller numerical outcome.
                    raise
                if eval_args.control_mode == "joint-velocity" and isinstance(
                    error, observation_numerical_errors
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
            except native_velocity_limit_errors as error:
                if position_adapter:
                    dynamic_report = native_arm_term.native_all_joint_dynamic_report(
                        include_samples=False
                    )
                    terminal_outcome = policy_client.record_execution_failure(
                        error, env, dynamic_report
                    )
                    numerical_failure_reason = terminal_outcome["reason"]
                    incident_artifact = error.incident_artifact
                    bar.update(1)
                    break
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
            if audited_droid:
                try:
                    native_arm_term.record_native_all_joint_post_policy_step()
                except native_velocity_limit_errors as error:
                    if position_adapter:
                        policy_client.record_execution(
                            obs,
                            env,
                            terminated=term,
                            truncated=trunc,
                        )
                        dynamic_report = (
                            native_arm_term.native_all_joint_dynamic_report(
                                include_samples=False
                            )
                        )
                        terminal_outcome = policy_client.record_execution_failure(
                            error, env, dynamic_report
                        )
                        numerical_failure_reason = terminal_outcome["reason"]
                        incident_artifact = error.incident_artifact
                        bar.update(1)
                        break
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
            if not audited_droid and not audited_jointpos and (term[0] or trunc[0]):
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
        elif (
            position_adapter
            and not numerical_failure_reason
            and episode_length == PI05_DROID_NATIVE_EPISODE_STEPS
        ):
            terminal_outcome = policy_client.finish_rollout(env, info["rubric"])
            dynamic_report = (
                getattr(env, "unwrapped", env)
                .action_manager._terms["arm"]
                .native_all_joint_dynamic_report(include_samples=False)
            )
            position_safety_report = make_position_safety_report(
                dynamic_report, outer_steps=episode_length
            )
        bar.close()

        video_artifact = None
        if video:
            filename = run_folder / f"episode_{episode}.mp4"
            if audited_droid:
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
                    if audited_droid
                    else info["rubric"]["success"]
                )
            ),
            "progress": (
                0.0
                if numerical_failure_reason
                else (
                    terminal_outcome["rubric"]["progress"]
                    if audited_droid
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
        elif position_adapter:
            trace_artifact = policy_client.finalized_trace_artifact
            if trace_artifact is None or video_artifact is None:
                raise RuntimeError(
                    "DROID position-adapter episode transaction is incomplete"
                )
            if numerical_failure_reason:
                if (
                    incident_artifact is None
                    or dynamic_report is None
                    or terminal_outcome is None
                ):
                    raise RuntimeError(
                        "DROID position numerical-failure evidence is incomplete"
                    )
                sidecar_payload = make_position_failure_sidecar(
                    episode_result=episode_data,
                    environment_runtime_contract=environment_runtime_contract,
                    terminal_failure=terminal_outcome,
                    dynamic_report=dynamic_report,
                    trace_artifact=trace_artifact,
                    video_artifact=video_artifact,
                    incident_artifact=incident_artifact,
                )
            else:
                if (
                    position_safety_report is None
                    or episode_length != PI05_DROID_NATIVE_EPISODE_STEPS
                    or terminal_outcome is None
                ):
                    raise RuntimeError(
                        "DROID position successful episode evidence is incomplete"
                    )
                sidecar_payload = make_position_episode_sidecar(
                    episode_result=episode_data,
                    environment_runtime_contract=environment_runtime_contract,
                    terminal_rollout=terminal_outcome,
                    safety_report=position_safety_report,
                    trace_artifact=trace_artifact,
                    video_artifact=video_artifact,
                )
            sidecar_artifact = publish_immutable_json(sidecar_path, sidecar_payload)
            episode_sidecar_artifact = {
                key: sidecar_artifact[key]
                for key in ("path", "size", "sha256", "mode", "nlink")
            }
        episode_df = pd.concat(
            [episode_df, pd.DataFrame([episode_data])], ignore_index=True
        )
        if audited_droid:
            temporary_csv = csv_path.with_name(
                f".{csv_path.name}.partial-{os.getpid()}"
            )
            try:
                episode_df.to_csv(temporary_csv, index=False)
                metrics_artifact = publish_immutable_file_from_temporary(
                    temporary_csv, csv_path
                )
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
    elif position_adapter:
        if (
            runtime_artifact is None
            or environment_runtime_contract is None
            or terminal_outcome is None
            or trace_artifact is None
            or video_artifact is None
            or metrics_artifact is None
            or episode_sidecar_artifact is None
            or not eval_args.policy.trace_path
        ):
            raise RuntimeError("DROID position-adapter close evidence is incomplete")
        if numerical_failure_reason:
            close_payload = make_position_failure_close_ready(
                runtime_artifact=runtime_artifact,
                trace_artifact=trace_artifact,
                video_artifact=video_artifact,
                metrics_artifact=metrics_artifact,
                sidecar_artifact=episode_sidecar_artifact,
                environment_runtime_contract=environment_runtime_contract,
                terminal_failure=terminal_outcome,
            )
        else:
            if position_safety_report is None:
                raise RuntimeError("DROID position success safety report is missing")
            close_payload = make_position_close_ready(
                runtime_artifact=runtime_artifact,
                trace_artifact=trace_artifact,
                video_artifact=video_artifact,
                metrics_artifact=metrics_artifact,
                sidecar_artifact=episode_sidecar_artifact,
                safety_report=position_safety_report,
                environment_runtime_contract=environment_runtime_contract,
                terminal_rollout=terminal_outcome,
                outer_steps=PI05_DROID_NATIVE_EPISODE_STEPS,
            )
        lifecycle.prepare_close_ready(
            publish_immutable_json,
            Path(eval_args.lifecycle_ready_path),
            close_payload,
        )


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
