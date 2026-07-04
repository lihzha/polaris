"""Closed evidence gate authorizing standard v5 Ego-LAP evaluation.

This is an evidence-only descendant of the sealed ``9ab844b`` runtime.  It
does not change evaluator, environment, policy-client, or controller behavior.
It promotes the runtime from the bounded six-task smoke stage to the canonical
50-rollout-per-task stage only after validating the exact immutable evidence
manifest below.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
from typing import Any, Mapping

from polaris import eef_velocity_recovery_promotion as canary_promotion


PROMOTION_PROFILE = "measured_velocity_recovery_v5_repaired_smoke_suites_standard_v1"
PROMOTION_STATUS = "validated_on_two_repaired_six_task_smoke_suites_standard_authorized"
PROMOTION_PARENT_COMMIT = "9ab844b3fcaac6d29b51bc9fb2c2758c125201f3"
PROMOTION_PARENT_TREE = "2063a7d091ee9d1c6e0646a60ee501a8abd395e8"
PROMOTION_BASE_COMMIT = "0142e8518769d386c0a8227778767800b30c7e83"
PROMOTION_BASE_TREE = "4a659255f1dc26ebcb15ffce599cd2831b5b548f"
SMOKE_EGO_LAP_COMMIT = "5ad7da3057829d5a90cb5da78197bb3bd21f969f"
PREDECESSOR_SOURCE_PATH = "src/polaris/eef_velocity_recovery_promotion.py"
PREDECESSOR_SOURCE_SHA256 = (
    "f98f6d3ae6eb06f0127e3ec686fa70e3bb524ea892582b0ee3461b0dd6d84df4"
)
PREDECESSOR_EVIDENCE_SHA256 = (
    "9576a178253741571a50cd23fe8a16b75b9a386ced5bc43ee416348fa52454f7"
)

AUTHORIZED_EVAL_SCALES = ("canary", "smoke_suite", "standard")
NEXT_EVAL_SCALE = "standard"
CANONICAL_TASKS = canary_promotion.CANONICAL_SMOKE_SUITE_TASKS
STANDARD_ROLLOUTS_PER_TASK = 50
STANDARD_POLICY_STEPS = 450
STANDARD_POLICY_HZ = 15
STANDARD_PHYSICS_SUBSTEPS = 8

RUNTIME_DESCENDANT_ATTESTATION = {
    "profile": "headless_viewport_recovery_9ab844b_sealed_l40s_nonstandard_smoke_only_v1",
    "promotion_base": PROMOTION_BASE_COMMIT,
    "recovery_commit": "8db95a83cdfffde358893af1aced433f1029b65c",
    "lifecycle_commit": "38153722deecdd4af34d0926f6d095988c9a2306",
    "polaris_parent": "a90aea6320c25bc43c830530116b295581e4bc8b",
    "polaris_commit": PROMOTION_PARENT_COMMIT,
    "polaris_tree": PROMOTION_PARENT_TREE,
    "plain_binary_diff_sha256": (
        "8b53ff7819028e2c689f5fe84d82151a0902e90bc38140f72597e85cf6eb23b0"
    ),
    "cumulative_binary_diff_sha256": (
        "a669c72b9541a4dc534708827d74392b7eb6204f452436f94b01d15b2b2049b9"
    ),
    "attestation": {
        "job_id": 1098834,
        "path": (
            "/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/results/"
            "polaris_eval/headless_viewport_recovery/9ab844b/job_1098834/"
            "headless_viewport_recovery.attestation.json"
        ),
        "sha256": ("efded6682bce983a4d773b038990f9e9fd5968cd05efe42b204063c6b4c7b0c5"),
        "status": "sealed_validated_forced_l40s_recovery_v1",
        "wrapper_sha256": (
            "5e4f6ab9eec2833d4bea17715d925cb4c082f39bbae52afedc30fadead48fdeb"
        ),
        "finalizer_sha256": (
            "527e4d430b165820ba7027814dd0f37a2c7c9fbcb9518f9927f50034ed7b3032"
        ),
        "ego_lap_commit": "093701beb57c6d1baf46f1f7848cc6e1ad418b68",
    },
}

REASONING_RUN = (
    "reasoning-full-43075-main-5ad7da3-polaris-9ab844b-smoke6-rerun-20260704"
)
REASONING_ROOT = (
    "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/" + REASONING_RUN
)
REASONING_RESOLVED_ROOT = (
    "/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/results/"
    "polaris_eval/" + REASONING_RUN
)
REASONING_STEP_RELATIVE = (
    "polaris-droid-link8-eefpose-suite-smoke1-q99repair-controllercandidate-v5-"
    "iksafety-v5/ego-5ad7da3057829d5a90cb5da78197bb3bd21f969f/"
    "polaris-9ab844b3fcaac6d29b51bc9fb2c2758c125201f3/step_43075"
)
OFFICIAL_RUN = (
    "official-lap3b-601db9c1-main-5ad7da3-polaris-9ab844b-smoke6-rerun-20260704"
)
OFFICIAL_ROOT = (
    "/lustre/fsw/portfolios/nvr/users/lzha/results/polaris_eval/" + OFFICIAL_RUN
)
OFFICIAL_RESOLVED_ROOT = (
    "/lustre/fs11/portfolios/nvr/projects/nvr_lpr_rvp/users/lzha/results/"
    "polaris_eval/" + OFFICIAL_RUN
)
OFFICIAL_STEP_RELATIVE = (
    "source-hf-lihzha-LAP-3B-revision-"
    "601db9c1ab4bcaf6dddb160c7b2dec589a67b730-manifest-"
    "567cc3ff7d20f3f03913a6f11c3fa151f789e1c0118ed5af0eea24d9cc48f20e/"
    "polaris_droid_suite_v1/polaris-droid-link8-eefpose-suite-smoke1-r6rows-"
    "q99repair-controllercandidate-v5-iksafety-v5/requested-original_lap/"
    "checkpoint-auto/q99_train_matched_v1/flow/"
    "ego-5ad7da3057829d5a90cb5da78197bb3bd21f969f/"
    "polaris-9ab844b3fcaac6d29b51bc9fb2c2758c125201f3/step_0"
)


def _artifact(relative_path: str, sha256: str) -> dict[str, str]:
    return {"relative_path": relative_path, "sha256": sha256}


def _task_evidence(
    *,
    step_relative: str,
    run_name: str,
    output_step: int,
    task: str,
    worker_job_id: int,
    progress: float,
    hashes: Mapping[str, str],
    visual_finding: str,
) -> dict[str, Any]:
    task_relative = f"{step_relative}/{task}"
    attempt_relative = f"{task_relative}/attempts/job_{worker_job_id}_restart_0"
    summary_stem = f"{run_name}_{output_step}_polaris_{task}"
    paths = {
        "authoritative_attempt": f"{task_relative}/authoritative_attempt.env",
        "task_complete_verified": f"{task_relative}/task_complete_verified.env",
        "eval_success": f"{task_relative}/eval_success.txt",
        "completion_audit": f"{attempt_relative}/completion_audit.json",
        "registry_candidate": f"{attempt_relative}/registry_candidate.json",
        "runtime_contract": (
            f"{attempt_relative}/results/polaris_runtime_contract.json"
        ),
        "episode_sidecar": (
            f"{attempt_relative}/results/ik_safety/episode_000000.json"
        ),
        "finalized_trace": (
            f"{attempt_relative}/results/policy_traces/episode_000000.jsonl"
        ),
        "rollout_video": f"{attempt_relative}/results/episode_0.mp4",
        "summary_sidecar": f"{attempt_relative}/{summary_stem}.json",
        "summary_video": f"{attempt_relative}/{summary_stem}.mp4",
    }
    if set(hashes) != set(paths):
        raise RuntimeError(f"Task artifact schema drift for {task}")
    return {
        "worker_job_id": worker_job_id,
        "scheduler_state": "COMPLETED",
        "scheduler_exit_code": "0:0",
        "attempt_relative_path": attempt_relative,
        "policy_steps": STANDARD_POLICY_STEPS,
        "physics_apply_calls": STANDARD_POLICY_STEPS * STANDARD_PHYSICS_SUBSTEPS,
        "numerical_failures": 0,
        "controller_aborts": 0,
        "dropped_diagnostics": 0,
        "raw_rubric_successes": 0,
        "task_valid_successes": 0,
        "normalized_progress": progress,
        "artifacts": {key: _artifact(paths[key], hashes[key]) for key in sorted(paths)},
        "visual_finding": visual_finding,
    }


def _reasoning_tasks() -> dict[str, Any]:
    entries = (
        (
            "DROID-BlockStackKitchen",
            1098871,
            1 / 7,
            {
                "authoritative_attempt": "728f5674ab06ce4e24a1181cc40bd69819b4373a9730a02c197f0c3e5cec17ec",
                "completion_audit": "900433db2486087aaf04a0569bde868bbd64c24457b1cb5b49d64c75cf5a941e",
                "episode_sidecar": "380dabe9364a0cd70c5ad91b0a333509f0fe14a07e2ff3dfa68c9f3fd7fb9bf4",
                "eval_success": "f2b602c5a48375c653e843813b61b36b6bb0486869582df46863ce7b28adaef4",
                "finalized_trace": "92f40afc62091207d358935989a764015423cc7ea6b9a9b45739bb713d23ec90",
                "registry_candidate": "99b89cfbffd40561673c4232c8a2f23aacbdc7e1679e108b7d9dff18a8600e7d",
                "rollout_video": "43dc132422db9a9b0b43f08ea34a9bee01b8a6ee4986a4a7b133bc304da6cb32",
                "runtime_contract": "7614546fdda47115ac9b113b4f89913e3e9527630d2c415b182ad87077809ad2",
                "summary_sidecar": "916b34599cf39dec12e5ce5e58cb94e9033d9418f1c8488d9840036cdcd00ac1",
                "summary_video": "75f9c60f3cc229d823569236afff86b1e99a7d30e6ddd462b8882c59a9d9e6b8",
                "task_complete_verified": "5240d255cccb9c5d185e6209e221cebde02f78cbce81c1f28a291227534b8eae",
            },
            "moved_near_relevant_block_and_green_tray_without_stacking",
        ),
        (
            "DROID-FoodBussing",
            1098872,
            1 / 6,
            {
                "authoritative_attempt": "eece393b302043b41e10290b38f2943b056252caa46f869273aaaa72c5b7963e",
                "completion_audit": "13b116ae3e71e547b63b84ca2bb4d200bf21bde86dede890037f3eff8e7f383b",
                "episode_sidecar": "28bf4f73ce61ef39ba399b52aaa0a36b1cb43e0c517a8fbd0fdc1bc11fd458d6",
                "eval_success": "b81c8232e1fd79290b61d16dc09fae3f8435cd0b89c60c0834c7461f7eb844f6",
                "finalized_trace": "ceb2e93cc697498dc251e964e50492ce6ac29676aa3028234a8c339cd5d00d9a",
                "registry_candidate": "bae4698ca877928c37b0368a13eb89a5819c94c4a7cf95f9175bfb2786af5749",
                "rollout_video": "605df5729de571fc6bd9d0be9ea5afb955d037aa86b5b56dc58b782442b564e5",
                "runtime_contract": "8808466fe4bc9ecdc0cc09343a9686c2c28c7144fad85fb91b4a1871f8b3ac1f",
                "summary_sidecar": "1b07eca3903a36309765ae026c9ad2a4569974f4fb25399118359ac6a4abd768",
                "summary_video": "6294925a27b70442312ce7aa2fc665b1b70064901293a79b6939d872ed352830",
                "task_complete_verified": "101cff27ab55ae8324f6c29f6d9e20c08a417a7e2ee9e088a4e2c82a148a90e6",
            },
            "moved_toward_relevant_objects_without_food_placement_in_bowl",
        ),
        (
            "DROID-PanClean",
            1098878,
            1 / 3,
            {
                "authoritative_attempt": "22c9cb8275023f023d7595c3d982a864c7ad0047293a7b9b3549437500291fbd",
                "completion_audit": "4b1dcfcd615e75f9357989c0d2e8b74e98bd2ec45c9ca6d6b5d35238eb104c19",
                "episode_sidecar": "1d547d39cffa3c40e3fc11744e2bf1c9d1bd3388c227afe28908a745423f52db",
                "eval_success": "e1c00f7198777a420bae61f9971802b3c82ae7dc9d97485cd630bcb25775f5cb",
                "finalized_trace": "244e3291a6d2e2b8353dd088386a792afb4880e0e03e70127ee38e981530ec22",
                "registry_candidate": "011bd37f49582044d35fcece81fd4ba1c5ea730a02f89ca5ee4a3eea5698ee28",
                "rollout_video": "887dfe186c150db83cbe5c9ab69a3a2edb188cf804b15928b29f8065aba695c4",
                "runtime_contract": "49213b63ad7f1ecef06e86b3a00e5703000be537257c916d4d0245cee0bd082c",
                "summary_sidecar": "68593305af74e4fd341e2a1ceeb2efc7633cd4d3f85792e0ffd376e380b2342e",
                "summary_video": "f04b64ca685497dd504ceab3a2f27afeaaa27b54313fe7aa9b9c9113bb3eee9b",
                "task_complete_verified": "eb729a99f2ecb40539ef448833aad32efc08c42c87a417ebe5b608c53939c22e",
            },
            "approached_pan_sponge_and_can_region_without_completing_scrub",
        ),
        (
            "DROID-MoveLatteCup",
            1098879,
            0.0,
            {
                "authoritative_attempt": "89823608eb983e7e0be20443ecc19a65c3fc38adafbc0e2ef75d5eefb9afadc1",
                "completion_audit": "20cba4af74ba5269c7248e39d4992f9996f7973ac7d7bdbb6c35b93d94d0aee3",
                "episode_sidecar": "607fdd8ad3dd0cbf0fac52eb5852b308735049c3330e5b11741881d41ff7f43a",
                "eval_success": "53cd10cb7a1f52214231bc76408652a524fa4a22d6744de67574ba526aa1688d",
                "finalized_trace": "af8d822028fc96e8087fa96e172ad75b796e2903441e4145ef1239aa240fbfd5",
                "registry_candidate": "eace5376b76b5490c5eee443dd519c3862f1edaa9adf63eb97c777669d942cf8",
                "rollout_video": "618599313211651d295eee5da0fe9746d5e8cd00deed14bf33a7bf557c8ae148",
                "runtime_contract": "6a9da2da25d2b2412ed7fa4c360e144caecab20dd88077ab6929c248e3431041",
                "summary_sidecar": "fafb84beb9f22fb5faa00c023a7eb7629ec3138edbff4c4fcf25c5c8a31e1ee0",
                "summary_video": "1f6f39ef40566cf9119c6864254edee2b0b1d195c7e9e9400f35a2e2b216aa52",
                "task_complete_verified": "e938ca87e8e7b04596de8a5eac48b5913806144c590b1ed3d178eb520fec8ca0",
            },
            "did_not_move_latte_cup_to_cutting_board",
        ),
        (
            "DROID-OrganizeTools",
            1098882,
            0.0,
            {
                "authoritative_attempt": "d482c553ac283f259c774f2bc26b6e5970d58c5a0f384a1bb07541a91a0ce2c1",
                "completion_audit": "58606811eb12e81db48e201c7008f6eb9fa6913ad88ad123148d340c6ecd7b49",
                "episode_sidecar": "b0eedc67dc25feb71d0e64c056c39cc250c80819d32c9ee42733933a41db1c0a",
                "eval_success": "edfdb5ccbd90770ea2f31ce75ad27dd2edfc3fe71c51b5df65ea4024d999700c",
                "finalized_trace": "25ff29cce7f6c0c3f03bd66293e42a73966c82e305001c6d9f5ae39d01cc63b8",
                "registry_candidate": "7e1b907fd55c67ddad4e20f67b1202f9b31819a42fa940874b464fba4eeb3172",
                "rollout_video": "0f686a9e362146bcc5dd8e3cb5d9e579a9ff34cf428ab81c4dda544bbc19ec0b",
                "runtime_contract": "88e9f43f0acd6bf0f133fa95d3b5f3c2311ad86f9aab682cecc038211bc983b3",
                "summary_sidecar": "fb4e42cefe9d7280b6ff1915ede9eacae22fc4b1e61e510f266a81ece1878c20",
                "summary_video": "8e3c565bd9e9b68976bfb8b3fb8848d02655adf4d3865ff85312a9828b0e831d",
                "task_complete_verified": "1d8acb641d778e0d45854ab875cc0fc44c60b100fc26c0f5ed63e34fe7b1aac3",
            },
            "moved_near_container_without_inserting_scissors",
        ),
        (
            "DROID-TapeIntoContainer",
            1098884,
            0.0,
            {
                "authoritative_attempt": "3a07707da98c635cf22030d623bf5a5924050fa655c69476bd4aca979fbac2fb",
                "completion_audit": "9bc60d45687d57669540a5b6e013077fe9d278c71e68fb51a7985047ff1dd059",
                "episode_sidecar": "4b3c8ab2066de5fdf2183871ff994942c4cc863d441dd0c5f5992ba1b99ac3f9",
                "eval_success": "a734196570004aeca3fc9fa2a2c8f7d608e96ca973df8691a65d29e475176639",
                "finalized_trace": "c2acb4e9e5c721f37e5a06d723c880f474252acb6928b4c61d1e43d4b527b741",
                "registry_candidate": "89a2dc13460a97b4f8578c172678f86dafc3ea02c2e356370005e9676156564e",
                "rollout_video": "81e0daa3092565bff1424f6de4bacaf7ce294d14db1dda0d240dbb28e6560fe3",
                "runtime_contract": "e15d84eb78234fee8abc5fe64d3002c3c5fd4618ed5da2babf835935ff9d7e6b",
                "summary_sidecar": "397676824a416e738cff7b17ec0002c0109b6bf37998d4dbfc9b3a7d74959516",
                "summary_video": "fa581d8fe4cbe015ebaeebc16f3b684a013a2c9092edd0f8cfca4e41fd146a53",
                "task_complete_verified": "0f62ec145846cb7f54e0d58227a62846d2788d74083bc59c39bc5b1b4391997c",
            },
            "approached_tape_and_container_without_insertion",
        ),
    )
    return {
        task: _task_evidence(
            step_relative=REASONING_STEP_RELATIVE,
            run_name=REASONING_RUN,
            output_step=43075,
            task=task,
            worker_job_id=job,
            progress=progress,
            hashes=hashes,
            visual_finding=finding,
        )
        for task, job, progress, hashes, finding in entries
    }


def _official_tasks() -> dict[str, Any]:
    entries = (
        (
            "DROID-BlockStackKitchen",
            1098874,
            0.0,
            {
                "authoritative_attempt": "7121017798dbb698196d1ef18cf249e7af3eaf4a2ae44e7dfdd8e0c61974f84e",
                "completion_audit": "c13048cddc52f1f4632d0a9b6903cb22f7ffb3cfe0ccac6880e34bb2d5b91889",
                "episode_sidecar": "943a1cee60b7599c09463a4380bb7594f9ad7df95d131e05aa1fed09b1314e9c",
                "eval_success": "72969ea62ad066a44b34549f253a44c884545b5bb5b1be3c10008b26f138dd44",
                "finalized_trace": "ea51bebf46a0f0d8d84f43ee9d6eaf8cf0f4e5e1ef51723eaa2d483b2bd9d298",
                "registry_candidate": "b1668f1f1bf9437a48f8ea2ae88bd86f46d90358405ed98cd52156f19261849d",
                "rollout_video": "34a9d017a35b8bd45cfca7c56e17b5aede2bc450352e0c5d817e93a8d306f59f",
                "runtime_contract": "af25466fb8253c423c6926e57f8540fb2dd460950c7713f209015039add3a1dd",
                "summary_sidecar": "268adf8b21db9c0e54ef769616bdac0240bba07681549450511647a437bec064",
                "summary_video": "ec9a3b9299ba05fd6bccc2d4fdc884994848f4a0878e3e48358f7792c7f7e080",
                "task_complete_verified": "3c04d00c51d22bf47c034f06c95f725e0867536999a5477d709dbd7f60bf2c2f",
            },
            "moved_near_green_tray_and_blocks_without_stacking",
        ),
        (
            "DROID-FoodBussing",
            1098875,
            1 / 6,
            {
                "authoritative_attempt": "59848765a17a92ff8de02b9d7d8438e21a42913a771f40b864c0f361eadb42e1",
                "completion_audit": "2524122696eb8652f023da769dc2e7c24237f5f072096686e0451183b67c3ac7",
                "episode_sidecar": "9e9e0393faa608bffef61222e9afde4c9db70be0054cca299d07aa4e486225c7",
                "eval_success": "f2b602c5a48375c653e843813b61b36b6bb0486869582df46863ce7b28adaef4",
                "finalized_trace": "c9a0115202c01192b58d0cfc0994ff78e652554239885e98c0fc20c10a8211a8",
                "registry_candidate": "bc4e0918b5727fb78a40bdbaae497bb4ed6be210eccbcf80f330d021d857d5a3",
                "rollout_video": "8edf242fccb7ee168a7b98a4a98b8006924e149e51c6b963d37eddd5b7b172aa",
                "runtime_contract": "39ff8b7f3574cc2d57235dfe8ab0d262e6abb008555f16de4bc4a91f1f0f7744",
                "summary_sidecar": "9aa232136e1fea45f0f26777fc99ac630b031391a4770ac9d50be8af2b0d7de1",
                "summary_video": "2c38d565ac154066a27581a922031841bb93f1c1ee3c3eb5451132ba1baa6822",
                "task_complete_verified": "eb42d73c2e6d12e972e9a7688bb2a64bc09266908e789446f5631c14f6f9db1b",
            },
            "reached_yellow_bowl_area_without_food_placement",
        ),
        (
            "DROID-PanClean",
            1098876,
            0.0,
            {
                "authoritative_attempt": "c40e6b7425fccaed9e6a3fa86a88984ccb16d7f89f02f6ea0de7929d62be0e25",
                "completion_audit": "e7e03e22cb44d1c490cd2a09afd1a0b5d6ad71187e53263ca831fb96f1ea930a",
                "episode_sidecar": "df9ad42e695e01aaa24098eb44e487e8f31f24a5a54d02b168071b1901d89e4e",
                "eval_success": "7dcae59ae26eb56520213bc944c01a5aa40906048fc65d549d260104eb5712ea",
                "finalized_trace": "fe461a59ceb90fa0e1c6f82ecf6a771ea7e26a5776c55360f10f7ef53b226a98",
                "registry_candidate": "1f26929467486c9acf34da42b95be6e5874f5c64112f77a540e3ab1e2f2b06c3",
                "rollout_video": "12657443f55d92d5caa4cdec38fa15f0a079a3166714b273c5b3336c19652c8d",
                "runtime_contract": "4149eaa61af45723b6fdf194396854d3e510e3d470b351f31fe3b2b643836f43",
                "summary_sidecar": "d1e83a958723d4298bd51747832dcf66676073d3c204d6446742990f1c9f035d",
                "summary_video": "d948d1a18265df7aa3177caa41cdad7e56d7826c676207fd4bf6dfaf3ae93224",
                "task_complete_verified": "e64ef17005a7e49a07f51fbd97c87bf0b81be837bfa840e33f04b8414bc599a6",
            },
            "manipulated_and_moved_pan_instead_of_sponge_scrubbing",
        ),
        (
            "DROID-MoveLatteCup",
            1098877,
            0.0,
            {
                "authoritative_attempt": "cb8a20c581a0baae15f3b98bf63a0aeee3379929e39344c806a818df3b8fe990",
                "completion_audit": "a734ca2ce474d6271b885f14bf4829e3aadf6f5c4c4272206f4e3b002cabcad0",
                "episode_sidecar": "f62b87fc02c01116c402989387174389c1a1ca1d817c9e4dfc90f4540a7f4441",
                "eval_success": "0b318546a16c0e997cf238587331f52e17f6b8557a3b36baf9d1cfcfaeceef45",
                "finalized_trace": "704e2f1a250f633f82d2b4b810ca71376480dd72b8b005528a66731177bd37c5",
                "registry_candidate": "ec9a9dd76b50656434d787b66972e3fce01ff527c7c5c2e7f56562fc39cc191d",
                "rollout_video": "e902bb4f56d43786414735eefae2a7692e297b942c2c6ad79eae2eda584d4d0e",
                "runtime_contract": "b67c51c78f72b1d8139efcbbf56b6f60f78227749e3fb169b9eac83ccf4cc02f",
                "summary_sidecar": "051861a48aad016b33ba2b1b6054ce0385b97fa196ce28f88865bbdd28fbcd2c",
                "summary_video": "0f5e8e8eb2d0966d03d96d1530e27a18887ab6b80154df7b3373c36c35c014a7",
                "task_complete_verified": "a081ba638193c47a236e5793257a10364e82add9ed4e32a473f5cbaceb24b532",
            },
            "did_not_move_latte_cup_to_cutting_board",
        ),
        (
            "DROID-OrganizeTools",
            1098881,
            0.0,
            {
                "authoritative_attempt": "e3a2f0566615c8e6a7d035495a7fec00c7e10f1a054527033452855b48094d29",
                "completion_audit": "6c6d690690f91b2f410b4ac6478638d7bc0cfeb6ce840a10c0bd22538ba3ca95",
                "episode_sidecar": "f7291b090853b8188b6ef10dd35d8aa134576d94f85b2290f48f8b883a751212",
                "eval_success": "f1fb662e4824b9d56dbece14c741a9a31d77dc9540dc2d1e8230360591c7e6af",
                "finalized_trace": "d2568c620f62c52ea6d3e23f0bb62175902d2f72c2490210e5116ec34ca4afe5",
                "registry_candidate": "0418253ec13572088ccc8236026d0aa76c85a568cf97dc4810c793d7f02e2ec1",
                "rollout_video": "122cb71e42926752d05cc5bd7a61b51b70ae798ab33550ddf1f6d63cf0291a76",
                "runtime_contract": "3c75641d4936f56a33a34a0e5c1d415898b2e1b79eae8f2a3db10672446cc9b5",
                "summary_sidecar": "2972ca3f39ddb9ec56f2e25f06a9c643265681db055448e902506c0af6a61b65",
                "summary_video": "d9a2d1d2acb58dd9dc28e07a52cca90dad2bf7863bc0ec983d1a32c6e227bef5",
                "task_complete_verified": "ce7cc92195ab380b96a7488ce3baf5568a504f6dd37225a52bb1b8c6701c1fcb",
            },
            "moved_near_container_without_inserting_scissors",
        ),
        (
            "DROID-TapeIntoContainer",
            1098883,
            2 / 3,
            {
                "authoritative_attempt": "198d48ff91367df3e1d7ef6d2e041d71c3be625b971e295b39e9103b8ffd3cde",
                "completion_audit": "c765fe2c11da7052b7c2372e420dad39ff36485e55088fa2667cca8a8d71ebb5",
                "episode_sidecar": "5cae3d9502f819f70eef9f0dd7d28ef2990a24e29452ba06fa36e5125ea64fbf",
                "eval_success": "e876b88c396c901f5b5a4b3a1434247e40b807f0147cecde71c2c7358e53ad9e",
                "finalized_trace": "aca5748f06a6702444d152d3ad94d8622d8e62db29f6f302200080f40b22a5ce",
                "registry_candidate": "786ea563b7486a47661126b57160d7f770449db5109f4709adf1816dc8a991aa",
                "rollout_video": "4f67bf13760632f3b2412735e568acf16e5db45b772d34414ebc7a3e02b3ece1",
                "runtime_contract": "7ea407f7ac30de43d4303482d0b23931bc948ae8e98d9e60a13f4fb7bc42a137",
                "summary_sidecar": "25a2ef7b43d6cb82f723b18e0a5a336827e09bb65a3c944b4b0e4fac517018c5",
                "summary_video": "f0873a03f351e2408019af4a3e3dcee20c9091c7de7ccf9c7a1330dbdbc8a0e5",
                "task_complete_verified": "37873ba6005e0b20329948a9c06ddbe39972d269947b70e3b9822f13448c6d7e",
            },
            "approached_tape_and_container_with_two_of_three_rubric_progress_but_no_insertion",
        ),
    )
    return {
        task: _task_evidence(
            step_relative=OFFICIAL_STEP_RELATIVE,
            run_name=OFFICIAL_RUN,
            output_step=0,
            task=task,
            worker_job_id=job,
            progress=progress,
            hashes=hashes,
            visual_finding=finding,
        )
        for task, job, progress, hashes, finding in entries
    }


def _suite_artifacts(step_relative: str, hashes: Mapping[str, str]) -> dict[str, Any]:
    paths = {
        "eval_success": f"{step_relative}/eval_success.txt",
        "registry_candidates": f"{step_relative}/registry_candidates.json",
        "suite_summary": f"{step_relative}/suite_summary.json",
    }
    if set(hashes) != set(paths):
        raise RuntimeError("Suite artifact schema drift")
    return {key: _artifact(paths[key], hashes[key]) for key in sorted(paths)}


def canonical_eef_velocity_recovery_v5_standard_promotion_evidence() -> dict[str, Any]:
    """Return the closed repaired-smoke evidence authorizing standard scale."""

    predecessor = (
        canary_promotion.canonical_eef_velocity_recovery_v5_promotion_evidence()
    )
    canary_promotion.validate_eef_velocity_recovery_v5_promotion_evidence(predecessor)
    reasoning_tasks = _reasoning_tasks()
    official_tasks = _official_tasks()
    return {
        "schema_version": 1,
        "profile": PROMOTION_PROFILE,
        "status": PROMOTION_STATUS,
        "lineage": {
            "controller_producer_commit": canary_promotion.PRODUCER_POLARIS_COMMIT,
            "controller_producer_tree": canary_promotion.PRODUCER_POLARIS_TREE,
            "predecessor_promotion_commit": PROMOTION_BASE_COMMIT,
            "predecessor_promotion_tree": PROMOTION_BASE_TREE,
            "predecessor_evidence_source_path": PREDECESSOR_SOURCE_PATH,
            "predecessor_evidence_source_sha256": PREDECESSOR_SOURCE_SHA256,
            "predecessor_evidence_manifest_sha256": PREDECESSOR_EVIDENCE_SHA256,
            "runtime_parent_commit": PROMOTION_PARENT_COMMIT,
            "runtime_parent_tree": PROMOTION_PARENT_TREE,
            "smoke_ego_lap_commit": SMOKE_EGO_LAP_COMMIT,
            "runtime_descendant_attestation": deepcopy(RUNTIME_DESCENDANT_ATTESTATION),
            "controller_or_evaluator_semantics_changed_by_promotion": False,
        },
        "smoke_suites": {
            "reasoning_43075": {
                "root": REASONING_ROOT,
                "resolved_root": REASONING_RESOLVED_ROOT,
                "step_root": f"{REASONING_ROOT}/{REASONING_STEP_RELATIVE}",
                "checkpoint": {
                    "uri": (
                        "gs://v6_east1d/checkpoints/lap_oxe_magic_soup_reasoning_full/"
                        "oxe_magic_soup_reasoning_full_v2_flow_pred0_cf0_ckpt25_"
                        "v6_32_b512_s42_20260630/43075"
                    ),
                    "step": 43075,
                    "output_step": 43075,
                    "inference_subset_profile": "policy-inference-params-assets-v1",
                    "inference_subset_sha256": (
                        "bb9ea5bb041f689a08f914cac7dfe5d061c822ddbe87e292f9c7878a9d3bfc4d"
                    ),
                },
                "protocol_variant": (
                    "polaris-droid-link8-eefpose-suite-smoke1-q99repair-"
                    "controllercandidate-v5-iksafety-v5"
                ),
                "watcher_job_id": 1098870,
                "watcher_scheduler_state": "COMPLETED",
                "watcher_scheduler_exit_code": "0:0",
                "watcher_completed_at": "2026-07-04T12:11:15-07:00",
                "metrics": {
                    "tasks": 6,
                    "episodes_completed": 6,
                    "raw_rubric_successes": 0,
                    "task_valid_successes": 0,
                    "mean_progress": 3 / 28,
                    "numerical_failures": 0,
                },
                "suite_artifacts": _suite_artifacts(
                    REASONING_STEP_RELATIVE,
                    {
                        "eval_success": "7d5d7321b35f765bd0cda7c4359a5adebf21fba5d6bdf3118d76be2ffed0840c",
                        "registry_candidates": "0b78776aa43f417199500a1b2949285a7de86d1380c2e0f7cbbc8ae1154040ec",
                        "suite_summary": "80a0c2e5f7439af989e8d57e6ddf9decdac5cd28ab611fc20ce4ed1e05a3790b",
                    },
                ),
                "tasks": reasoning_tasks,
            },
            "official_lap3b": {
                "root": OFFICIAL_ROOT,
                "resolved_root": OFFICIAL_RESOLVED_ROOT,
                "step_root": f"{OFFICIAL_ROOT}/{OFFICIAL_STEP_RELATIVE}",
                "checkpoint": {
                    "uri": (
                        "hf://lihzha/LAP-3B@601db9c1ab4bcaf6dddb160c7b2dec589a67b730"
                    ),
                    "step": None,
                    "output_step": 0,
                    "content_manifest_sha256": (
                        "567cc3ff7d20f3f03913a6f11c3fa151f789e1c0118ed5af0eea24d9cc48f20e"
                    ),
                },
                "protocol_variant": (
                    "polaris-droid-link8-eefpose-suite-smoke1-r6rows-q99repair-"
                    "controllercandidate-v5-iksafety-v5"
                ),
                "watcher_job_id": 1098873,
                "watcher_scheduler_state": "COMPLETED",
                "watcher_scheduler_exit_code": "0:0",
                "watcher_completed_at": "2026-07-04T12:11:37-07:00",
                "metrics": {
                    "tasks": 6,
                    "episodes_completed": 6,
                    "raw_rubric_successes": 0,
                    "task_valid_successes": 0,
                    "mean_progress": 5 / 36,
                    "numerical_failures": 0,
                },
                "suite_artifacts": _suite_artifacts(
                    OFFICIAL_STEP_RELATIVE,
                    {
                        "eval_success": "cd2d050fa37c0a364dca67e123e3efe8f64eae516932254fa027e2c493e8ee64",
                        "registry_candidates": "10eccd6c015aba14d608dd797cfa35e7a6434aa1215b4492c77bd6741e8a7f0c",
                        "suite_summary": "8100e9e6f9e216208f1628d49d13dcaebd3f6c2ab17535c0828c27c1a761e058",
                    },
                ),
                "tasks": official_tasks,
            },
        },
        "inspection": {
            "fresh_full_task_complete_rechecks": {
                "validator_ego_lap_commit": SMOKE_EGO_LAP_COMMIT,
                "validated_attempts": 12,
                "expected_attempts": 12,
                "expected_rollouts_per_attempt": 1,
                "all_passed": True,
                "performed_on_utc": "2026-07-04",
            },
            "local_artifact_roots": [
                "/home/lzha/code/ego-lap/.codex_artifacts/"
                "polaris-v5-smoke-rerun-videos-20260704/first_pair",
                "/home/lzha/code/ego-lap/.codex_artifacts/"
                "polaris-v5-smoke-rerun-videos-20260704/second_pair",
                "/home/lzha/code/ego-lap/.codex_artifacts/"
                "polaris-v5-smoke-rerun-videos-20260704/third_pair",
            ],
            "rollout_pairs_inspected": 12,
            "raw_and_summary_video_files_fully_decoded": 24,
            "expected_video_files": 24,
            "video_codec": "h264",
            "video_pixel_format": "yuv420p",
            "video_fps": 15,
            "video_frames": 450,
            "video_duration_seconds": 30.0,
            "raw_video_dimensions": [448, 224],
            "summary_video_dimensions": [960, 608],
            "contact_sheet_frames_per_rollout": 9,
            "contact_sheets_viewed_at_original_resolution": True,
            "correct_task_and_camera_views": True,
            "blank_or_corrupt_views": 0,
            "physics_explosions": 0,
            "motion_stable_and_physically_plausible": True,
            "raw_positive_rollouts": 0,
            "raw_positive_rollouts_requiring_adjudication": 0,
            "all_task_specific_failure_findings_recorded": True,
        },
        "scientific_scope": {
            "smoke_rollouts_per_task": 1,
            "smoke_is_wiring_and_promotion_evidence_only": True,
            "smoke_establishes_standard_success_rate": False,
            "reasoning_smoke_raw_successes": "0/6",
            "official_smoke_raw_successes": "0/6",
        },
        "authorization": {
            "allowed_eval_scales": list(AUTHORIZED_EVAL_SCALES),
            "next_eval_scale": NEXT_EVAL_SCALE,
            "standard_authorized": True,
            "standard_protocol": {
                "benchmark": "polaris_droid_suite_v1",
                "tasks": list(CANONICAL_TASKS),
                "rollouts_per_task": STANDARD_ROLLOUTS_PER_TASK,
                "policy_steps": STANDARD_POLICY_STEPS,
                "policy_hz": STANDARD_POLICY_HZ,
                "control_mode": "absolute_end_effector_pose",
                "eef_frame": "panda_link8_relative_to_panda_link0",
                "environments": 1,
            },
            "requires_exact_manifest_validation": True,
            "controller_or_evaluator_behavior_change_authorized": False,
        },
    }


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


PROMOTION_EVIDENCE_SHA256 = _canonical_sha256(
    canonical_eef_velocity_recovery_v5_standard_promotion_evidence()
)
EXPECTED_PROMOTION_EVIDENCE_SHA256 = (
    "7fd7a390da9dbe61531bdc8de75f83867a2011a6685fa2cfe761b1d965aba458"
)
if PROMOTION_EVIDENCE_SHA256 != EXPECTED_PROMOTION_EVIDENCE_SHA256:
    raise RuntimeError("Canonical v5 standard promotion evidence drift")


def validate_eef_velocity_recovery_v5_standard_promotion_evidence(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless ``value`` is the exact standard-promotion manifest."""

    predecessor = (
        canary_promotion.canonical_eef_velocity_recovery_v5_promotion_evidence()
    )
    canary_promotion.validate_eef_velocity_recovery_v5_promotion_evidence(predecessor)
    expected = canonical_eef_velocity_recovery_v5_standard_promotion_evidence()
    if type(value) is not dict or value != expected:
        raise ValueError("V5 standard promotion evidence drift")
    if _canonical_sha256(value) != EXPECTED_PROMOTION_EVIDENCE_SHA256:
        raise ValueError("V5 standard promotion digest drift")
    return deepcopy(expected)


def _validated_relative_artifact_parts(
    *, relative_path: str, step_relative: str
) -> tuple[str, ...]:
    if type(relative_path) is not str or type(step_relative) is not str:
        raise ValueError("Artifact paths must be exact strings")
    relative = PurePosixPath(relative_path)
    step = PurePosixPath(step_relative)
    if (
        relative.is_absolute()
        or step.is_absolute()
        or not relative.parts
        or not step.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(part in {"", ".", ".."} for part in step.parts)
        or relative.as_posix() != relative_path
        or step.as_posix() != step_relative
        or relative.parts[: len(step.parts)] != step.parts
        or len(relative.parts) <= len(step.parts)
    ):
        raise ValueError(
            f"Artifact path escapes exact suite step root: {relative_path}"
        )
    return relative.parts


def _open_verified_root(path: Path) -> int:
    try:
        root_stat = path.lstat()
    except FileNotFoundError as exc:
        raise ValueError(f"Pinned artifact root is missing: {path}") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise ValueError(f"Pinned artifact root must not be a symlink: {path}")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError(f"Pinned artifact root is not a directory: {path}")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise ValueError(
            f"Could not securely open pinned artifact root: {path}"
        ) from exc


def _hash_regular_file_at(
    *, root_fd: int, root_display: Path, parts: tuple[str, ...]
) -> tuple[str, int, int]:
    directory_fd = os.dup(root_fd)
    file_fd: int | None = None
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
        for component in parts[:-1]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Pinned artifact path is missing below {root_display}: "
                    f"{PurePosixPath(*parts)}"
                ) from exc
            except OSError as exc:
                raise ValueError(
                    f"Pinned artifact parent is a symlink or non-directory below "
                    f"{root_display}: {PurePosixPath(*parts)}"
                ) from exc
            os.close(directory_fd)
            directory_fd = next_fd
        try:
            file_fd = os.open(
                parts[-1],
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )
        except FileNotFoundError as exc:
            raise ValueError(
                f"Pinned artifact is missing below {root_display}: "
                f"{PurePosixPath(*parts)}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Pinned artifact is a symlink or cannot be securely opened below "
                f"{root_display}: {PurePosixPath(*parts)}"
            ) from exc
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(
                f"Pinned artifact is not a regular file below {root_display}: "
                f"{PurePosixPath(*parts)}"
            )
        if before.st_nlink != 1:
            raise ValueError(
                f"Pinned artifact must have exactly one hard link below "
                f"{root_display}: {PurePosixPath(*parts)}"
            )
        digest = hashlib.sha256()
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(file_fd)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ValueError(
                f"Pinned artifact changed during verification below {root_display}: "
                f"{PurePosixPath(*parts)}"
            )
        if after.st_nlink != 1:
            raise ValueError(
                f"Pinned artifact hard-link count changed during verification below "
                f"{root_display}: {PurePosixPath(*parts)}"
            )
        return digest.hexdigest(), before.st_size, before.st_nlink
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def validate_eef_velocity_recovery_v5_standard_promotion_artifacts(
    evidence: Mapping[str, Any],
    *,
    root_overrides: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    """Freshly hash every pinned artifact and return a canonical verification.

    The standard authorization path always uses the exact NFS roots in the
    manifest. ``root_overrides`` exists for fail-closed tests and for an
    explicitly selected content-addressed mirror; it never changes canonical
    path identity in the verification digest.
    """

    validated = validate_eef_velocity_recovery_v5_standard_promotion_evidence(evidence)
    suites = validated["smoke_suites"]
    if root_overrides is not None:
        if type(root_overrides) is not dict or any(
            type(name) is not str or not isinstance(path, Path)
            for name, path in root_overrides.items()
        ):
            raise ValueError("Root overrides must map suite names to pathlib.Path")
        unknown = set(root_overrides) - set(suites)
        if unknown:
            raise ValueError(
                f"Unknown artifact root override suites: {sorted(unknown)}"
            )

    inventory: list[dict[str, Any]] = []
    verified_physical_roots: dict[str, str] = {}
    suite_counts: dict[str, int] = {}
    seen_relative_paths: set[tuple[str, str]] = set()
    for suite_name, suite in suites.items():
        canonical_root = Path(suite["root"])
        expected_resolved_root = Path(suite["resolved_root"])
        if root_overrides is not None and suite_name in root_overrides:
            physical_root = root_overrides[suite_name]
        else:
            try:
                physical_root = canonical_root.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Pinned canonical artifact root is missing: {canonical_root}"
                ) from exc
            if physical_root != expected_resolved_root:
                raise ValueError(
                    f"Pinned artifact root resolved-path drift for {suite_name}: "
                    f"{physical_root} != {expected_resolved_root}"
                )
        root_fd = _open_verified_root(physical_root)
        verified_physical_roots[suite_name] = str(physical_root)
        step_relative = suite["step_root"].removeprefix(f"{suite['root']}/")
        if f"{suite['root']}/{step_relative}" != suite["step_root"]:
            os.close(root_fd)
            raise ValueError(f"Suite step-root path drift: {suite_name}")
        suite_start = len(inventory)
        artifact_groups = [("suite", None, suite["suite_artifacts"])]
        artifact_groups.extend(
            ("task", task_name, task["artifacts"])
            for task_name, task in suite["tasks"].items()
        )
        try:
            for scope, task_name, artifacts in artifact_groups:
                if type(artifacts) is not dict:
                    raise ValueError(f"Artifact schema drift: {suite_name}/{task_name}")
                for artifact_name, artifact in artifacts.items():
                    if (
                        type(artifact_name) is not str
                        or type(artifact) is not dict
                        or set(artifact) != {"relative_path", "sha256"}
                        or type(artifact["sha256"]) is not str
                        or len(artifact["sha256"]) != 64
                    ):
                        raise ValueError(
                            f"Artifact identity schema drift: "
                            f"{suite_name}/{task_name}/{artifact_name}"
                        )
                    try:
                        int(artifact["sha256"], 16)
                    except ValueError as exc:
                        raise ValueError(
                            f"Artifact SHA-256 is not hexadecimal: "
                            f"{suite_name}/{task_name}/{artifact_name}"
                        ) from exc
                    parts = _validated_relative_artifact_parts(
                        relative_path=artifact["relative_path"],
                        step_relative=step_relative,
                    )
                    path_key = (suite_name, artifact["relative_path"])
                    if path_key in seen_relative_paths:
                        raise ValueError(
                            f"Duplicate pinned artifact path: {artifact['relative_path']}"
                        )
                    seen_relative_paths.add(path_key)
                    actual_sha256, size, nlink = _hash_regular_file_at(
                        root_fd=root_fd,
                        root_display=physical_root,
                        parts=parts,
                    )
                    if actual_sha256 != artifact["sha256"]:
                        raise ValueError(
                            f"Pinned artifact SHA-256 drift: "
                            f"{suite_name}/{task_name}/{artifact_name}"
                        )
                    inventory.append(
                        {
                            "suite": suite_name,
                            "scope": scope,
                            "task": task_name,
                            "artifact": artifact_name,
                            "canonical_path": (
                                f"{suite['root']}/{artifact['relative_path']}"
                            ),
                            "size": size,
                            "nlink": nlink,
                            "sha256": actual_sha256,
                        }
                    )
        finally:
            os.close(root_fd)
        suite_counts[suite_name] = len(inventory) - suite_start

    inventory.sort(
        key=lambda item: (
            item["suite"],
            item["scope"],
            item["task"] or "",
            item["artifact"],
            item["canonical_path"],
        )
    )
    expected_artifact_count = 2 * (3 + len(CANONICAL_TASKS) * 11)
    if len(inventory) != expected_artifact_count or set(suite_counts.values()) != {69}:
        raise ValueError(
            f"Pinned artifact count drift: {len(inventory)} != {expected_artifact_count}"
        )
    inventory_sha256 = _canonical_sha256({"artifacts": inventory})
    summary = {
        "schema_version": 1,
        "profile": "fresh_descriptor_relative_nofollow_sha256_walk_v1",
        "promotion_evidence_sha256": EXPECTED_PROMOTION_EVIDENCE_SHA256,
        "suite_count": len(suites),
        "task_count": sum(len(suite["tasks"]) for suite in suites.values()),
        "artifact_count": len(inventory),
        "suite_artifact_counts": suite_counts,
        "artifact_inventory_sha256": inventory_sha256,
        "canonical_roots": {name: suite["root"] for name, suite in suites.items()},
        "root_overrides_used": root_overrides is not None,
    }
    verification_sha256 = _canonical_sha256(summary)
    return {
        **summary,
        "verified_physical_roots": verified_physical_roots,
        "verification_sha256": verification_sha256,
    }


def validate_and_authorize_eef_velocity_recovery_v5_standard(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Authorize standard scale only after closed and fresh artifact checks."""

    validated = validate_eef_velocity_recovery_v5_standard_promotion_evidence(evidence)
    authorization = validated["authorization"]
    if (
        authorization["standard_authorized"] is not True
        or authorization["requires_exact_manifest_validation"] is not True
        or authorization["next_eval_scale"] != "standard"
    ):
        raise ValueError("Closed evidence does not authorize standard evaluation")
    artifact_verification = (
        validate_eef_velocity_recovery_v5_standard_promotion_artifacts(validated)
    )
    return {
        "standard_authorized": True,
        "promotion_evidence_sha256": EXPECTED_PROMOTION_EVIDENCE_SHA256,
        "artifact_verification": artifact_verification,
    }


def eef_velocity_recovery_v5_standard_eval_scale_allowed(
    evidence: Mapping[str, Any], eval_scale: str
) -> bool:
    """Authorize a scale only after strict validation of the closed evidence."""

    validated = validate_eef_velocity_recovery_v5_standard_promotion_evidence(evidence)
    if type(eval_scale) is not str:
        return False
    authorization = validated["authorization"]
    if eval_scale == "standard":
        validate_and_authorize_eef_velocity_recovery_v5_standard(validated)
    return eval_scale in authorization["allowed_eval_scales"]


def validate_promotion_lineage_source_identity(repo_root: Path) -> dict[str, Any]:
    """Reject predecessor or controller-source drift in the evidence descendant."""

    predecessor_path = repo_root / PREDECESSOR_SOURCE_PATH
    if not predecessor_path.is_file():
        raise ValueError(
            f"Missing predecessor evidence source: {PREDECESSOR_SOURCE_PATH}"
        )
    predecessor_digest = hashlib.sha256(predecessor_path.read_bytes()).hexdigest()
    if predecessor_digest != PREDECESSOR_SOURCE_SHA256:
        raise ValueError("Predecessor evidence source digest drift")
    producer_sources = canary_promotion.validate_producer_source_identity(repo_root)
    return {
        "predecessor_source_sha256": predecessor_digest,
        "producer_source_sha256": producer_sources,
    }
