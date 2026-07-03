#!/usr/bin/env python3
"""Post-job verify one immutable controller-candidate failure transaction."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import smoke_eef_pose_canary_trace_replay as gate0
import validate_eef_pose_canary_controller_candidate as validator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(validator.candidate.CANDIDATE_BY_VARIANT),
        required=True,
    )
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--raw-result", type=Path, required=True)
    parser.add_argument("--polaris-repo", type=Path, required=True)
    parser.add_argument("--expected-polaris-commit", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-validator-sha256", required=True)
    parser.add_argument("--expected-failure-verifier-sha256", required=True)
    parser.add_argument("--expected-safety-validator-sha256", required=True)
    parser.add_argument("--expected-gate0-helper-sha256", required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--container-image", type=Path, required=True)
    parser.add_argument("--expected-container-size-bytes", type=int, required=True)
    parser.add_argument("--expected-container-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.failure_verifier = Path(__file__)
    try:
        result = validator.validate_failure(args)
    except (
        OSError,
        UnicodeError,
        ValueError,
        gate0.Gate0ReplayValidationError,
        subprocess.CalledProcessError,
    ) as error:
        print(
            f"POLARIS_CONTROLLER_CANDIDATE_FAILURE_VERIFY_FAIL={error}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        "POLARIS_CONTROLLER_CANDIDATE_FAILURE_VERIFY_PASS="
        f"{result['variant']};job={result['job_id']};"
        f"raw_sha256={result['raw_result']['sha256']};"
        f"joint={result['joint_name']};policy_step={result['policy_step']};"
        f"physics_substep={result['physics_substep']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
