#!/usr/bin/env python3
"""Publish immutable full-content verification for official ``pi05_droid``."""

from __future__ import annotations

import argparse
from pathlib import Path

from polaris.pi05_droid_jointvelocity_contract import (
    PI05_DROID_CHECKPOINT_URI,
    verify_pi05_droid_checkpoint,
)
from polaris.pi05_droid_native_eval_contract import (
    publish_immutable_json,
    verify_official_norm_reference_probes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify_pi05_droid_checkpoint(args.checkpoint, args.manifest, full_md5=True)
    norm_reference = verify_official_norm_reference_probes(
        args.checkpoint / "assets/droid/norm_stats.json"
    )
    publish_immutable_json(
        args.output,
        {
            "schema_version": 1,
            "status": "pass",
            "checkpoint_uri": PI05_DROID_CHECKPOINT_URI,
            "manifest_path": str(args.manifest.resolve()),
            **report,
            "norm_reference": norm_reference,
        },
    )


if __name__ == "__main__":
    main()
