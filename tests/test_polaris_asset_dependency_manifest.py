import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/polaris/polaris_asset_dependency_manifest.py"
)
SPEC = importlib.util.spec_from_file_location("polaris_asset_manifest", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _fake_asset_tree(root: Path) -> None:
    task = root / "food_bussing"
    robot = root / "nvidia_droid"
    (task / "assets/bowl").mkdir(parents=True)
    (task / "initial_conditions.json").write_text("{}\n")
    (task / "scene.usda").write_text(
        '#usda 1.0\ndef Xform "bowl" (prepend payload = @./assets/bowl/mesh.usdz@) {}\n'
    )
    (task / "assets/bowl/mesh.usdz").write_bytes(b"payload")
    (task / "assets/bowl/splat.ply").write_bytes(b"task splat")
    (robot / "SEGMENTED").mkdir(parents=True)
    (robot / "noninstanceable.usd").write_bytes(b"robot usd")
    (robot / "splat.ply").write_bytes(b"robot splat")
    (robot / "SEGMENTED/panda_link0.ply").write_bytes(b"segment")


def _expected(root: Path) -> dict:
    scanned = MODULE.scan_asset_tree(root, "food_bussing")
    return {key: scanned[key] for key in ("file_count", "total_bytes", "tree_sha256")}


def test_complete_tree_manifest_binds_payloads_splats_and_segmented_robot(tmp_path):
    data = tmp_path / "data"
    _fake_asset_tree(data)
    expected = _expected(data)
    manifest = MODULE.make_asset_manifest(data, "food_bussing", expected=expected)
    records = {record["relative_path"]: record for record in manifest["records"]}
    assert (
        "usd_layer_or_payload" in records["food_bussing/assets/bowl/mesh.usdz"]["roles"]
    )
    assert "splat_runtime" in records["food_bussing/assets/bowl/splat.ply"]["roles"]
    assert (
        "robot_segmented_splat"
        in records["nvidia_droid/SEGMENTED/panda_link0.ply"]["roles"]
    )
    assert manifest["file_count"] == 7
    assert manifest["tree_sha256"] == expected["tree_sha256"]


def test_manifest_artifact_is_immutable_and_live_tree_verification_is_closed(
    tmp_path,
):
    data = tmp_path / "data"
    _fake_asset_tree(data)
    expected = _expected(data)
    manifest = MODULE.make_asset_manifest(data, "food_bussing", expected=expected)
    destination = tmp_path / "asset_manifest.json"
    artifact = MODULE.publish_asset_manifest(destination, manifest, expected=expected)
    assert artifact["mode"] == "0444"
    assert artifact["nlink"] == 1
    assert (
        MODULE.validate_asset_manifest_artifact(
            destination, data_root=data, expected=expected
        )["status"]
        == "pass"
    )
    (data / "food_bussing/unreviewed.bin").write_bytes(b"drift")
    with pytest.raises(ValueError, match="differs from immutable manifest"):
        MODULE.validate_asset_manifest_artifact(
            destination, data_root=data, expected=expected
        )
    with pytest.raises(FileExistsError):
        MODULE.publish_asset_manifest(destination, manifest, expected=expected)


def test_scan_rejects_symlinked_runtime_assets(tmp_path):
    data = tmp_path / "data"
    _fake_asset_tree(data)
    target = data / "food_bussing/assets/bowl/mesh.usdz"
    link = data / "food_bussing/assets/bowl/linked.usdz"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        MODULE.scan_asset_tree(data, "food_bussing")


def test_manifest_rejects_tampering_even_when_aggregate_fields_are_unchanged(
    tmp_path,
):
    data = tmp_path / "data"
    _fake_asset_tree(data)
    expected = _expected(data)
    manifest = MODULE.make_asset_manifest(data, "food_bussing", expected=expected)
    manifest["records"][0]["roles"] = ["task_tree"]
    with pytest.raises(ValueError, match="role mismatch"):
        MODULE.validate_asset_manifest(manifest, expected=expected)


def test_pinned_tree_constants_cover_every_supported_eval_task():
    assert MODULE.EXPECTED_ASSET_TREES == {
        "block_stack_kitchen": {
            "file_count": 38,
            "total_bytes": 483_960_124,
            "tree_sha256": (
                "f5f1fe057fc5daf7edc4b597eebedf5ac02796509cebee481059422b875e3c1c"
            ),
        },
        "food_bussing": {
            "file_count": 36,
            "total_bytes": 380_426_267,
            "tree_sha256": (
                "36b80c4a9499b0643bec2775e6c33a0517212b494065cba6c1f94e511f9fd094"
            ),
        },
        "pan_clean": {
            "file_count": 38,
            "total_bytes": 385_693_281,
            "tree_sha256": (
                "2a2f3a46821371564dbefe22af256553e58a5d6116f7b795abb0289bf4e5871a"
            ),
        },
        "move_latte_cup": {
            "file_count": 62,
            "total_bytes": 471_416_390,
            "tree_sha256": (
                "41c6b70bb3d5cc25f46ea15539308800cc9bb2abdcb168452541065b30f9173f"
            ),
        },
        "organize_tools": {
            "file_count": 53,
            "total_bytes": 376_291_177,
            "tree_sha256": (
                "3e5039dcab1a78dba193e2cdc3e460095dd4e119f01fd5a7b702732947d2cce6"
            ),
        },
        "tape_into_container": {
            "file_count": 44,
            "total_bytes": 374_778_953,
            "tree_sha256": (
                "de58ccda5f7090c3c128989f01be893a171349b632035cd50bfa04462949f87c"
            ),
        },
    }
    # The expected JSON is itself stable and can be embedded in launch records.
    json.dumps(MODULE.EXPECTED_ASSET_TREES, sort_keys=True, allow_nan=False)


def test_worker_builds_then_rechecks_and_final_manifest_binds_asset_closure():
    root = Path(__file__).parents[1]
    worker = (root / "scripts/polaris/eval_pi05_droid_jointpos_polaris.sh").read_text()
    evidence = (root / "src/polaris/pi05_droid_jointpos_evidence.py").read_text()
    dry_run = worker.index('if [[ "${DRY_RUN}" == 1 ]]')
    build = worker.index('asset_manifest_result="$', dry_run)
    server = worker.index("Starting official pi0.5 policy server", build)
    verify = worker.index('final_asset_manifest_result="$', server)
    rng_finalize = worker.index('kill -USR1 "${rng_server_pid}"', verify)
    assert dry_run < build < server < verify < rng_finalize
    assert (
        '"asset_dependency_manifest": "polaris_asset_dependency_manifest.json"'
        in evidence
    )
    assert "asset_module.validate_asset_manifest_artifact(" in evidence
