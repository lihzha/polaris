from __future__ import annotations

import copy
from pathlib import Path
import shutil

import pytest

from scripts import generate_eef_pose_canary_trace_fixtures as generator


def _require_source_traces() -> Path:
    root = generator.DEFAULT_TRACE_ROOT
    if not all(
        generator._trace_path(root, config).is_file()
        for config in generator.VARIANTS.values()
    ):
        pytest.skip("exact mirrored Gate-0 source traces are not present")
    return root


def test_exact_source_traces_regenerate_committed_fixtures(tmp_path: Path) -> None:
    trace_root = _require_source_traces()
    generator.generate_all(trace_root, tmp_path, check=False)
    for config in generator.VARIANTS.values():
        generated = tmp_path / config["filename"]
        committed = generator.DEFAULT_OUTPUT_DIR / config["filename"]
        assert generated.read_bytes() == committed.read_bytes()


@pytest.mark.parametrize("variant", sorted(generator.VARIANTS))
def test_fixture_replay_is_only_the_eight_action_query14_execute_window(
    variant: str,
) -> None:
    trace_root = _require_source_traces()
    config = generator.VARIANTS[variant]
    payload = generator.build_fixture(
        variant, generator._trace_path(trace_root, config)
    )
    plan = payload["action_plan"]
    assert plan["replay_action_count"] == 120
    assert plan["query14_start_step"] == 112
    assert plan["query14_executable_action_count"] == 8
    assert plan["query14_planned_action_count"] == 16
    assert (
        plan["planned_continuation_action_count"] == 120 - plan["observed_action_count"]
    )
    assert plan["continuation_semantics"].startswith("recorded_query_execute_window")


def test_official_fixture_contains_arrival_tail_but_no_synthetic_step120() -> None:
    trace_root = _require_source_traces()
    config = generator.VARIANTS["official_lap3b"]
    payload = generator.build_fixture(
        "official_lap3b", generator._trace_path(trace_root, config)
    )
    assert payload["action_plan"]["observed_action_count"] == 118
    assert payload["action_plan"]["planned_continuation_action_count"] == 2
    assert payload["action_encoding"]["action_count"] == 120


def test_source_trace_byte_tamper_fails_before_generation(tmp_path: Path) -> None:
    trace_root = _require_source_traces()
    config = generator.VARIANTS["official_lap3b"]
    source = generator._trace_path(trace_root, config)
    tampered = tmp_path / source.name
    shutil.copy2(source, tampered)
    data = bytearray(tampered.read_bytes())
    data[-2] ^= 1
    tampered.write_bytes(data)
    with pytest.raises(generator.FixtureGenerationError, match="digest drift"):
        generator.build_fixture("official_lap3b", tampered)


def test_query_contract_tamper_fails_even_with_rebound_trace_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_root = _require_source_traces()
    config = generator.VARIANTS["reasoning_43075"]
    source = generator._trace_path(trace_root, config)
    lines = source.read_text(encoding="utf-8").splitlines()
    query_line = next(
        index for index, line in enumerate(lines) if '"event":"query"' in line
    )
    lines[query_line] = lines[query_line].replace(
        '"normalization_scope":"global"', '"normalization_scope":"single_arm"'
    )
    tampered = tmp_path / source.name
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rebound = copy.deepcopy(config)
    data = tampered.read_bytes()
    rebound["trace_size_bytes"] = len(data)
    rebound["trace_sha256"] = generator._sha256(data)
    monkeypatch.setitem(generator.VARIANTS, "reasoning_43075", rebound)
    with pytest.raises(
        generator.FixtureGenerationError, match="query 0 contract drift"
    ):
        generator.build_fixture("reasoning_43075", tampered)
