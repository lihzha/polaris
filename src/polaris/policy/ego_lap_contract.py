"""Validation and persistence for Ego-LAP websocket serving metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Literal


EGO_LAP_SERVING_CONTRACT_KEY = "ego_lap_serving_contract"
SUPPORTED_SCHEMA_VERSION = 2
ORIGINAL_LAP_PROFILE = "original_lap_public_3b_v1"
FLOW_RESPONSE_SEMANTICS = "cumulative_delta_targets"
AR_RESPONSE_SEMANTICS = "total_delta_endpoint"
TRAIN_MATCHED_INPUT_FORMULA = "q99_input_eps1e-8_clip_zero0_v1"
TRAIN_MATCHED_OUTPUT_FORMULA = "q99_output_eps1e-8_zeroq01_extrapolate_v1"
LEGACY_INPUT_FORMULA = "q99_input_eps1e-6_no_clip_zero0_v1"
LEGACY_OUTPUT_FORMULA = "q99_output_eps1e-6_no_zero_override_extrapolate_v1"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclasses.dataclass(frozen=True)
class ValidatedEgoLAPContract:
    """Runtime values derived from one validated server contract."""

    document: Mapping[str, Any]
    contract_sha256: str
    checkpoint_profile: str
    checkpoint_path: str
    policy_type: Literal["flow", "ar"]
    response_horizon: int
    response_semantics: str
    execution_horizon: int
    interpolation_steps: int
    frame_description: str
    action_frame: Literal["robot_base", "egocentric"]
    dataset_name: str
    state_type: str
    rotate_wrist_180: bool
    normalization_scope: Literal["global", "category"]
    normalization_stats_sha256: str
    normalization_profile: str
    normalization_input_formula: str
    normalization_output_formula: str
    normalization_formula_probe_sha256: str
    polaris_profile: str


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Ego-LAP serving contract field {field!r} must be an object")
    return value


def _required_string(mapping: Mapping[str, Any], key: str, *, field: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Ego-LAP serving contract field {field}.{key} must be a nonempty string"
        )
    return value


def _require_equal(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise ValueError(
            f"Ego-LAP serving contract mismatch for {field}: "
            f"server={actual!r}, expected={expected!r}"
        )


def _require_zero(value: Any, *, field: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Ego-LAP serving contract field {field} must be numeric"
        ) from error
    if numeric != 0.0:
        raise ValueError(f"Ego-LAP inference requires {field}=0, got {numeric!r}")


def _normalize_checkpoint_path(path: str) -> str:
    return path.rstrip("/")


def _require_sequence(value: Any, expected: Sequence[Any], *, field: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Ego-LAP serving contract field {field} must be a sequence")
    _require_equal(list(value), list(expected), field=field)


def ego_lap_contract_digest(document: Mapping[str, Any]) -> str:
    """Return the serving contract identity, excluding its identity field."""

    payload = dict(document)
    payload.pop("sha256", None)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Ego-LAP serving contract is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def validate_ego_lap_server_metadata(
    server_metadata: Mapping[str, Any],
    *,
    expected_checkpoint_profile: str | None,
    expected_checkpoint_path: str | None,
    expected_policy_type: Literal["flow", "ar"] | None,
    expected_normalization_scope: Literal["global", "category"] | None,
    expected_normalization_stats_sha256: str | None,
    expected_normalization_profile: str | None,
    expected_normalization_input_formula: str | None,
    expected_normalization_output_formula: str | None,
    expected_frame_description: str | None,
    expected_action_frame: Literal["robot_base", "egocentric"] | None,
    expected_dataset_name: str,
    expected_state_type: str,
    expected_open_loop_horizon: int | None,
    ar_interpolation_steps: int,
) -> ValidatedEgoLAPContract:
    """Validate metadata and derive the only runtime values the client may use.

    Optional ``expected_*`` values are caller assertions. The returned values
    always come from the server contract, never from those assertions.
    """

    if not isinstance(server_metadata, Mapping):
        raise ValueError("Ego-LAP websocket metadata must be an object")
    document = _mapping(
        server_metadata.get(EGO_LAP_SERVING_CONTRACT_KEY),
        field=EGO_LAP_SERVING_CONTRACT_KEY,
    )
    _require_equal(
        document.get("schema_version"), SUPPORTED_SCHEMA_VERSION, field="schema_version"
    )

    checkpoint_profile = _required_string(
        document, "checkpoint_profile", field="contract"
    )
    checkpoint_path = _required_string(document, "checkpoint_path", field="contract")
    policy_type = _required_string(document, "policy_type", field="contract")
    if policy_type not in {"flow", "ar"}:
        raise ValueError(
            f"Unsupported Ego-LAP policy type in serving contract: {policy_type!r}"
        )

    if expected_checkpoint_profile is not None:
        _require_equal(
            checkpoint_profile, expected_checkpoint_profile, field="checkpoint_profile"
        )
    if expected_checkpoint_path is not None:
        _require_equal(
            _normalize_checkpoint_path(checkpoint_path),
            _normalize_checkpoint_path(expected_checkpoint_path),
            field="checkpoint_path",
        )
    if expected_policy_type is not None:
        _require_equal(policy_type, expected_policy_type, field="policy_type")
    _require_equal(
        document.get("checkpoint_manifest_validated"),
        checkpoint_profile != ORIGINAL_LAP_PROFILE,
        field="checkpoint_manifest_validated",
    )

    model = _mapping(document.get("model"), field="model")
    _require_equal(model.get("action_dim"), 7, field="model.action_dim")
    _require_equal(model.get("action_horizon"), 16, field="model.action_horizon")
    _require_equal(model.get("state_dim"), 10, field="model.state_dim")
    _require_sequence(
        model.get("image_resolution"), [224, 224], field="model.image_resolution"
    )
    _require_equal(model.get("prompt_format"), "lap", field="model.prompt_format")
    if policy_type == "ar":
        _require_equal(
            model.get("enable_langact_training"),
            True,
            field="model.enable_langact_training",
        )

    if checkpoint_profile == ORIGINAL_LAP_PROFILE:
        expected_image_keys = ["base_0_rgb", "left_wrist_0_rgb"]
        expected_image_order = ["external", "wrist"]
    else:
        expected_image_keys = ["camera_0_rgb", "camera_1_rgb", "camera_2_rgb"]
        expected_image_order = ["wrist", "external", "blank"]
    _require_sequence(
        model.get("model_image_keys"),
        expected_image_keys,
        field="model.model_image_keys",
    )

    policy_input = _mapping(document.get("policy_input"), field="policy_input")
    _require_equal(
        policy_input.get("primary_image_key"),
        "base_0_rgb",
        field="policy_input.primary_image_key",
    )
    _require_equal(
        policy_input.get("wrist_image_key"),
        "left_wrist_0_rgb",
        field="policy_input.wrist_image_key",
    )
    _require_equal(
        policy_input.get("image_color_space"),
        "RGB",
        field="policy_input.image_color_space",
    )
    _require_equal(
        policy_input.get("image_dtype"), "uint8", field="policy_input.image_dtype"
    )
    _require_sequence(
        policy_input.get("image_resolution"),
        [224, 224],
        field="policy_input.image_resolution",
    )
    _require_sequence(
        policy_input.get("model_image_order"),
        expected_image_order,
        field="policy_input.model_image_order",
    )
    _require_equal(
        policy_input.get("wrist_rotation_degrees"),
        180,
        field="policy_input.wrist_rotation_degrees",
    )
    dataset_name = _required_string(policy_input, "dataset_name", field="policy_input")
    state_type = _required_string(
        policy_input, "request_state_type", field="policy_input"
    )
    _require_equal(
        dataset_name, expected_dataset_name, field="policy_input.dataset_name"
    )
    _require_equal(
        state_type, expected_state_type, field="policy_input.request_state_type"
    )
    _require_equal(
        policy_input.get("is_bimanual"), False, field="policy_input.is_bimanual"
    )
    _require_equal(
        policy_input.get("state_encoding"),
        "EEF_R6",
        field="policy_input.state_encoding",
    )
    _require_equal(
        policy_input.get("state_layout"),
        "xyz+r6_first_two_columns+gripper_open",
        field="policy_input.state_layout",
    )
    _require_equal(
        policy_input.get("gripper_open_value"),
        1.0,
        field="policy_input.gripper_open_value",
    )
    _require_equal(
        policy_input.get("gripper_closed_value"),
        0.0,
        field="policy_input.gripper_closed_value",
    )

    policy_output = _mapping(document.get("policy_output"), field="policy_output")
    _require_equal(
        policy_output.get("action_encoding"),
        "EEF_POS",
        field="policy_output.action_encoding",
    )
    _require_equal(policy_output.get("action_dim"), 7, field="policy_output.action_dim")
    _require_equal(
        policy_output.get("model_action_horizon"),
        16,
        field="policy_output.model_action_horizon",
    )
    _require_equal(
        policy_output.get("action_layout"),
        "delta_xyz+delta_extrinsic_xyz_euler+gripper_open",
        field="policy_output.action_layout",
    )
    action_frame = _required_string(
        policy_output, "translation_frame", field="policy_output"
    )
    if action_frame not in {"robot_base", "egocentric"}:
        raise ValueError(f"Unsupported Ego-LAP numeric action frame: {action_frame!r}")
    if expected_action_frame is not None:
        _require_equal(
            action_frame, expected_action_frame, field="policy_output.translation_frame"
        )
    _require_equal(
        policy_output.get("rotation_representation"),
        "extrinsic_xyz_euler_delta",
        field="policy_output.rotation_representation",
    )
    _require_equal(
        policy_output.get("rotation_composition"),
        "right_multiply_current_by_delta",
        field="policy_output.rotation_composition",
    )
    _require_equal(
        policy_output.get("translation_units"),
        "meters",
        field="policy_output.translation_units",
    )
    _require_equal(
        policy_output.get("rotation_units"),
        "radians",
        field="policy_output.rotation_units",
    )
    _require_equal(
        policy_output.get("gripper_open_value"),
        1.0,
        field="policy_output.gripper_open_value",
    )
    _require_equal(
        policy_output.get("gripper_closed_value"),
        0.0,
        field="policy_output.gripper_closed_value",
    )

    response_horizon = policy_output.get("response_horizon")
    response_semantics = policy_output.get("response_semantics")
    if policy_type == "flow":
        _require_equal(response_horizon, 16, field="policy_output.response_horizon")
        _require_equal(
            response_semantics,
            FLOW_RESPONSE_SEMANTICS,
            field="policy_output.response_semantics",
        )
        execution_horizon = 8
        interpolation_steps = 16
    else:
        _require_equal(response_horizon, 1, field="policy_output.response_horizon")
        _require_equal(
            response_semantics,
            AR_RESPONSE_SEMANTICS,
            field="policy_output.response_semantics",
        )
        if ar_interpolation_steps != 16:
            raise ValueError(
                "Ego-LAP AR evaluation requires ar_interpolation_steps=16; "
                f"got {ar_interpolation_steps}"
            )
        execution_horizon = 4
        interpolation_steps = ar_interpolation_steps
    if expected_open_loop_horizon is not None:
        _require_equal(
            expected_open_loop_horizon, execution_horizon, field="open_loop_horizon"
        )

    language_action = _mapping(document.get("language_action"), field="language_action")
    _require_equal(
        language_action.get("format"),
        "verbose_eef_with_rotation",
        field="language_action.format",
    )
    frame_description = _required_string(
        language_action, "frame_description", field="language_action"
    )
    if frame_description not in {"robot base frame", "egocentric frame"}:
        raise ValueError(f"Unsupported Ego-LAP language frame: {frame_description!r}")
    if expected_frame_description is not None:
        _require_equal(
            frame_description,
            expected_frame_description,
            field="language_action.frame_description",
        )

    normalization = _mapping(document.get("normalization"), field="normalization")
    _require_equal(
        normalization.get("source"), "checkpoint_assets", field="normalization.source"
    )
    _require_equal(normalization.get("type"), "bounds_q99", field="normalization.type")
    normalization_scope = _required_string(
        normalization, "scope", field="normalization"
    )
    if normalization_scope not in {"global", "category"}:
        raise ValueError(
            f"Unsupported Ego-LAP normalization scope: {normalization_scope!r}"
        )
    if expected_normalization_scope is not None:
        _require_equal(
            normalization_scope,
            expected_normalization_scope,
            field="normalization.scope",
        )
    if normalization_scope == "category":
        _require_equal(
            normalization.get("policy_category"),
            "single_arm",
            field="normalization.policy_category",
        )
    normalization_stats_sha256 = _required_string(
        normalization,
        "selected_stats_sha256",
        field="normalization",
    )
    if not _SHA256_PATTERN.fullmatch(normalization_stats_sha256):
        raise ValueError(
            "Ego-LAP selected normalization digest must be a lowercase SHA-256"
        )
    if expected_normalization_stats_sha256 is not None:
        _require_equal(
            normalization_stats_sha256,
            expected_normalization_stats_sha256.lower(),
            field="normalization.selected_stats_sha256",
        )
    selected_stats_keys = normalization.get("selected_stats_keys")
    if (
        not isinstance(selected_stats_keys, Sequence)
        or isinstance(selected_stats_keys, (str, bytes))
        or not selected_stats_keys
    ):
        raise ValueError(
            "normalization.selected_stats_keys must be a nonempty sequence"
        )
    formula_schema_version = normalization.get("formula_schema_version")
    if not isinstance(formula_schema_version, int) or formula_schema_version < 1:
        raise ValueError(
            "normalization.formula_schema_version must be a positive integer"
        )
    normalization_profile = _required_string(
        normalization,
        "formula_profile",
        field="normalization",
    )
    normalization_input_formula = _required_string(
        normalization,
        "input_formula_id",
        field="normalization",
    )
    normalization_output_formula = _required_string(
        normalization,
        "output_formula_id",
        field="normalization",
    )
    formula_pair = (normalization_input_formula, normalization_output_formula)
    if formula_pair not in {
        (TRAIN_MATCHED_INPUT_FORMULA, TRAIN_MATCHED_OUTPUT_FORMULA),
        (LEGACY_INPUT_FORMULA, LEGACY_OUTPUT_FORMULA),
    }:
        raise ValueError(f"Unsupported Ego-LAP Q99 formula pair: {formula_pair!r}")
    normalization_formula_probe_sha256 = _required_string(
        normalization,
        "formula_probe_sha256",
        field="normalization",
    )
    if not _SHA256_PATTERN.fullmatch(normalization_formula_probe_sha256):
        raise ValueError(
            "normalization.formula_probe_sha256 must be a lowercase SHA-256"
        )
    for actual, expected, field in (
        (
            normalization_profile,
            expected_normalization_profile,
            "normalization.formula_profile",
        ),
        (
            normalization_input_formula,
            expected_normalization_input_formula,
            "normalization.input_formula_id",
        ),
        (
            normalization_output_formula,
            expected_normalization_output_formula,
            "normalization.output_formula_id",
        ),
    ):
        if expected is not None:
            _require_equal(actual, expected, field=field)

    execution = _mapping(document.get("execution"), field="execution")
    _require_equal(
        execution.get("live_pipeline_validated"),
        True,
        field="execution.live_pipeline_validated",
    )
    _require_equal(execution.get("schema_version"), 2, field="execution.schema_version")
    inference_config = _mapping(
        execution.get("inference_data_config"), field="execution.inference_data_config"
    )
    _require_zero(
        inference_config.get("wrist_image_dropout_prob"),
        field="wrist_image_dropout_prob",
    )
    _require_zero(
        inference_config.get("mask_zero_img_prob"), field="mask_zero_img_prob"
    )
    _require_zero(inference_config.get("state_dropout"), field="state_dropout")
    image_routing = _mapping(
        execution.get("image_routing"), field="execution.image_routing"
    )
    _require_sequence(
        image_routing.get("model_image_order"),
        expected_image_order,
        field="execution.image_routing.model_image_order",
    )
    droid_preprocessing = _mapping(
        execution.get("droid_image_preprocessing"),
        field="execution.droid_image_preprocessing",
    )
    for key, expected in (
        ("dataset_name", "droid"),
        ("registry_requires_wrist_rotation", True),
        ("not_rotate_wrist_prob", 0.0),
        ("resize_resolution", [224, 224]),
        ("rotation_applied", True),
        ("wrist_rotation_degrees", 180),
    ):
        _require_equal(
            droid_preprocessing.get(key),
            expected,
            field=f"execution.droid_image_preprocessing.{key}",
        )
    ar_roundtrip = _mapping(
        execution.get("ar_training_roundtrip_probe"),
        field="execution.ar_training_roundtrip_probe",
    )
    _require_equal(
        ar_roundtrip.get("dataset_name"),
        "droid",
        field="execution.ar_training_roundtrip_probe.dataset_name",
    )
    _require_equal(
        ar_roundtrip.get("matches"),
        True,
        field="execution.ar_training_roundtrip_probe.matches",
    )

    formula_execution = _mapping(
        execution.get("normalization_formula"),
        field="execution.normalization_formula",
    )
    _require_equal(
        formula_execution.get("schema_version"),
        formula_schema_version,
        field="execution.normalization_formula.schema_version",
    )
    _require_equal(
        formula_execution.get("profile"),
        normalization_profile,
        field="execution.normalization_formula.profile",
    )
    _require_equal(
        formula_execution.get("input_formula_id"),
        normalization_input_formula,
        field="execution.normalization_formula.input_formula_id",
    )
    _require_equal(
        formula_execution.get("output_formula_id"),
        normalization_output_formula,
        field="execution.normalization_formula.output_formula_id",
    )
    _require_equal(
        formula_execution.get("sha256"),
        normalization_formula_probe_sha256,
        field="execution.normalization_formula.sha256",
    )
    input_probe = _mapping(
        formula_execution.get("training_policy_input_probe"),
        field="execution.normalization_formula.training_policy_input_probe",
    )
    roundtrip_probe = _mapping(
        formula_execution.get("training_policy_roundtrip_probe"),
        field="execution.normalization_formula.training_policy_roundtrip_probe",
    )
    extrapolation_probe = _mapping(
        formula_execution.get("output_extrapolation_probe"),
        field="execution.normalization_formula.output_extrapolation_probe",
    )
    _require_equal(
        extrapolation_probe.get("extrapolates_beyond_q01_q99"),
        True,
        field="execution.normalization_formula.output_extrapolation_probe.extrapolates_beyond_q01_q99",
    )
    if normalization_input_formula == TRAIN_MATCHED_INPUT_FORMULA:
        for probe, probe_name in (
            (input_probe, "training_policy_input_probe"),
            (roundtrip_probe, "training_policy_roundtrip_probe"),
        ):
            _require_equal(
                probe.get("matches"),
                True,
                field=f"execution.normalization_formula.{probe_name}.matches",
            )
        _require_equal(
            extrapolation_probe.get("zero_range_is_exact_q01"),
            True,
            field="execution.normalization_formula.output_extrapolation_probe.zero_range_is_exact_q01",
        )

    polaris = _mapping(document.get("polaris"), field="polaris")
    polaris_profile = _required_string(polaris, "profile", field="polaris")
    _require_equal(polaris.get("compatible"), True, field="polaris.compatible")
    _require_sequence(
        polaris.get("incompatibilities"),
        [],
        field="polaris.incompatibilities",
    )
    _require_equal(polaris.get("eef_frame"), "panda_link8", field="polaris.eef_frame")
    _require_equal(
        polaris.get("normalization_scope"),
        normalization_scope,
        field="polaris.normalization_scope",
    )
    expected_polaris_category = (
        "single_arm" if normalization_scope == "category" else None
    )
    _require_equal(
        polaris.get("normalization_category"),
        expected_polaris_category,
        field="polaris.normalization_category",
    )
    _require_equal(
        polaris.get("q99_formula_profile"),
        normalization_profile,
        field="polaris.q99_formula_profile",
    )
    _require_equal(
        polaris.get("numeric_action_frame"),
        action_frame,
        field="polaris.numeric_action_frame",
    )

    contract_sha256 = _required_string(document, "sha256", field="contract")
    if not _SHA256_PATTERN.fullmatch(contract_sha256):
        raise ValueError("Ego-LAP serving contract sha256 must be a lowercase SHA-256")
    _require_equal(
        contract_sha256,
        ego_lap_contract_digest(document),
        field="sha256",
    )

    return ValidatedEgoLAPContract(
        document=document,
        contract_sha256=contract_sha256,
        checkpoint_profile=checkpoint_profile,
        checkpoint_path=checkpoint_path,
        policy_type=policy_type,
        response_horizon=int(response_horizon),
        response_semantics=str(response_semantics),
        execution_horizon=execution_horizon,
        interpolation_steps=interpolation_steps,
        frame_description=frame_description,
        action_frame=action_frame,
        dataset_name=dataset_name,
        state_type=state_type,
        rotate_wrist_180=True,
        normalization_scope=normalization_scope,
        normalization_stats_sha256=normalization_stats_sha256,
        normalization_profile=normalization_profile,
        normalization_input_formula=normalization_input_formula,
        normalization_output_formula=normalization_output_formula,
        normalization_formula_probe_sha256=normalization_formula_probe_sha256,
        polaris_profile=polaris_profile,
    )


def persist_ego_lap_contract(
    document: Mapping[str, Any], output_path: str | os.PathLike[str]
) -> Path:
    """Atomically persist the exact validated contract, refusing resume drift."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Existing Ego-LAP contract is unreadable: {path}"
            ) from error
        if existing != document:
            raise ValueError(
                f"Refusing to replace a different Ego-LAP serving contract: {path}"
            )
        return path

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
