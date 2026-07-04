import dataclasses
from pathlib import Path

import pytest

from scripts.polaris.serve_pi05_droid_native_jointvelocity import (
    validate_official_pi05_data_config,
    validate_official_pi05_policy_runtime,
    validate_official_pi05_train_config,
)

from openpi import transforms
from openpi.training import config


def _official_data_config():
    train_config = config.get_config("pi05_droid")
    return train_config.data.create(train_config.assets_dirs, train_config.model)


def test_official_runtime_transform_pipeline_has_no_action_conversion():
    report = validate_official_pi05_data_config(_official_data_config())

    assert report["asset_id"] == "droid"
    assert report["use_quantile_norm"] is True
    assert report["data_inputs"] == ["openpi.policies.droid_policy.DroidInputs"]
    assert report["data_outputs"] == ["openpi.policies.droid_policy.DroidOutputs"]
    assert report["forbidden_transforms_absent"] == [
        "openpi.transforms.DeltaActions",
        "openpi.transforms.AbsoluteActions",
    ]
    assert report["resize"] == [224, 224]
    assert report["output_projection"] == "DroidOutputs_leading8"


def test_official_train_config_and_policy_runtime_are_type_exact():
    train_config = config.get_config("pi05_droid")
    assert validate_official_pi05_train_config(train_config)["action_horizon"] == 15
    assert validate_official_pi05_policy_runtime(
        metadata={}, sample_kwargs={}, rng_key_data=[0, 0]
    )["rng_key_data"] == [0, 0]

    for field, drifted in (("action_horizon", 15.0), ("action_dim", 32.0)):
        model = dataclasses.replace(train_config.model, **{field: drifted})
        with pytest.raises(ValueError, match="config mismatch"):
            validate_official_pi05_train_config(
                dataclasses.replace(train_config, model=model)
            )

    for rng_key_data in ([0.0, 0], [False, 0]):
        with pytest.raises(ValueError, match="policy runtime mismatch"):
            validate_official_pi05_policy_runtime(
                metadata={}, sample_kwargs={}, rng_key_data=rng_key_data
            )


def test_runtime_transform_pipeline_rejects_category_delta_absolute_and_resize():
    data_config = _official_data_config()
    with pytest.raises(ValueError, match="transform pipeline mismatch"):
        validate_official_pi05_data_config(
            dataclasses.replace(data_config, asset_id="single_arm")
        )

    data_config.data_transforms.inputs.append(transforms.DeltaActions(mask=None))
    with pytest.raises(ValueError, match="transform pipeline mismatch"):
        validate_official_pi05_data_config(data_config)
    data_config.data_transforms.inputs.pop()

    data_config.data_transforms.outputs.append(transforms.AbsoluteActions(mask=None))
    with pytest.raises(ValueError, match="transform pipeline mismatch"):
        validate_official_pi05_data_config(data_config)
    data_config.data_transforms.outputs.pop()

    original_resize = data_config.model_transforms.inputs[1]
    data_config.model_transforms.inputs[1] = dataclasses.replace(
        original_resize, height=256
    )
    with pytest.raises(ValueError, match="transform parameters mismatch"):
        validate_official_pi05_data_config(data_config)


@pytest.mark.parametrize(
    ("transform_index", "field", "drifted_value"),
    [
        (1, "height", 224.0),
        (1, "width", 224.0),
        (3, "model_action_dim", 32.0),
    ],
)
def test_runtime_transform_dimensions_are_type_exact(
    transform_index, field, drifted_value
):
    data_config = _official_data_config()
    transform = data_config.model_transforms.inputs[transform_index]
    data_config.model_transforms.inputs[transform_index] = dataclasses.replace(
        transform, **{field: drifted_value}
    )
    with pytest.raises(ValueError, match="transform parameters mismatch"):
        validate_official_pi05_data_config(data_config)


def test_runtime_transform_observed_subset_is_canonical_type_exact():
    data_config = dataclasses.replace(_official_data_config(), use_quantile_norm=1)
    assert type(data_config.use_quantile_norm) is int
    assert data_config.use_quantile_norm == 1
    with pytest.raises(ValueError, match="transform pipeline mismatch"):
        validate_official_pi05_data_config(data_config)


def test_authoritative_server_artifacts_publish_only_inside_bound_listener_callback():
    source = Path("scripts/polaris/serve_pi05_droid_native_jointvelocity.py").read_text(
        encoding="utf-8"
    )
    callback = source.index("def publish_listener_artifacts(actual_port: int)")
    serving_publication = source.index("publish_immutable_serving_contract(")
    runtime_publication = source.index("publish_immutable_json(", callback)
    server = source.index("server = BoundPortWebsocketPolicyServer(")
    serve_forever = source.index("server.serve_forever()")
    assert callback < serving_publication < runtime_publication < server < serve_forever
    assert "port=args.port" in source
    assert "bound_port_output=args.bound_port_output" in source
    assert "launch_token=args.bound_port_token" in source
