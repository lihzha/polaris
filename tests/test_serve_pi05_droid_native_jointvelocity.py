import dataclasses

import pytest

from scripts.polaris.serve_pi05_droid_native_jointvelocity import (
    validate_official_pi05_data_config,
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
