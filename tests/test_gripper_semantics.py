import numpy as np

from polaris.gripper_semantics import closed_positive_gripper_mask


def test_closed_positive_threshold_is_exact_inverse_of_training_open_rule():
    closed_positive = np.array([0.49, 0.5, 0.51], dtype=np.float32)

    np.testing.assert_array_equal(
        closed_positive_gripper_mask(closed_positive),
        [False, True, True],
    )
    np.testing.assert_array_equal(
        closed_positive_gripper_mask(np.array([False, True])),
        [False, True],
    )
