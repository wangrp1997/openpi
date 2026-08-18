import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from openpi.models import model as _model
from openpi.policies import dexjoco_policy


def test_state46_to_action44_layout():
    right_quat = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_quat(scalar_first=True)
    left_quat = Rotation.from_rotvec([-0.4, 0.2, 0.1]).as_quat(scalar_first=True)
    state = np.concatenate(
        [
            [1.0, 2.0, 3.0],
            right_quat,
            [4.0, 5.0, 6.0],
            left_quat,
            np.arange(16),
            np.arange(16, 32),
        ]
    )

    converted = dexjoco_policy.state46_to_action44(state)

    assert converted.shape == (44,)
    np.testing.assert_allclose(converted[:6], [1.0, 2.0, 3.0, 0.1, -0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(converted[6:22], np.arange(16), atol=1e-6)
    np.testing.assert_allclose(converted[22:28], [4.0, 5.0, 6.0, -0.4, 0.2, 0.1], atol=1e-6)
    np.testing.assert_allclose(converted[28:44], np.arange(16, 32), atol=1e-6)


def test_hand_action_slices_are_allegro_channels():
    covered = np.zeros(dexjoco_policy.DEXJOCO_ACTION_DIM, dtype=bool)
    for slc in dexjoco_policy.HAND_ACTION_SLICES:
        covered[slc] = True
    assert covered[6:22].all()
    assert covered[28:44].all()
    assert not covered[:6].any()
    assert not covered[22:28].any()


def test_inputs_reject_wrong_action_dim():
    transform = dexjoco_policy.DexJocoInputs(model_type=_model.ModelType.PI05)
    with pytest.raises(ValueError, match="action dimension"):
        transform(
            {
                "base": np.zeros((3, 8, 8), dtype=np.float32),
                "wrist_left": np.zeros((3, 8, 8), dtype=np.float32),
                "wrist_right": np.zeros((3, 8, 8), dtype=np.float32),
                "state": np.zeros(44, dtype=np.float32),
                "actions": np.zeros((30, 43), dtype=np.float32),
            }
        )
