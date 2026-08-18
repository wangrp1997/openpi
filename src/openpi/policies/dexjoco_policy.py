"""Data transforms for DexJoCo dual-arm manipulation tasks."""

import dataclasses

import einops
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model as _model

DEXJOCO_ACTION_DIM = 44
DEXJOCO_STATE_DIM = 46

# action44: right(xyz3 + rotvec3 + hand16) + left(same). Official pi0.5 is a 32D
# padded canvas and never trained Allegro fingers, so hand channels stay random.
RIGHT_ARM = slice(0, 6)
RIGHT_HAND = slice(6, 22)
LEFT_ARM = slice(22, 28)
LEFT_HAND = slice(28, 44)
HAND_ACTION_SLICES = (RIGHT_HAND, LEFT_HAND)


def state46_to_action44(state: np.ndarray) -> np.ndarray:
    """Convert quaternion proprioception to the rotvec layout used by DexJoCo actions."""
    state = np.asarray(state)
    if state.shape[-1] != DEXJOCO_STATE_DIM:
        raise ValueError(f"Expected DexJoCo state dimension {DEXJOCO_STATE_DIM}, got {state.shape[-1]}")

    right_arm = state[..., :7]
    left_arm = state[..., 7:14]
    right_hand = state[..., 14:30]
    left_hand = state[..., 30:46]
    right_rotvec = Rotation.from_quat(right_arm[..., 3:7], scalar_first=True).as_rotvec()
    left_rotvec = Rotation.from_quat(left_arm[..., 3:7], scalar_first=True).as_rotvec()
    return np.concatenate(
        [right_arm[..., :3], right_rotvec, right_hand, left_arm[..., :3], left_rotvec, left_hand],
        axis=-1,
    ).astype(np.float32)


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DexJocoInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] == DEXJOCO_STATE_DIM:
            state = state46_to_action44(state)
        elif state.shape[-1] != DEXJOCO_ACTION_DIM:
            raise ValueError(
                f"Expected DexJoCo state dimension {DEXJOCO_STATE_DIM} or {DEXJOCO_ACTION_DIM}, got {state.shape[-1]}"
            )

        inputs = {
            "state": state.astype(np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["base"]),
                "left_wrist_0_rgb": _parse_image(data["wrist_left"]),
                "right_wrist_0_rgb": _parse_image(data["wrist_right"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != DEXJOCO_ACTION_DIM:
                raise ValueError(f"Expected DexJoCo action dimension {DEXJOCO_ACTION_DIM}, got {actions.shape[-1]}")
            inputs["actions"] = actions.astype(np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class DexJocoOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :DEXJOCO_ACTION_DIM]}
