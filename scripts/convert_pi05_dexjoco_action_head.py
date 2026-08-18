"""Convert official 32D pi05_base into a 44D DexJoCo checkpoint.

Allegro finger channels are filled with deterministic NNX Linear init instead of
copying unrelated pretrained dims. Training can then use CheckpointWeightLoader.
"""

from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp
import tyro

import openpi.models.model as _model
from openpi.policies.dexjoco_policy import DEXJOCO_ACTION_DIM
from openpi.policies.dexjoco_policy import HAND_ACTION_SLICES
from openpi.training.weight_loaders import expand_params_action_dim


def main(
    input_path: Path = Path("/home/wangrenpeng/dexjoco/checkpoints/pi05_base"),
    output_path: Path = Path("/mnt/hdd/dexjoco/shared_checkpoints/pi05_base_action_dim_44_hand_fresh"),
) -> None:
    input_params_path = Path(input_path) / "params"
    output_params_path = Path(output_path) / "params"
    params = _model.restore_params(str(input_params_path), restore_type=np.ndarray)
    params = expand_params_action_dim(
        params,
        target_action_dim=DEXJOCO_ACTION_DIM,
        reset_action_slices=HAND_ACTION_SLICES,
    )

    output_params_path.parent.mkdir(parents=True, exist_ok=False)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(
            str(output_params_path),
            args=ocp.args.PyTreeSave(item={"params": params}),  # type: ignore[arg-type]
        )

    in_shape = params["action_in_proj"]["kernel"].shape
    out_shape = params["action_out_proj"]["kernel"].shape
    bias_shape = params["action_out_proj"]["bias"].shape
    print(f"Saved converted params to: {output_params_path}")
    print(f"action_in_proj.kernel: {in_shape}")
    print(f"action_out_proj.kernel: {out_shape}")
    print(f"action_out_proj.bias: {bias_shape}")


if __name__ == "__main__":
    tyro.cli(main)
