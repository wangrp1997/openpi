import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.nnx as nnx
import flax.traverse_util
import jax
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class ActionDimAdaptedCheckpointWeightLoader(WeightLoader):
    """Loads a 32D pi0.5 checkpoint and expands action projections to the model dim.

    Extra / reset slices are filled with deterministic NNX Linear init, not zeros.
    Prefer converting offline and using CheckpointWeightLoader for training.
    """

    params_path: str
    # After copying overlapping pretrained dims, keep these action-dim slices at the
    # NNX Linear initialization (Allegro fingers were never in the 32D canvas).
    reset_action_slices: tuple[slice, ...] = ()

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return merge_params_with_action_dim_adaptation(
            loaded_params, params, reset_action_slices=self.reset_action_slices
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


_ACTION_IN_KERNEL = "action_in_proj/kernel"
_ACTION_OUT_KERNEL = "action_out_proj/kernel"
_ACTION_OUT_BIAS = "action_out_proj/bias"
_ADAPTABLE_ACTION_KEYS = (_ACTION_IN_KERNEL, _ACTION_OUT_KERNEL, _ACTION_OUT_BIAS)


def _nnx_linear_kernel_bias(in_features: int, out_features: int, rng_seed: int) -> tuple[np.ndarray, np.ndarray]:
    layer = nnx.Linear(in_features, out_features, rngs=nnx.Rngs(jax.random.key(rng_seed)))
    return np.asarray(layer.kernel.value), np.asarray(layer.bias.value)


def _keep_indices(size: int, overlap: int, reset_action_slices: tuple[slice, ...]) -> np.ndarray:
    reset_mask = np.zeros(size, dtype=bool)
    for slc in reset_action_slices:
        reset_mask[slc] = True
    return np.flatnonzero(~reset_mask[:overlap])


def _copy_kept_action_dims(
    loaded: np.ndarray,
    nnx_init: np.ndarray,
    *,
    action_axis: int,
    reset_action_slices: tuple[slice, ...],
) -> np.ndarray:
    adapted = nnx_init.astype(loaded.dtype, copy=True)
    overlap = min(loaded.shape[action_axis], adapted.shape[action_axis])
    keep = _keep_indices(adapted.shape[action_axis], overlap, reset_action_slices)
    if keep.size == 0:
        return adapted
    if action_axis == 0:
        adapted[keep] = loaded[keep]
    else:
        adapted[:, keep] = loaded[:, keep]
    return adapted


def expand_action_projections(
    action_in_kernel: np.ndarray,
    action_out_kernel: np.ndarray,
    action_out_bias: np.ndarray,
    *,
    target_action_dim: int,
    reset_action_slices: tuple[slice, ...] = (),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand 32D pi0.5 action projections with DexJoCo's NNX Linear init."""
    if action_in_kernel.ndim != 2 or action_out_kernel.ndim != 2 or action_out_bias.ndim != 1:
        raise ValueError("Expected 2D action kernels and 1D action_out bias")
    hidden_dim = action_in_kernel.shape[1]
    if action_out_kernel.shape[0] != hidden_dim:
        raise ValueError(
            f"Hidden dim mismatch: action_in_proj {action_in_kernel.shape} vs action_out_proj {action_out_kernel.shape}"
        )

    nnx_in_kernel, _nnx_in_bias = _nnx_linear_kernel_bias(target_action_dim, hidden_dim, 0)
    nnx_out_kernel, nnx_out_bias = _nnx_linear_kernel_bias(hidden_dim, target_action_dim, 1)
    expanded_in = _copy_kept_action_dims(
        action_in_kernel, nnx_in_kernel, action_axis=0, reset_action_slices=reset_action_slices
    )
    expanded_out = _copy_kept_action_dims(
        action_out_kernel, nnx_out_kernel, action_axis=1, reset_action_slices=reset_action_slices
    )
    expanded_bias = _copy_kept_action_dims(
        action_out_bias, nnx_out_bias, action_axis=0, reset_action_slices=reset_action_slices
    )
    logger.info(
        "Expanded action projections %s/%s/%s -> %s/%s/%s (reset_slices=%s)",
        action_in_kernel.shape,
        action_out_kernel.shape,
        action_out_bias.shape,
        expanded_in.shape,
        expanded_out.shape,
        expanded_bias.shape,
        reset_action_slices,
    )
    return expanded_in, expanded_out, expanded_bias


def expand_params_action_dim(
    params: at.Params,
    *,
    target_action_dim: int,
    reset_action_slices: tuple[slice, ...] = (),
) -> at.Params:
    """Return a copy of `params` with action projections expanded in-place."""
    flat = flax.traverse_util.flatten_dict(params, sep="/")
    in_kernel = flat[_ACTION_IN_KERNEL]
    out_kernel = flat[_ACTION_OUT_KERNEL]
    out_bias = flat[_ACTION_OUT_BIAS]
    expanded_in, expanded_out, expanded_bias = expand_action_projections(
        in_kernel,
        out_kernel,
        out_bias,
        target_action_dim=target_action_dim,
        reset_action_slices=reset_action_slices,
    )
    flat[_ACTION_IN_KERNEL] = expanded_in
    flat[_ACTION_OUT_KERNEL] = expanded_out
    flat[_ACTION_OUT_BIAS] = expanded_bias
    return flax.traverse_util.unflatten_dict(flat, sep="/")


def merge_params_with_action_dim_adaptation(
    loaded_params: at.Params,
    params: at.Params,
    *,
    reset_action_slices: tuple[slice, ...] = (),
) -> at.Params:
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    expanded_action = None
    result = {}

    for key, ref_value in flat_ref.items():
        if key not in flat_loaded:
            if re.fullmatch(".*lora.*", key):
                result[key] = ref_value
                continue
            raise KeyError(f"Checkpoint is missing required parameter: {key}")

        loaded_value = flat_loaded[key]
        if loaded_value.shape == ref_value.shape:
            result[key] = loaded_value.astype(ref_value.dtype, copy=False)
            continue
        if key not in _ADAPTABLE_ACTION_KEYS or loaded_value.ndim != ref_value.ndim:
            raise ValueError(
                f"Unsupported checkpoint shape mismatch for {key}: {loaded_value.shape} -> {ref_value.shape}"
            )
        if expanded_action is None:
            expanded_action = {
                _ACTION_IN_KERNEL: None,
                _ACTION_OUT_KERNEL: None,
                _ACTION_OUT_BIAS: None,
            }
            (
                expanded_action[_ACTION_IN_KERNEL],
                expanded_action[_ACTION_OUT_KERNEL],
                expanded_action[_ACTION_OUT_BIAS],
            ) = expand_action_projections(
                flat_loaded[_ACTION_IN_KERNEL],
                flat_loaded[_ACTION_OUT_KERNEL],
                flat_loaded[_ACTION_OUT_BIAS],
                target_action_dim=int(flat_ref[_ACTION_OUT_BIAS].shape[0]),
                reset_action_slices=reset_action_slices,
            )
        result[key] = expanded_action[key].astype(np.dtype(ref_value.dtype), copy=False)

    return flax.traverse_util.unflatten_dict(result, sep="/")
