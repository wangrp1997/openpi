import jax
import numpy as np
import pytest

from openpi.training import weight_loaders

_HAND_SLICES = (slice(6, 22), slice(28, 44))


def _loaded_32() -> dict:
    return {
        "action_in_proj": {"kernel": np.arange(32 * 4, dtype=np.float32).reshape(32, 4)},
        "action_out_proj": {
            "kernel": np.arange(4 * 32, dtype=np.float32).reshape(4, 32),
            "bias": np.arange(32, dtype=np.float32),
        },
        "backbone": {"kernel": np.ones((2, 2), dtype=np.float32)},
    }


def _ref_44_arrays() -> dict:
    return {
        "action_in_proj": {"kernel": np.full((44, 4), -1.0, dtype=np.float32)},
        "action_out_proj": {
            "kernel": np.full((4, 44), -2.0, dtype=np.float32),
            "bias": np.full(44, -3.0, dtype=np.float32),
        },
        "backbone": {"kernel": np.zeros((2, 2), dtype=np.float32)},
        "adapter": {"lora": np.full((2, 2), 7.0, dtype=np.float32)},
    }


def _ref_44_structs() -> dict:
    return {
        "action_in_proj": {"kernel": jax.ShapeDtypeStruct((44, 4), np.float32)},
        "action_out_proj": {
            "kernel": jax.ShapeDtypeStruct((4, 44), np.float32),
            "bias": jax.ShapeDtypeStruct((44,), np.float32),
        },
        "backbone": {"kernel": jax.ShapeDtypeStruct((2, 2), np.float32)},
        "adapter": {"lora": jax.ShapeDtypeStruct((2, 2), np.float32)},
    }


def test_merge_params_expands_pi0_action_projections():
    loaded = _loaded_32()
    merged = weight_loaders.merge_params_with_action_dim_adaptation(loaded, _ref_44_arrays())
    nnx_in, _ = weight_loaders._nnx_linear_kernel_bias(44, 4, 0)  # noqa: SLF001
    nnx_out, nnx_bias = weight_loaders._nnx_linear_kernel_bias(4, 44, 1)  # noqa: SLF001

    np.testing.assert_array_equal(merged["action_in_proj"]["kernel"][:32], loaded["action_in_proj"]["kernel"])
    np.testing.assert_allclose(merged["action_in_proj"]["kernel"][32:], nnx_in[32:].astype(np.float32))
    np.testing.assert_array_equal(merged["action_out_proj"]["kernel"][:, :32], loaded["action_out_proj"]["kernel"])
    np.testing.assert_allclose(merged["action_out_proj"]["kernel"][:, 32:], nnx_out[:, 32:].astype(np.float32))
    np.testing.assert_array_equal(merged["action_out_proj"]["bias"][:32], loaded["action_out_proj"]["bias"])
    np.testing.assert_allclose(merged["action_out_proj"]["bias"][32:], nnx_bias[32:].astype(np.float32))
    np.testing.assert_array_equal(merged["adapter"]["lora"], 7.0)


def test_merge_params_rejects_unrelated_shape_mismatch():
    with pytest.raises(ValueError, match="Unsupported checkpoint shape mismatch"):
        weight_loaders.merge_params_with_action_dim_adaptation(
            {"backbone": {"kernel": np.zeros((2, 2), dtype=np.float32)}},
            {"backbone": {"kernel": np.zeros((3, 2), dtype=np.float32)}},
        )


def test_merge_params_resets_hand_slices_with_shape_struct():
    loaded = _loaded_32()
    merged = weight_loaders.merge_params_with_action_dim_adaptation(
        loaded, _ref_44_structs(), reset_action_slices=_HAND_SLICES
    )
    nnx_in, _ = weight_loaders._nnx_linear_kernel_bias(44, 4, 0)  # noqa: SLF001

    np.testing.assert_array_equal(merged["action_in_proj"]["kernel"][:6], loaded["action_in_proj"]["kernel"][:6])
    np.testing.assert_allclose(merged["action_in_proj"]["kernel"][6:22], nnx_in[6:22].astype(np.float32))
    np.testing.assert_array_equal(merged["action_in_proj"]["kernel"][22:28], loaded["action_in_proj"]["kernel"][22:28])
    np.testing.assert_allclose(merged["action_in_proj"]["kernel"][28:], nnx_in[28:].astype(np.float32))
    assert not np.allclose(merged["action_in_proj"]["kernel"][6:22], 0.0)
    assert not np.allclose(merged["action_in_proj"]["kernel"][6:22], loaded["action_in_proj"]["kernel"][6:22])


def test_expand_action_projections_is_deterministic():
    loaded = _loaded_32()
    first = weight_loaders.expand_action_projections(
        loaded["action_in_proj"]["kernel"],
        loaded["action_out_proj"]["kernel"],
        loaded["action_out_proj"]["bias"],
        target_action_dim=44,
        reset_action_slices=_HAND_SLICES,
    )
    second = weight_loaders.expand_action_projections(
        loaded["action_in_proj"]["kernel"],
        loaded["action_out_proj"]["kernel"],
        loaded["action_out_proj"]["bias"],
        target_action_dim=44,
        reset_action_slices=_HAND_SLICES,
    )
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a, b)
