import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
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
class ExpandedActionCheckpointWeightLoader(WeightLoader):
    """Loads pi0/pi0.5 checkpoint and expands the action_in_proj / action_out_proj / state_proj
    weights to a larger action_dim. The first `source_action_dim` rows/cols are copied from the
    checkpoint; new rows/cols are zero-filled so the model initially predicts zeros for the new
    dims and exactly the pretrained values for the original dims.
    """

    params_path: str
    source_action_dim: int = 32

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        loaded = self._expand_action_projections(loaded, params)
        return _merge_params(loaded, params, missing_regex=".*lora.*")

    def _expand_action_projections(self, loaded: at.Params, ref: at.Params) -> at.Params:
        flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
        flat_ref = flax.traverse_util.flatten_dict(ref, sep="/")

        # action_in_proj.kernel: [action_dim, expert_width]  -- expand rows (axis 0)
        # state_proj.kernel:     [action_dim, expert_width]  -- expand rows (axis 0)
        # action_out_proj.kernel:[expert_width, action_dim]  -- expand cols (axis 1)
        # action_out_proj.bias:  [action_dim]                -- expand
        targets = [
            ("action_in_proj/kernel", 0),
            ("state_proj/kernel", 0),
            ("action_out_proj/kernel", 1),
            ("action_out_proj/bias", 0),
        ]
        for suffix, axis in targets:
            for k in list(flat_loaded.keys()):
                if not k.endswith(suffix):
                    continue
                if k not in flat_ref:
                    continue
                src = flat_loaded[k]
                dst_shape = flat_ref[k].shape
                if src.shape == dst_shape:
                    continue
                if src.shape[axis] != self.source_action_dim:
                    raise ValueError(
                        f"{k}: expected source dim {self.source_action_dim} on axis {axis}, got {src.shape}"
                    )
                new = np.zeros(dst_shape, dtype=src.dtype)
                slicer = [slice(None)] * len(dst_shape)
                slicer[axis] = slice(0, self.source_action_dim)
                new[tuple(slicer)] = src
                flat_loaded[k] = new
                logger.info(f"Expanded {k}: {src.shape} -> {dst_shape}")

        return flax.traverse_util.unflatten_dict(flat_loaded, sep="/")


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
