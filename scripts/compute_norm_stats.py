"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]

    # If the dataset carries `action_mask`, do a mask-aware per-dim accumulation so masked
    # zero-padded dims don't pollute the mean/std. Otherwise fall back to the standard path.
    use_masked = False
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        use_masked = "action_mask" in batch
        break

    if not use_masked:
        stats = {key: normalize.RunningStats() for key in keys}
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
            for key in keys:
                stats[key].update(np.asarray(batch[key]))
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    else:
        acc = {k: {"sum": None, "sumsq": None, "count": None, "min": None, "max": None} for k in keys}
        for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats (masked)"):
            mask = np.asarray(batch["action_mask"]).astype(bool)  # [B, ad]
            for key in keys:
                arr = np.asarray(batch[key]).astype(np.float64)
                if arr.ndim == 3:  # actions: [B, ah, ad] -> broadcast mask along horizon
                    m = np.broadcast_to(mask[:, None, :], arr.shape)
                    arr_flat = arr.reshape(-1, arr.shape[-1])
                    m_flat = m.reshape(-1, m.shape[-1])
                else:  # state: [B, ad]
                    arr_flat = arr
                    m_flat = mask
                w = m_flat.astype(np.float64)
                contrib_sum = (arr_flat * w).sum(axis=0)
                contrib_sumsq = ((arr_flat ** 2) * w).sum(axis=0)
                contrib_count = w.sum(axis=0)
                if acc[key]["sum"] is None:
                    acc[key]["sum"] = contrib_sum
                    acc[key]["sumsq"] = contrib_sumsq
                    acc[key]["count"] = contrib_count
                    valid_vals = np.where(m_flat, arr_flat, np.nan)
                    acc[key]["min"] = np.nanmin(valid_vals, axis=0)
                    acc[key]["max"] = np.nanmax(valid_vals, axis=0)
                else:
                    acc[key]["sum"] += contrib_sum
                    acc[key]["sumsq"] += contrib_sumsq
                    acc[key]["count"] += contrib_count
                    valid_vals = np.where(m_flat, arr_flat, np.nan)
                    acc[key]["min"] = np.fmin(acc[key]["min"], np.nanmin(valid_vals, axis=0))
                    acc[key]["max"] = np.fmax(acc[key]["max"], np.nanmax(valid_vals, axis=0))

        norm_stats = {}
        for key in keys:
            cnt = np.maximum(acc[key]["count"], 1)
            mean = acc[key]["sum"] / cnt
            var = np.maximum(acc[key]["sumsq"] / cnt - mean ** 2, 0)
            std = np.sqrt(var)
            # Zero-out stats for dims that never saw a valid sample (count==0) so normalize is a no-op.
            never_seen = acc[key]["count"] == 0
            mean = np.where(never_seen, 0.0, mean)
            std = np.where(never_seen, 1.0, std)
            mn = np.where(never_seen, 0.0, np.where(np.isnan(acc[key]["min"]), 0.0, acc[key]["min"]))
            mx = np.where(never_seen, 0.0, np.where(np.isnan(acc[key]["max"]), 0.0, acc[key]["max"]))
            norm_stats[key] = normalize.NormStats(mean=mean.astype(np.float32), std=std.astype(np.float32), q01=mn.astype(np.float32), q99=mx.astype(np.float32))

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
