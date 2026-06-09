"""Prove the mix data path end-to-end on the REAL loaded dataset (no training, no GPU).

For a given mix config it checks, against the actual frames LeRobot loads:
  1. episodes= filter      -> exactly the intended robot + human episode_indices are present
  2. ordering              -> robot episode_indices are all < 64, human all >= 64 (build invariant)
  3. robot/human marker    -> action_mask sums split cleanly into {7 (human), 24 (robot)}
  4. sampler realizes ratio-> draw N indices from the WeightedRandomSampler, measure robot fraction

Run on the cluster (login node is fine; this only reads the cached Arrow dataset):
  cd ~/openpi && HF_LEROBOT_HOME=/cluster/scratch/lichin/lerobot \
    HF_HOME=/cluster/scratch/lichin/hf_cache HF_DATASETS_CACHE=/cluster/scratch/lichin/hf_cache/datasets \
    UV_FROZEN=1 uv run python 3dvision-experiments/verify_mix_sampler.py pi05_ego_mix_oic_n5
"""

import collections
import pathlib
import sys

import numpy as np

from openpi.training import config as C
from openpi.training import data_loader as DL


def unwrap(ds):
    while not hasattr(ds, "hf_dataset") and hasattr(ds, "_dataset"):
        ds = ds._dataset
    return ds


def main(config_name: str, expected_robot: list[int] | None = None):
    cfg = C.get_config(config_name)
    dc = cfg.data.create(pathlib.Path("assets"), cfg.model)
    print(f"config={config_name}")
    print(f"  repo_id={dc.repo_id}  robot_sampling_fraction={dc.robot_sampling_fraction}")
    print(f"  episodes requested: {len(dc.episodes) if dc.episodes else 'ALL'}")

    ds = DL.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model)
    hf = unwrap(ds).hf_dataset
    n = len(hf)
    ei = np.asarray(hf["episode_index"])
    ms = np.asarray(hf["action_mask"], dtype=np.float32).reshape(n, -1).sum(axis=1)

    robot_eps = sorted(int(e) for e in set(ei[ms >= ms.max()].tolist()))
    human_eps = sorted(int(e) for e in set(ei[ms < ms.max()].tolist()))
    print(f"\n[1/4] frames loaded: {n}")
    print(f"[1/4] episode_indices present: {len(set(ei.tolist()))} "
          f"(robot={len(robot_eps)}, human={len(human_eps)})")
    print(f"      robot episode_indices: {robot_eps}")
    print(f"      human episode_index range: {min(human_eps)}..{max(human_eps)}")

    ok_order = all(e < 64 for e in robot_eps) and all(e >= 64 for e in human_eps)
    print(f"[2/4] ordering robot<64<=human: {'PASS' if ok_order else 'FAIL'}")

    sums = dict(zip(*[x.tolist() for x in np.unique(ms, return_counts=True)]))
    n_robot = int((ms >= ms.max()).sum())
    n_human = n - n_robot
    print(f"[3/4] action_mask sums -> counts: {sums}")
    print(f"      robot frames={n_robot}  human frames={n_human}  "
          f"natural robot frac={n_robot / n:.3f}")

    if expected_robot is not None:
        ok_eps = robot_eps == sorted(expected_robot)
        print(f"      robot episode set == {sorted(expected_robot)}: {'PASS' if ok_eps else 'FAIL'}")

    # [4/4] Does the sampler actually realize the target ratio?
    if dc.robot_sampling_fraction is not None:
        sampler = DL._make_mix_sampler(ds, dc.robot_sampling_fraction, dc.robot_episode_threshold, seed=0)
        draws = np.fromiter((i for i in sampler), dtype=np.int64)
        realized = float((ms[draws] >= ms.max()).mean())
        print(f"[4/4] sampler drew {len(draws)} frames -> realized robot fraction={realized:.3f} "
              f"(target {dc.robot_sampling_fraction})")
        # robustness: which distinct episodes got sampled at least once
        drawn_eps = collections.Counter(ei[draws].tolist())
        print(f"      distinct episodes hit: {len(drawn_eps)} "
              f"(robot {sum(1 for e in drawn_eps if e < 64)}, human {sum(1 for e in drawn_eps if e >= 64)})")
    else:
        print("[4/4] robot_sampling_fraction=None -> plain shuffle (no sampler). (Expected for n64 / rid.)")


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "pi05_ego_mix_oic_n5"
    # For n5 the intended robot episodes are these (from _RID_SUBSETS); pass to assert exact match.
    expected = {"pi05_ego_mix_oic_n5": [0, 17, 23, 39, 44]}.get(name)
    main(name, expected)
