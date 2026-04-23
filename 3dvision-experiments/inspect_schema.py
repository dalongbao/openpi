"""Print the observation/action schema for the `pi05_egoverse` policy at serve time.

Run on Euler (GPU not strictly required for schema dump, but checkpoint load + smoke
inference will use one if available):

    cd ~/openpi && uv run python 3dvision-experiments/inspect_schema.py \
        --checkpoint-dir /cluster/work/cvg/data/<username>/checkpoints/pi05_egoverse/<exp_name>/<step>

If --checkpoint-dir is omitted, only the static config is printed (no checkpoint load,
no smoke inference).
"""

import argparse
import pathlib
import pprint

import numpy as np

from openpi.policies import egoverse_policy, policy_config
from openpi.training import config as _config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_egoverse")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Path to a checkpoint dir (…/<exp>/<step>). Omit to skip load + smoke test.",
    )
    parser.add_argument("--default-prompt", default="put the object in the bowl")
    args = parser.parse_args()

    print("=" * 72)
    print(f"CONFIG: {args.config_name}")
    print("=" * 72)
    cfg = _config.get_config(args.config_name)

    print(f"model.__class__        : {cfg.model.__class__.__name__}")
    print(f"model.model_type       : {cfg.model.model_type}")
    print(f"model.pi05             : {getattr(cfg.model, 'pi05', None)}")
    print(f"model.action_dim       : {getattr(cfg.model, 'action_dim', None)}")
    print(f"model.action_horizon   : {getattr(cfg.model, 'action_horizon', None)}")
    print(f"model.max_token_len    : {getattr(cfg.model, 'max_token_len', None)}")
    print(f"model.paligemma_variant: {getattr(cfg.model, 'paligemma_variant', None)}")
    print(f"model.action_expert_variant: {getattr(cfg.model, 'action_expert_variant', None)}")
    print(f"model.discrete_state_input : {getattr(cfg.model, 'discrete_state_input', None)}")

    print()
    print(f"data.__class__         : {cfg.data.__class__.__name__}")
    print(f"data.repo_id           : {cfg.data.repo_id}")
    print(f"data.base_config       : {cfg.data.base_config}")

    print()
    print(f"ACTION_DIM (egoverse_policy) : {egoverse_policy.ACTION_DIM}")

    print()
    print("Canonical example (what the LeRobot dataset row looks like after repack):")
    example = egoverse_policy.make_egoverse_example()
    for k, v in example.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:25s} shape={v.shape}  dtype={v.dtype}")
        else:
            print(f"  {k:25s} {type(v).__name__}={v!r}")

    print()
    print("After EgoverseInputs transform (what the model sees pre-tokenization):")
    transformed = egoverse_policy.EgoverseInputs(model_type=cfg.model.model_type)(dict(example))
    for k, v in transformed.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                if isinstance(vv, np.ndarray):
                    print(f"    {kk:25s} shape={vv.shape}  dtype={vv.dtype}")
                else:
                    print(f"    {kk:25s} {vv!r}")
        elif isinstance(v, np.ndarray):
            print(f"  {k:25s} shape={v.shape}  dtype={v.dtype}")
        else:
            print(f"  {k:25s} {type(v).__name__}={v!r}")

    print()
    print("Client should send (the serve_policy server runs the transforms itself):")
    print("  observation/image : HxWx3 uint8 RGB  (single Aria egocentric cam)")
    print("  observation/state : (24,) float32  [7 arm + 17 hand]")
    print("  prompt            : str            (task language, e.g. 'put the object in the bowl')")
    print("Server returns: actions shape (action_horizon, 24) float32")

    if args.checkpoint_dir is None:
        print()
        print("Skipping checkpoint load + smoke inference (no --checkpoint-dir provided).")
        return

    ckpt = pathlib.Path(args.checkpoint_dir)
    print()
    print("=" * 72)
    print(f"LOADING CHECKPOINT: {ckpt}")
    print("=" * 72)
    policy = policy_config.create_trained_policy(
        cfg, str(ckpt), default_prompt=args.default_prompt
    )

    print("policy.metadata:")
    pprint.pprint(policy.metadata, width=100)

    print()
    print("Norm stats (from policy.metadata if present):")
    norm = policy.metadata.get("norm_stats") if isinstance(policy.metadata, dict) else None
    if norm is None:
        print("  (no norm_stats key in metadata; check assets/pi05_egoverse/egoverse/all)")
    else:
        for k, v in norm.items():
            print(f"  {k}: {getattr(v, 'shape', type(v))}")

    print()
    print("Smoke inference on make_egoverse_example()…")
    out = policy.infer(egoverse_policy.make_egoverse_example())
    print("policy.infer() returned keys:", list(out.keys()))
    for k, v in out.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:25s} shape={v.shape}  dtype={v.dtype}")
        else:
            print(f"  {k:25s} {type(v).__name__}={v!r}")


if __name__ == "__main__":
    main()
