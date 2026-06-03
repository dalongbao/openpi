"""CLI: compute LoRA Δ-norms across a set of pi0.5 checkpoints.

Example::

    python run_delta_norm.py \
        --base-dir /checkpoints/pi05_base/29999 \
        --finetuned-dirs \
            object_in_bowl:/checkpoints/pi05_egoverse/test/29999 \
            bag_grocery:/checkpoints/pi05_egoverse/bag_grocery/29999 \
            human_oic:/checkpoints/pi05_egoverse/human_oic/29999 \
            mix:/checkpoints/pi05_egoverse/mix/29999 \
        --out /tmp/delta_norms.csv

The output is a wide CSV: one row per
(expert, block_index, submodule, parameter, adapter), one column per
named finetune. Index into the result with pandas for plots.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _parse_named_paths(items):
    """Parse ``name:/path`` entries from the CLI."""
    out: dict[str, str] = {}
    for it in items:
        if ":" not in it:
            raise argparse.ArgumentTypeError(
                f"expected 'name:/path' got {it!r}"
            )
        name, path = it.split(":", 1)
        out[name.strip()] = path.strip()
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Compute LoRA Δ-norms across pi0.5 checkpoints")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Directory of the base (pre-finetune) checkpoint. Should be the orbax "
        "step dir or its 'params' subdir.",
    )
    parser.add_argument(
        "--finetuned-dirs",
        nargs="+",
        required=True,
        help="Whitespace-separated list of name:/path entries. e.g. "
        "object_in_bowl:/path/to/object_in_bowl/29999",
    )
    parser.add_argument(
        "--out",
        default="delta_norms.csv",
        help="Output CSV path (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-AB",
        action="store_true",
        help="If set, keep lora_a and lora_b as separate rows (default: combined).",
    )
    parser.add_argument(
        "--no-expand-scan",
        action="store_true",
        help="If set, do NOT slice along the depth axis. Only one row per LoRA tensor.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="DEBUG | INFO | WARNING (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="[%(asctime)s] %(levelname)s %(name)s :: %(message)s",
    )
    log = logging.getLogger("probing.run_delta_norm")

    # Resolve here so that --help works without pandas installed.
    from probing.delta_norm import delta_norm_table  # type: ignore

    ckpts = _parse_named_paths(args.finetuned_dirs)
    log.info("Base: %s", args.base_dir)
    for n, p in ckpts.items():
        log.info("Finetune '%s': %s", n, p)

    df = delta_norm_table(
        checkpoint_dirs=ckpts,
        base_dir=args.base_dir,
        combine_AB=not args.keep_AB,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote %s rows to %s", len(df), out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
