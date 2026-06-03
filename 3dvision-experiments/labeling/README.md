# VLM-Judge Failure-Mode Labeling

Standalone module for labeling Isaac Sim rollout videos with a RoboFAC-style
failure mode using Claude as a vision-language judge.

This module is intentionally decoupled from the openpi training/inference
code: it takes an MP4 + a natural-language instruction (+ optionally a joint
CSV for traceability) and returns a structured `FailureLabel`.

## Taxonomy

Following RoboFAC ([arXiv:2505.12224](https://arxiv.org/abs/2505.12224)) plus
three simulator-specific extensions:

| Label | Meaning |
| --- | --- |
| `success` (outcome) + `none` (primary_failure) | Goal state visibly reached. |
| `wrong_object` | Robot interacted with the wrong object. |
| `wrong_position` | Robot brought the object to the wrong place. |
| `wrong_orientation` | Grasp / placement orientation prevented success. |
| `wrong_timing` | Released / grasped at the wrong moment. |
| `instruction_ignored` | Robot did nothing relevant to the instruction. |
| `physically_impossible` | Sim glitch — gripper clipping, exploded physics. |
| `other` | None of the above. |

## Install

The module needs `anthropic`, `pydantic`, `imageio[ffmpeg]`, and `pillow`.
From the repo root:

```bash
pip install anthropic pydantic 'imageio[ffmpeg]' pillow
```

Set your API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quickstart

### Label a single rollout

```bash
cd 3dvision-experiments
python -m labeling.cli label-one \
    --video isaac-sim/evaluation.mp4 \
    --instruction "put the plate in the crate" \
    --out labels/single.json
```

### Label a batch of rollouts (resumable + parallel)

```bash
python -m labeling.cli label-dir \
    --rollouts-dir runs/ \
    --instructions runs/instructions.csv \
    --out runs/labels.jsonl \
    --parallelism 4
```

`instructions.csv` is a two-column file:

```csv
rollout_id,instruction
ckpt29999_seed0,put the plate in the crate
ckpt29999_seed1,put the plate in the crate
ckpt5000_seed0,put the plate in the crate
```

Each subfolder of `runs/` named `<rollout_id>` must contain an MP4 (the first
`*.mp4` found is used) and optionally a `results.csv` and `metadata.json`. If
`metadata.json` has an `instruction` field, it overrides the CSV.

The runner appends one JSON object per line to the output file. Re-running
with the same output path skips rollouts already present (set `--no-resume`
to disable).

## Cost estimates

Default config: 8 keyframes per video at ~1MP each.

- Per image: ~1500 input tokens (resized to 1024px max edge).
- Per call: ~13k input tokens + ~400 output tokens.
- Claude Opus pricing: ~$15/M input tokens, $75/M output tokens.
- **Estimated cost per video: ~$0.22** (mostly input image tokens).

To label 1000 rollouts: about **$220** with Opus. Switching to a Sonnet-tier
model (set `--model-id`) drops cost roughly 5x.

Cost estimates are returned in each `LabelingResult.cost_usd_estimate`. They
are best-effort — verify against current pricing.

## How the structured output is forced

We use Anthropic tool-use, not free-text JSON parsing. The model is given a
single tool (`submit_failure_label`) and `tool_choice` is set so it MUST be
called. Inputs to the tool are validated against the `FailureLabel` pydantic
schema. If validation fails or the tool isn't called, the labeler raises
`VLMJudgeError` rather than silently emitting garbage. See
`labeling.prompts.LABEL_TOOL` and `labeling.vlm_judge._parse_tool_response`.

## Calibration recipe

Once you have VLM labels, sanity-check them against your own judgment on at
least 20 rollouts:

1. Run the VLM judge on the calibration set:
   ```bash
   python -m labeling.cli label-dir --rollouts-dir cal/ --instructions cal/instructions.csv --out cal/vlm_labels.jsonl
   ```
2. Copy the VLM JSONL to `cal/hand_labels.jsonl` and hand-edit each `label`
   to match your own judgment. Leave the `rollout_id` field intact.
3. Compute Cohen's kappa + confusion matrix:
   ```bash
   python -m labeling.calibrate \
       --hand cal/hand_labels.jsonl \
       --vlm cal/vlm_labels.jsonl \
       --out cal/calibration_report.json
   ```

Interpretation (Landis & Koch):

| Kappa | Agreement |
| --- | --- |
| < 0.2 | Slight — VLM judge is not usable |
| 0.2 – 0.4 | Fair |
| 0.4 – 0.6 | Moderate — usable as a noisy signal |
| 0.6 – 0.8 | Substantial — reasonable for ranking checkpoints |
| > 0.8 | Almost perfect |

If kappa is low, look at the `confusion` matrix to spot systematic errors
(e.g. VLM always says `success` when the robot drops the object near the
target). Then tweak `SYSTEM_PROMPT` in `prompts.py` and re-run.

## Known limitations

- **Single camera view.** The judge only sees the `RecordingCamera` HD feed.
  Some failure modes (e.g. wrong gripper width) are easier to see from the
  wrist camera. Adding multiple camera views would mean ~2x more tokens.
- **Sparse temporal sampling.** 8 keyframes across a 60s rollout means
  ~7.5s between frames. Fast events (transient grasps, mid-air drops) can
  be missed. Crank `--n-keyframes` for important calibration runs.
- **No joint-data grounding.** The `joint_csv_path` is logged but not sent
  to the model — early experiments showed the model gets distracted by
  numeric data and underweights visual evidence. The CSV is still useful
  downstream for quantitative scoring.
- **Cost.** At ~$0.22/video, labeling 10k rollouts is ~$2.2k. Consider
  bootstrapping with a cheaper Sonnet-tier model for screening and using
  Opus only on the ambiguous cases.

## Files

- `schema.py` — `FailureLabel`, `LabelingResult` pydantic models.
- `prompts.py` — system prompt, user template, tool schema, `build_messages`.
- `vlm_judge.py` — `extract_keyframes`, `label_rollout`, `label_directory`.
- `cli.py` — `python -m labeling.cli label-one|label-dir`.
- `calibrate.py` — Cohen's kappa + confusion matrix.
- `tests/` — pytest tests, all SDK calls mocked.
