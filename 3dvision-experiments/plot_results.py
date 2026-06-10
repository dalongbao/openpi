"""Results figures from ablation eval JSONs.

Fig-results: data-efficiency curve. x = in-domain robot episodes N in {0,5,15,30,64},
y = reach_object_err (displayed in cm; JSONs store meters). Curves: rid (robot-only)
vs mix (robot+human 50:50); hid = the N=0 point of the mix curve; base = horizontal
dashed reference (untrained pi0.5). 95% bootstrap CI over the 12 held-out episodes;
ordered_success annotated per point (k/n).

Fig-success: companion subgoal plot, same x-axis. y = fraction of episodes with
ordered_success (reached object first, then bowl; solid) and STRICT success
(ordered AND gripper_ok; dash-dot). Strict deflates base's threshold-wander
inflation (base reaches "success" by drifting within 8 cm, but its gripper is
wrong ~half the time). hid has no hand dims -> gripper_ok is NaN -> no strict curve.

Missing condition JSONs are skipped gracefully — rerun as results land.

Usage:
  python plot_results.py [--results-dir DIR] [--metric reach_object_err] [--out-dir DIR]
Outputs Fig-results.{pdf,png} and Fig-success.{pdf,png} (PDF for Overleaf, PNG preview).
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RID = {"rid5": 5, "rid15": 15, "rid30": 30, "rid64": 64}
MIX = {"hid": 0, "mix5": 5, "mix15": 15, "mix30": 30, "mix64": 64}
FAMILIES = [("robot only (rid)", RID, "#1f77b4", "o"),
            ("robot + human (mix)", MIX, "#d62728", "s")]
N_BOOT = 10_000


def load(results_dir: pathlib.Path, cond: str):
    p = results_dir / f"{cond}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def ep_values(payload, metric):
    return np.array([e[metric] for e in payload["per_episode"] if e.get(metric) == e.get(metric)])


def strict_values(payload):
    """ordered_success AND gripper_ok per episode; NaN gripper (6-dim hid) -> empty."""
    vals = [e["ordered_success"] * e["gripper_ok"] for e in payload["per_episode"]
            if e["gripper_ok"] == e["gripper_ok"]]
    return np.array(vals)


def order_bit(e):
    """Threshold-free object-before-bowl bit. New JSONs store it; for older ones recover it
    from ordered_success when both targets were reached at the 8 cm eval threshold. Episodes
    that missed a target at 8 cm fail every stricter threshold regardless -> bit irrelevant."""
    if "subgoal_order_ok" in e:
        return e["subgoal_order_ok"]
    return e["ordered_success"] if (e["reached_object"] and e["reached_bowl"]) else 0.0


def success_at(payload, thresh_m):
    return np.array([float(e["reach_object_err"] < thresh_m and e["reach_bowl_err"] < thresh_m
                           and order_bit(e)) for e in payload["per_episode"]])


def boot_ci(vals, rng):
    idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
    means = vals[idx].mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def style_axis(ax):
    ax.set_xlabel("in-domain robot episodes $N$")
    ax.set_xticks([0, 5, 15, 30, 64])
    ax.spines[["top", "right"]].set_visible(False)


def fig_reach(results_dir, metric, out_dir, rng):
    to_cm = metric.endswith("_err")
    scale = 100.0 if to_cm else 1.0
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    missing = []

    for name, cond_map, color, marker in FAMILIES:
        xs, ys, los, his, succ = [], [], [], [], []
        for cond, n in sorted(cond_map.items(), key=lambda kv: kv[1]):
            payload = load(results_dir, cond)
            if payload is None:
                missing.append(cond)
                continue
            vals = ep_values(payload, metric) * scale
            lo, hi = boot_ci(vals, rng)
            xs.append(n); ys.append(vals.mean()); los.append(lo); his.append(hi)
            s = payload["per_episode"]
            k = sum(int(e["ordered_success"]) for e in s)
            succ.append((n, vals.mean(), f"{k}/{len(s)}"))
        if xs:
            ax.plot(xs, ys, marker=marker, color=color, label=name, lw=2, ms=7, zorder=3)
            ax.fill_between(xs, los, his, color=color, alpha=0.15, zorder=2)
            for n, y, frac in succ:
                ax.annotate(frac, (n, y), textcoords="offset points", xytext=(6, 7),
                            fontsize=8, color=color)

    base = load(results_dir, "base")
    if base is None:
        missing.append("base")
    else:
        vals = ep_values(base, metric) * scale
        ax.axhline(vals.mean(), color="0.35", ls="--", lw=1.5, zorder=1)
        ax.annotate("untrained $\\pi_{0.5}$ (base)", (0.99, vals.mean()),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=9, color="0.35", xytext=(0, 3), textcoords="offset points")

    style_axis(ax)
    unit = "cm" if to_cm else ""
    ax.set_ylabel(f"{metric.replace('_', ' ')}{f'  [{unit}]' if unit else ''}")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="center right")
    ax.set_title("Held-out reach error vs. teleop data budget (12 episodes, 95% bootstrap CI)",
                 fontsize=10)
    fig.tight_layout()
    out = out_dir / "Fig-results"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)
    return out, missing


def fig_success(results_dir, out_dir, rng):
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    missing = []

    for name, cond_map, color, marker in FAMILIES:
        pts = {"ordered": [], "strict": []}
        for cond, n in sorted(cond_map.items(), key=lambda kv: kv[1]):
            payload = load(results_dir, cond)
            if payload is None:
                missing.append(cond)
                continue
            ov = ep_values(payload, "ordered_success")
            pts["ordered"].append((n, ov.mean(), *boot_ci(ov, rng)))
            sv = strict_values(payload)
            if len(sv):  # hid (6-dim) has no gripper -> skip strict there
                pts["strict"].append((n, sv.mean(), *boot_ci(sv, rng)))
        for key, ls, lw, label in [("ordered", "-", 2, f"{name} — ordered subgoals"),
                                   ("strict", "-.", 1.5, f"{name} — + gripper OK")]:
            if pts[key]:
                xs, ys, los, his = map(np.array, zip(*pts[key]))
                ax.plot(xs, ys, ls, marker=marker, color=color, lw=lw, ms=6, label=label,
                        alpha=1.0 if key == "ordered" else 0.7, zorder=3)
                ax.fill_between(xs, los, his, color=color, alpha=0.08, zorder=2)

    base = load(results_dir, "base")
    if base is not None:
        for vals, label, dy in [(ep_values(base, "ordered_success"), "base ordered", 3),
                                (strict_values(base), "base + gripper", -11)]:
            ax.axhline(vals.mean(), color="0.35", ls="--", lw=1.2, zorder=1)
            ax.annotate(label, (0.99, vals.mean()), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color="0.35",
                        xytext=(0, dy), textcoords="offset points")

    style_axis(ax)
    ax.set_ylabel("fraction of held-out episodes")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("Subgoal success: reached object → then bowl (solid), AND gripper correct (dash-dot)",
                 fontsize=10)
    fig.tight_layout()
    out = out_dir / "Fig-success"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)
    return out, missing


def fig_threshold(results_dir, out_dir):
    """Ordered-success rate vs. success threshold (PCK-style). No chosen bar: the whole
    strictness axis is shown; the pre-registered 8 cm companion value is marked. Exact for
    tau <= 8 cm (see order_bit). One curve per available condition."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    taus = np.linspace(0.005, 0.08, 76)  # 0.5 .. 8 cm
    missing = []

    conds = []  # (cond, color, alpha, ls)
    for _, cond_map, color, _ in FAMILIES:
        order = sorted(cond_map.items(), key=lambda kv: kv[1])
        for i, (cond, n) in enumerate(order):
            conds.append((cond, color, 0.35 + 0.65 * (i + 1) / len(order), "-"))
    conds.append(("base", "0.35", 1.0, "--"))

    for cond, color, alpha, ls in conds:
        payload = load(results_dir, cond)
        if payload is None:
            missing.append(cond)
            continue
        ys = [success_at(payload, t).mean() for t in taus]
        ax.plot(taus * 100, ys, ls, color=color, alpha=alpha, lw=2, label=cond)

    ax.axvline(8.0, color="0.6", ls=":", lw=1)
    ax.annotate("pre-registered\ncompanion (8 cm)", (8.0, 0.04), ha="right", fontsize=8,
                color="0.45", xytext=(-4, 0), textcoords="offset points")
    ax.set_xlabel("success threshold  [cm]")
    ax.set_ylabel("ordered subgoal success rate")
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_title("Success vs. threshold strictness (object → bowl ordering enforced)",
                 fontsize=10)
    fig.tight_layout()
    out = out_dir / "Fig-threshold"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200)
    return out, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(pathlib.Path(__file__).parent / "results"))
    ap.add_argument("--metric", default="reach_object_err")
    ap.add_argument("--out-dir", default=str(pathlib.Path(__file__).parent))
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    out_dir = pathlib.Path(args.out_dir)
    rng = np.random.default_rng(0)

    out1, miss1 = fig_reach(results_dir, args.metric, out_dir, rng)
    out2, miss2 = fig_success(results_dir, out_dir, rng)
    out3, miss3 = fig_threshold(results_dir, out_dir)
    missing = sorted(set(miss1) | set(miss2) | set(miss3))
    for out in (out1, out2, out3):
        print(f"wrote {out}.pdf / .png")
    if missing:
        print(f"missing conditions skipped: {', '.join(missing)}")


if __name__ == "__main__":
    main()
