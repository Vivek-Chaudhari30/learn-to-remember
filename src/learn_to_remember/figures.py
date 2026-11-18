"""Static figures for the write-up (M6). Plotting only; the data comes from
analysis.py. Each fig_* returns a matplotlib Figure, or None when none of its
runs exist. A figure with some runs missing is drawn with what exists and says
"not run: ..." under the plot; a missing bar shows the words "not run" in its
slot. Nothing is filled in.

Colors follow the entity across every figure (dataviz reference palette, light
surface, all-pairs validated for these three slots):
  base model = blue, trained on v3 notes = orange, trained on v0 notes = aqua.
The λ sweep training curves use one blue ordinal ramp (λ=0 light -> 0.5 dark),
validated with --ordinal. Aqua is under 3:1 contrast on the surface, so every
series is direct-labeled or named on the axis.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import analysis as an  # noqa: E402
from .report import Runs  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ENTITY = {"base": BLUE, "trained": ORANGE, "trained on v3": ORANGE, "trained on v0": AQUA}
LAMBDA_RAMP = {"0": "#86b6ef", "0.2": "#2a78d6", "0.5": "#104281"}
NOT_RUN = "not run"
# Training-log keys that are fractions: drawn on a fixed 0-1 axis so noise is not magnified.
RATE_KEYS = {"frac_correct", "valid_call_rate", "zero_lookup_rate", "frac_traj_with_repeats",
             "frac_zero_var_groups"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "lines.linewidth": 2, "lines.markersize": 8,
    "errorbar.capsize": 3,
})


def _pct(p):
    return None if p is None else tuple(100 * x for x in p)


def _yerr(p):
    v, lo, hi = p
    return [[v - lo], [hi - v]]


def _footnote(fig, missing: list[str], extra: str = ""):
    parts = [extra] if extra else []
    if missing:
        parts.append(f"{NOT_RUN}: " + "; ".join(missing))
    if parts:
        fig.text(0.01, 0.005, "\n".join(parts), color=MUTED, fontsize=7.5,
                 ha="left", va="bottom", wrap=True)


def _bars(ax, groups: list[str], series: dict[str, list], *, scale=100, fmt="{:.1f}",
          colors=ENTITY, width=0.36):
    """Grouped bars with CI whiskers and a value label on each bar; a missing
    value shows 'not run' in its slot. series: label -> [point or None per group]."""
    n, top = len(series), 0.0
    for j, (lab, pts) in enumerate(series.items()):
        for i, p in enumerate(pts):
            x = i + (j - (n - 1) / 2) * (width + 0.04)
            if p is None:
                ax.text(x, 0, NOT_RUN, rotation=90, ha="center", va="bottom",
                        color=MUTED, fontsize=7)
                continue
            v, lo, hi = (scale * t for t in p)
            ax.bar(x, v, width, color=colors[lab], edgecolor=SURFACE, linewidth=2,
                   label=lab if i == next(k for k, q in enumerate(pts) if q is not None)
                   else None)
            if hi > lo:
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK2, lw=1, capsize=3)
            ax.text(x, hi, fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK2)
            top = max(top, hi)
    ax.set_xticks(range(len(groups)), groups)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, (top or 1) * 1.28)          # headroom so the legend clears the labels
    if scale == 100:                           # percent axis: no ticks above 100
        ax.set_yticks([t for t in range(0, 101, 20) if t <= (top or 1) * 1.28])
    ax.grid(axis="x", visible=False)
    if n >= 2:
        ax.legend(loc="upper left", ncols=n)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def fig_train(curves: dict | None, run: str):
    if not curves or not curves["series"]:
        return None
    ser = curves["series"]
    k = len(ser)
    cols = 3
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.4 * rows + 0.6), squeeze=False)
    steps = curves["step"]
    for ax, (key, s) in zip(axes.flat, ser.items()):
        xs = [x for x, v in zip(steps, s["raw"]) if v is not None]
        ax.plot(xs, [v for v in s["raw"] if v is not None], color=ORANGE, lw=0.8, alpha=0.35)
        ax.plot(steps, s["smooth"], color=ORANGE, lw=2)
        ax.set_title(s["label"], fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
        if key in RATE_KEYS:
            ax.set_ylim(-0.02, 1.02)
    for ax in list(axes.flat)[k:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("step")
    fig.suptitle(f"Training run {run}: steps {steps[0]}–{steps[-1]}", x=0.01, ha="left",
                 fontweight="bold")
    _footnote(fig, [], f"Faint: per step. Solid: trailing mean over {curves['window']} steps.")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return fig


def fig_lambda_curves(train: dict):
    keys = (("tokens_read_mean", "tokens read / trajectory"),
            ("frac_correct", "fraction correct"), ("zero_lookup_rate", "zero-lookup answers"))
    runs = {lam: train.get(f"notes_l{lam}") for lam in an.LAMBDAS}
    present = {lam: c for lam, c in runs.items() if c}
    if len(present) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, (key, label) in zip(axes, keys):
        for lam, c in present.items():
            s = c["series"].get(key)
            if not s:
                continue
            ax.plot(c["step"], s["smooth"], color=LAMBDA_RAMP[lam], label=f"λ={lam}")
            last = next((v for v in reversed(s["smooth"]) if v is not None), None)
            if last is not None:
                ax.annotate(f"λ={lam}", (c["step"][-1], last), xytext=(4, 0),
                            textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        ax.set_title(label, fontsize=9, loc="left")
        ax.set_xlabel("step")
        if key in RATE_KEYS:
            ax.set_ylim(-0.02, 1.02)
    axes[0].legend(loc="best")
    fig.suptitle("Cost penalty λ: training curves (trailing mean)", x=0.01, ha="left",
                 fontweight="bold")
    _footnote(fig, [f"λ={lam}" for lam, c in runs.items() if not c])
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return fig


# --------------------------------------------------------------------------- #
# RQ1 - RQ5
# --------------------------------------------------------------------------- #
def fig_rq1(d: dict):
    if all(p is None for pts in d["series"].values() for p in pts):
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    _bars(ax, d["groups"], d["series"])
    ax.set_ylabel("accuracy % (95% CI)")
    ax.set_title("RQ1: notes × training (company A eval)", loc="left")
    _footnote(fig, d["missing"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


def fig_rq2(d: dict):
    if not d["points"]:
        return None
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    lam_pts = []
    for p in d["points"]:
        if p["accuracy"] is None or p["tokens"] is None:
            continue
        acc, tok = _pct(p["accuracy"]), p["tokens"]
        color = BLUE if p["label"] == "base" else ORANGE
        ax.errorbar(tok[0], acc[0], xerr=_yerr(tok), yerr=_yerr(acc), fmt="o", color=color,
                    ms=8, lw=1, capsize=3, mec=SURFACE, mew=2)
        ax.annotate(p["label"], (tok[0], acc[0]), xytext=(6, 6), textcoords="offset points",
                    fontsize=8, color=INK2)
        if p["label"] != "base":
            lam_pts.append((tok[0], acc[0]))
    if len(lam_pts) >= 2:
        ax.plot(*zip(*lam_pts), color=ORANGE, lw=1, alpha=0.6, zorder=0)
    ax.set_xlabel("tokens read per question (95% CI)")
    ax.set_ylabel("accuracy % (95% CI)")
    ax.set_title("RQ2: accuracy vs reading cost (company A, notes v3)", loc="left")
    ax.grid(axis="both")
    _footnote(fig, d["missing"], "Blue: base model. Orange: trained on v3 notes, one point "
                                 "per λ, joined in λ order.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def fig_rq3(d: dict):
    if not d["series"]:
        return None
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    xs = range(len(d["versions"]))
    k = len(d["series"])
    for j, (lab, pts) in enumerate(d["series"].items()):
        dx = (j - (k - 1) / 2) * 0.07           # dodge so the CI whiskers do not overlap
        got = [(x + dx, _pct(p)) for x, p in zip(xs, pts) if p is not None]
        if not got:
            continue
        x, p = zip(*got)
        ax.errorbar(x, [q[0] for q in p], yerr=[[q[0] - q[1] for q in p], [q[2] - q[0] for q in p]],
                    color=ENTITY[lab], marker="o", ms=8, lw=2, elinewidth=1, capsize=3,
                    mec=SURFACE, mew=2, label=lab)
        ax.annotate(lab, (x[-1], p[-1][0]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)
    ax.set_xticks(list(xs), d["versions"])
    ax.set_xlim(-0.3, len(d["versions"]) - 0.3 + 0.8)
    ax.set_xlabel("notes version at eval time (v3 = current)")
    ax.set_ylabel("accuracy % (95% CI)")
    ax.set_title("RQ3: notes drift (company A)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="best")
    _footnote(fig, d["missing"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


def fig_rq4(d: dict):
    if not d["bars"]:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for i, (lab, p) in enumerate(d["bars"]):
        v, lo, hi = _pct(p)
        ax.bar(i, v, 0.55, color=ENTITY[lab], edgecolor=SURFACE, linewidth=2)
        ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], color=INK2, lw=1, capsize=3)
        ax.text(i, hi, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(d["bars"])), [lab for lab, _ in d["bars"]])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("accuracy % (95% CI)")
    ax.set_title("RQ4: transfer to unseen company B (notes v3)", loc="left")
    _footnote(fig, d["missing"], "Trained runs saw only company A.")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def fig_behavior_qtype(d: dict):
    if not d["accuracy"]:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _bars(axes[0], d["qtypes"], d["accuracy"], width=0.36)
    axes[0].set_ylabel("accuracy % (95% CI)")
    axes[0].set_title("Accuracy by question type", loc="left")
    _bars(axes[1], d["qtypes"], d["used_files"], width=0.36)
    axes[1].set_ylabel("% of questions using file tools")
    axes[1].set_title("File-tool use by question type", loc="left")
    fig.suptitle("RQ5: behavior, base vs trained λ=0.2 (company A, notes v3)", x=0.01,
                 ha="left", fontweight="bold")
    _footnote(fig, d["missing"], "File use is wanted on gap and stale questions; on lookup "
                                 "questions it is a false alarm.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    return fig


def fig_lookups(d: dict):
    if not d["series"]:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    _bars(ax, d["bins"], {k: [(v, v, v) for v in vs] for k, vs in d["series"].items()},
          fmt="{:.0f}")
    ax.set_xlabel("lookups before answering")
    ax.set_ylabel("% of questions")
    ax.set_title("RQ5: lookups per question (company A, notes v3)", loc="left")
    _footnote(fig, d["missing"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return fig


# --------------------------------------------------------------------------- #
# Everything
# --------------------------------------------------------------------------- #
def make_all(spec: dict, root: str, out_dir: str, *, dpi: int = 150) -> dict:
    """Write every figure that has data to out_dir/<name>.png and an INDEX.md
    listing what was drawn and what was not run. Returns {'written', 'not_run'}."""
    runs = Runs(spec, root)
    train = an.train_runs(spec, root)
    figs = {f"train_{run}": (lambda c=c, run=run: fig_train(c, run))
            for run, c in train.items()}
    figs.update({
        "train_lambda_sweep": lambda: fig_lambda_curves(train),
        "rq1_notes_x_training": lambda: fig_rq1(an.rq1_data(runs)),
        "rq2_cost_tradeoff": lambda: fig_rq2(an.rq2_data(runs)),
        "rq3_drift": lambda: fig_rq3(an.rq3_data(runs)),
        "rq4_transfer": lambda: fig_rq4(an.rq4_data(runs)),
        "rq5_behavior_by_qtype": lambda: fig_behavior_qtype(an.behavior_by_qtype(runs)),
        "rq5_lookups": lambda: fig_lookups(an.lookup_histogram(runs)),
    })
    os.makedirs(out_dir, exist_ok=True)
    written, not_run = {}, []
    for name, make in figs.items():
        fig = make()
        if fig is None:
            not_run.append(name)
            continue
        path = os.path.join(out_dir, f"{name}.png")
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written[name] = path
    lines = ["# Figures (generated by scripts/make_figures.py)", ""]
    lines += [f"- [{n}]({n}.png)" for n in written] or ["No figure has data yet."]
    if not_run:
        lines += ["", f"{NOT_RUN.capitalize()} (no result files yet): "
                  + ", ".join(f"`{n}`" for n in not_run)]
    with open(os.path.join(out_dir, "INDEX.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"written": written, "not_run": not_run}
