"""Generate publication-style figures from saved experiment logs and reports."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#f8fafc",
        "axes.edgecolor": "#cbd5e1",
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": True,
        "legend.facecolor": "#ffffff",
        "legend.edgecolor": "#cbd5e1",
    }
)

SLATE = "#475569"
BLUE = "#2563eb"
EMERALD = "#059669"
AMBER = "#d97706"
ROSE = "#e11d48"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_looping_gains() -> None:
    report = json.loads((ROOT / "runs/names_curr_v2/report/report.json").read_text())
    test = report["accuracy"]["test"]
    base = test["baseline"]
    h1 = test["heads"]["1x-plain"]
    h3 = test["heads"]["3x-chain"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=200)

    # Panel A: Depth sweep R=0..3 with 95% Wilson CIs
    depths = [0, 1, 2, 3]
    y_1x = [base["acc"]] + [h1[str(r)]["acc"] for r in (1, 2, 3)]
    ci_1x = [base["ci"]] + [h1[str(r)]["ci"] for r in (1, 2, 3)]
    y_3x = [base["acc"]] + [h3[str(r)]["acc"] for r in (1, 2, 3)]
    ci_3x = [base["ci"]] + [h3[str(r)]["ci"] for r in (1, 2, 3)]

    def err_bars(ys, cis):
        lo = [y - c[0] for y, c in zip(ys, cis)]
        hi = [c[1] - y for y, c in zip(ys, cis)]
        return [lo, hi]

    ax1.errorbar(
        depths,
        [y * 100 for y in y_1x],
        yerr=[
            [e * 100 for e in err_bars(y_1x, ci_1x)[0]],
            [e * 100 for e in err_bars(y_1x, ci_1x)[1]],
        ],
        marker="s",
        markersize=7,
        linewidth=2.3,
        capsize=4,
        color=BLUE,
        label="1x-plain control (trained R=1)",
    )
    ax1.errorbar(
        depths,
        [y * 100 for y in y_3x],
        yerr=[
            [e * 100 for e in err_bars(y_3x, ci_3x)[0]],
            [e * 100 for e in err_bars(y_3x, ci_3x)[1]],
        ],
        marker="o",
        markersize=7.5,
        linewidth=2.5,
        capsize=4,
        color=EMERALD,
        label="3x-chain looped (trained R=3)",
    )
    ax1.axhline(
        base["acc"] * 100,
        color=SLATE,
        linestyle=":",
        linewidth=1.3,
        label="Frozen backbone (R=0: 14.0%)",
    )

    # Annotate key points
    ax1.annotate(
        "48.5%\n(control R=1)",
        xy=(1, 48.5),
        xytext=(1.18, 56.5),
        fontsize=9.5,
        color=BLUE,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.2},
    )
    ax1.annotate(
        "53.3% (+4.8 pp vs 1x)\nMcNemar p = 0.007",
        xy=(3, 53.3),
        xytext=(1.65, 63.0),
        fontsize=9.5,
        color=EMERALD,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ecfdf5",
            "edgecolor": EMERALD,
            "lw": 1.0,
        },
        arrowprops={"arrowstyle": "->", "color": EMERALD, "lw": 1.3},
    )

    ax1.set_title("Overall Test Accuracy vs. Loop Depth R (n = 600)")
    ax1.set_xlabel("Loop Depth R (0 = frozen SmolLM2-360M-Instruct)")
    ax1.set_ylabel("Exact Match Accuracy (%)")
    ax1.set_xticks(depths)
    ax1.set_xticklabels(["R=0\n(base)", "R=1", "R=2", "R=3"])
    ax1.set_ylim(0, 72)
    ax1.legend(loc="lower right", fontsize=9)

    # Panel B: Breakdown by hop count
    hops = ["1-hop\n(n=191)", "2-hop\n(n=203)", "3-hop\n(n=206)"]
    x = np.arange(len(hops))
    w = 0.25
    b_vals = [base["by_hops"][str(k)]["acc"] * 100 for k in (1, 2, 3)]
    c_vals = [h1["1"]["by_hops"][str(k)]["acc"] * 100 for k in (1, 2, 3)]
    l_vals = [h3["3"]["by_hops"][str(k)]["acc"] * 100 for k in (1, 2, 3)]

    r0 = ax2.bar(
        x - w,
        b_vals,
        w,
        color="#94a3b8",
        edgecolor="#475569",
        label="Frozen Backbone (R=0)",
    )
    r1 = ax2.bar(
        x, c_vals, w, color="#60a5fa", edgecolor=BLUE, label="1x-plain Control (R=1)"
    )
    r3 = ax2.bar(
        x + w,
        l_vals,
        w,
        color="#34d399",
        edgecolor=EMERALD,
        label="3x-chain Looped (R=3)",
    )

    for bars in (r0, r1, r3):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.5,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#1e293b",
            )

    ax2.set_title("Accuracy Breakdown by Reasoning Hops (Matched 2.07M Head)")
    ax2.set_ylabel("Exact Match Accuracy (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(hops)
    ax2.set_ylim(0, 112)
    ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Looped Transformer Head on Frozen SmolLM2-360M-Instruct — Multi-Hop Name Chain",
        fontsize=13.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "looping_gains.png", bbox_inches="tight")
    plt.close(fig)


def plot_phase_transition() -> None:
    def val_series(rel: str):
        rows = [r for r in load_jsonl(ROOT / rel) if "val_acc" in r]
        return (
            [r["step"] for r in rows],
            [r["val_acc"] * 100 for r in rows],
            [r["val_loss"] for r in rows],
        )

    s_old, a_old, l_old = val_series("runs/names1/1x-L1-tap8-long/train_log.jsonl")
    s_fix, a_fix, l_fix = val_series("runs/names1/1x-L1-tap8-fixed/train_log.jsonl")
    s_new, a_new, l_new = val_series(
        "runs/names_curr/pretrain-1hop-wd05/train_log.jsonl"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6), dpi=200)

    # Panel A: Validation Accuracy
    ax1.plot(
        s_old,
        a_old,
        marker="x",
        markersize=7,
        linestyle="--",
        linewidth=2.0,
        color=ROSE,
        label="Old: scalar gate + fp16 (NaN freeze @ 4.5k)",
    )
    ax1.plot(
        s_fix,
        a_fix,
        marker="^",
        markersize=6.5,
        linewidth=2.0,
        color=AMBER,
        label="Fixed opt: matrix + QK-norm + norm-out (L1, d=960)",
    )
    ax1.plot(
        s_new,
        a_new,
        marker="o",
        markersize=7,
        linewidth=2.6,
        color=EMERALD,
        label="Fixed + L2, d=256, wd=0.05, WSD (2.07M params)",
    )
    ax1.axhline(
        31.7,
        color=SLATE,
        linestyle=":",
        linewidth=1.3,
        label="Frozen backbone (R=0: 31.7%)",
    )

    ax1.annotate(
        "Induction-head\nphase transition\n(68% → 91% → 96.7%)",
        xy=(3500, 91.0),
        xytext=(4300, 74.0),
        fontsize=9.2,
        color=EMERALD,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#ecfdf5",
            "edgecolor": EMERALD,
            "lw": 1.0,
        },
        arrowprops={"arrowstyle": "->", "color": EMERALD, "lw": 1.3},
    )
    ax1.annotate(
        "fp16 NaN freeze",
        xy=(4500, a_old[2]),
        xytext=(5150, 61.5),
        fontsize=8.8,
        fontweight="bold",
        color=ROSE,
        arrowprops={"arrowstyle": "->", "color": ROSE, "lw": 1.2},
    )

    ax1.set_title("1-Hop Lookup Validation Accuracy (data/names1, n=300)")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation Accuracy (%)")
    ax1.set_ylim(25, 105)
    ax1.set_xlim(0, 7800)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel B: Validation Loss
    ax2.plot(
        s_old,
        l_old,
        marker="x",
        markersize=7,
        linestyle="--",
        linewidth=2.0,
        color=ROSE,
        label="Old: scalar gate + fp16",
    )
    ax2.plot(
        s_fix,
        l_fix,
        marker="^",
        markersize=6.5,
        linewidth=2.0,
        color=AMBER,
        label="Fixed opt (L1, d=960)",
    )
    ax2.plot(
        s_new,
        l_new,
        marker="o",
        markersize=7,
        linewidth=2.6,
        color=EMERALD,
        label="Fixed + L2, d=256, wd=0.05, WSD",
    )

    ax2.annotate(
        "val_loss = 0.0172",
        xy=(5000, 0.01725),
        xytext=(5400, 0.055),
        fontsize=9.5,
        color=EMERALD,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": EMERALD, "lw": 1.2},
    )

    ax2.set_title("1-Hop Validation Cross-Entropy Loss")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Validation Loss (nats / token)")
    ax2.set_ylim(0.0, 0.21)
    ax2.set_xlim(0, 7800)
    ax2.legend(loc="upper right", fontsize=8.8)

    fig.suptitle(
        "Unlocking Content-Based Attention Lookup on Frozen SmolLM2-360M-Instruct (Tap Layer 8)",
        fontsize=13.2,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "phase_transition.png", bbox_inches="tight")
    plt.close(fig)


def plot_per_loop_probe() -> None:
    probe = json.loads((ROOT / "runs/names_curr_v2/probe/probe.json").read_text())
    base_rows = probe["frozen backbone"]
    chain_rows = probe["3x-chain (R=3, chain-supervised)"]

    def get_row(rows, hops, loop):
        for r in rows:
            if r["hops"] == hops and r["loop"] == loop:
                return r
        raise KeyError((hops, loop))

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(11.8, 4.3),
        dpi=200,
        sharey=True,
        gridspec_kw={"width_ratios": [3, 4], "wspace": 0.12},
    )

    # 2-hop heatmap: rows = Loop 0..3, cols = Hop 1, Hop 2, Other
    mat2 = np.zeros((4, 3))
    for i, loop in enumerate([0, 1, 2, 3]):
        r = get_row(base_rows if loop == 0 else chain_rows, 2, loop)
        mat2[i, 0] = r["holds"].get("1", 0.0) * 100
        mat2[i, 1] = r["holds"].get("2", 0.0) * 100
        mat2[i, 2] = r["holds"].get("0", 0.0) * 100

    # 3-hop heatmap: rows = Loop 0..3, cols = Hop 1, Hop 2, Hop 3, Other
    mat3 = np.zeros((4, 4))
    for i, loop in enumerate([0, 1, 2, 3]):
        r = get_row(base_rows if loop == 0 else chain_rows, 3, loop)
        mat3[i, 0] = r["holds"].get("1", 0.0) * 100
        mat3[i, 1] = r["holds"].get("2", 0.0) * 100
        mat3[i, 2] = r["holds"].get("3", 0.0) * 100
        mat3[i, 3] = r["holds"].get("0", 0.0) * 100

    for idx, (ax, mat, cols, title) in enumerate(
        [
            (
                ax1,
                mat2,
                ["Hop 1\n(team)", "Hop 2\n(site / ans)", "Other"],
                "2-Hop Questions (n = 203)",
            ),
            (
                ax2,
                mat3,
                ["Hop 1\n(team)", "Hop 2\n(site)", "Hop 3\n(unit / ans)", "Other"],
                "3-Hop Questions (n = 206)",
            ),
        ]
    ):
        ax.grid(False)
        im = ax.imshow(mat, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=9.8)
        ax.set_yticks(range(4))
        if idx == 0:
            ax.set_yticklabels(
                ["Loop 0 (base)", "Loop 1", "Loop 2", "Loop 3"], fontsize=9.8
            )
        else:
            ax.tick_params(axis="y", left=False)
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                color = "#ffffff" if val >= 55 else "#0f172a"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontweight="bold",
                    fontsize=10,
                )

    cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.025, pad=0.03)
    cbar.set_label("Share of Test Examples (%)")
    fig.suptitle(
        "Per-Loop Latent State Decoding (3x-chain Head Projected Through Frozen lm_head)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.savefig(ASSETS / "per_loop_probe.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_looping_gains()
    plot_phase_transition()
    plot_per_loop_probe()
    print("wrote charts to", ASSETS)
