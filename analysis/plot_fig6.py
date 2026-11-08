"""Figure 6: anticipated power of the MMN measurement and the LRLR hit thresholds (Anticipated Results).

Reads results/mmn_power_curves.csv and results/hits_fisher_curves.csv, written by mmn_power.py and hits_fisher.py.
Output: paper/figures/fig6.png (300 dpi)
Usage: python -I analysis/plot_fig6.py
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "figures" / "fig6.png"

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3
RAMP = {0.4: "#86b6ef", 0.6: "#3987e5", 0.8: "#1c5cab", 1.0: "#0d366b"}  # ordinal blue ramp, light to dark
TARGET = 0.8

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False,
})


def read(name):
    with open(REPO / "results" / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def vmark(ax, x, text, y=0.03):
    ax.axvline(x, color=AXIS, linewidth=0.6)
    ax.text(x, y, " " + text, color=MUTED, fontsize=6.5, ha="left", va="bottom", transform=ax.get_xaxis_transform())


def main():
    power = read("mmn_power_curves.csv")
    fisher = read("hits_fisher_curves.csv")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.7), dpi=300)

    # A: wake calibration
    wake = [r for r in power if r["analysis"] == "wake calibration"]
    x = [int(r["deviants"]) for r in wake]
    y = [float(r["power"]) for r in wake]
    a.plot(x, y, color=BLUE, linewidth=1.4)
    a.axhline(TARGET, color=AXIS, linewidth=0.6)
    a.text(1000, TARGET, "80% ", color=MUTED, fontsize=6.5, ha="right", va="bottom")
    vmark(a, 400, "Day 1")
    vmark(a, 500, "+ 1 night", y=0.12)
    style(a, "A  Wake MMN, calibration", "Deviants in the wake calibration", "Power")
    a.set_xlim(0, 1000)
    a.set_ylim(0, 1)

    # B: stable N2 against minutes of successful blocks, one curve per assumed fraction of the wake MMN
    n2 = defaultdict(list)
    for r in power:
        if r["analysis"] == "stable N2":
            n2[(float(r["retention"]), r["noise"])].append((float(r["stim_min"]), float(r["power"])))
    for ret, col in RAMP.items():  # direct labels in ink, above each curve end
        pts = n2[(ret, "N2")]
        b.plot([p[0] for p in pts], [p[1] for p in pts], color=col, linewidth=1.4)
        b.text(29.6, pts[-1][1] + 0.015, f"{ret:.1f}", color=INK2, fontsize=6.5, va="bottom", ha="right")
    pts = n2[(0.6, "N3")]  # nearly coincides with the 0.4 curve, so its label goes below the line
    b.plot([p[0] for p in pts], [p[1] for p in pts], color=RAMP[0.6], linewidth=1.2, linestyle=(0, (3, 2)))
    b.text(29.6, pts[-1][1] - 0.02, "0.6, N3 noise", color=INK2, fontsize=6.5, va="top", ha="right")
    b.axhline(TARGET, color=AXIS, linewidth=0.6)
    vmark(b, 5, "1 episode", y=0.88)
    vmark(b, 10, "2 episodes", y=0.81)
    style(b, "B  MMN in stable N2, per night", "Minutes of successful blocks per night", "Power")
    b.set_xlim(0, 30)
    b.set_ylim(0, 1)
    b.text(0.3, 0.995, "N2 MMN as a fraction of the wake MMN", color=INK2, fontsize=6.5, va="top")

    # C: Fisher thresholds
    series = [((20, 0), BLUE, "-", "20 control nights, 0 hits"),
              ((20, 1), BLUE, (0, (3, 2)), "20 control nights, 1 hit"),
              ((50, 0), ORANGE, "-", "50 control nights, 0 hits"),
              (("equal", 0), AQUA, "-", "1 sham per cue, 0 hits (H2)")]
    for (m, ch), col, ls, label in series:
        pts = [(int(r["protocol_windows"]), int(r["min_hits"])) for r in fisher
               if r["control_windows"] == str(m) and int(r["control_hits"]) == ch and r["min_hits"] not in ("", "None")]
        c.plot([p[0] for p in pts], [p[1] for p in pts], color=col, linewidth=1.3, linestyle=ls,
               drawstyle="steps-mid", label=label)
    style(c, "C  LRLR hits for p ≤ 0.05", "Protocol windows (nights or cues)", "Smallest number of hits")
    c.set_xlim(0, 40)
    c.set_ylim(0, 12)
    c.set_yticks(range(0, 13, 2))
    c.legend(loc="upper left", fontsize=6.5, handlelength=2.0, borderaxespad=0.2, labelspacing=0.3)

    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.16, top=0.89, wspace=0.38)
    fig.savefig(OUT, dpi=300, facecolor="white")
    print("->", OUT.relative_to(REPO))


if __name__ == "__main__":
    main()
