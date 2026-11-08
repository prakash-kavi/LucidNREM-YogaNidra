"""Figures 2, 3A and 3B drawn by Dreamento with ezscore-f and with DreamentoScorer, and Figure 4 (both scorers).

figures_report.txt checks the drawn stages against the pinned hypnograms and gives the stages around each signal.
Outputs: <private>/derivatives/figures/
Usage: .venv-dreamento\\Scripts\\python -I tools\\make_scorer_figures.py <private_root> --dreamento <dreamento_dir>
"""
import argparse, subprocess, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))   # -I drops the script folder from sys.path
import stages

# figure: centre of the 30-s window (s), LRLR signal (s)
FIGURES = {
    "fig2": (2905, (2902.5, 2907.0)),
    "fig3a": (2942, (2938.0, 2944.0)),
    "fig3b": (2512, (2505.0, 2507.0)),
}
SCORERS = ("dreamento", "nidra")


def describe(hyp, t0, t1):
    """Stage at the signal epoch(s) and at the two epochs on either side."""
    sig = range(int(t0 // stages.EPOCH_S), int(t1 // stages.EPOCH_S) + 1)
    around = range(sig[0] - 2, sig[-1] + 3)
    return (f"signal {'/'.join(stages.name(hyp[e]) for e in sig)}  |  epochs {around[0]}-{around[-1]}: "
            + " ".join(stages.name(hyp[e]) for e in around))


def scorer_comparison_figure(out):
    """Rows: Figures 2, 3A, 3B. Columns: DreamentoScorer, ezscore-f. Whole-recording panels only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    rows = [("A", "fig2", "Figure 2"), ("B", "fig3a", "Figure 3A"), ("C", "fig3b", "Figure 3B")]
    imgs = {(k, s): mpimg.imread(out / f"{k}-{s}-overview.png") for _, k, _ in rows for s in SCORERS}
    h, w = imgs[("fig2", "dreamento")].shape[:2]
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 7.1 * 3 * h / (2 * w) + 0.5), gridspec_kw={"wspace": 0.04, "hspace": 0.12})
    for r, (letter, key, label) in enumerate(rows):
        for c, scorer in enumerate(SCORERS):
            ax = axes[r, c]
            ax.imshow(imgs[(key, scorer)])
            ax.axis("off")
            if r == 0:
                ax.set_title(("DreamentoScorer", "ezscore-f")[c], fontsize=9, fontweight="bold")
        axes[r, 0].text(-0.03, 1.0, f"({letter}) {label}", transform=axes[r, 0].transAxes, fontsize=8,
                        fontweight="bold", ha="left", va="bottom")
    fig.savefig(out / "fig4.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("private", type=Path)
    ap.add_argument("--dreamento", type=Path, required=True, help="folder with OfflineDreamento.py and DreamentoScorer/")
    ap.add_argument("--only", nargs="+", choices=list(FIGURES), help="only these figures (no Figure 4)")
    a = ap.parse_args()
    private = a.private.resolve()
    out = private / "derivatives" / "figures"
    pinned = {r["session"]: r for r in stages.read_sessions(private)}
    session_of = {r["figure"]: s for s, r in pinned.items() if r.get("figure")}
    report = []
    for key in a.only or FIGURES:
        window, sig = FIGURES[key]
        session = session_of[key]
        row = pinned[session]
        edf = private / "raw" / session / "EEG L.edf"
        for scorer in SCORERS:
            cmd = [sys.executable, str(REPO / "tools" / "run_dreamento.py"), str(edf), "--dreamento",
                   str(a.dreamento.resolve()), "--window", str(window), "--out", str(out), "--name", f"{key}-{scorer}"]
            if scorer == "nidra":
                cmd.append("--ezscore")
            print(f"\n### {key} {scorer}", flush=True)
            subprocess.run(cmd, check=True)
        drawn = {s: np.loadtxt(out / f"{key}-{s}_stages.txt", dtype=int) for s in SCORERS}
        dreamento = stages.load(private, {"session": session, "scorer": "dreamento", "hypnogram": row["hypnogram_dreamento"]})
        ezscore = stages.load(private, {"session": session, "scorer": "ezscore", "hypnogram": row["hypnogram"]})
        ezscore = np.where(ezscore == stages.UNSCORED, drawn["nidra"], ezscore)   # artifact epochs are drawn with a stage
        report += [f"{key} ({session})",
                   f"  DreamentoScorer figure stages equal the pinned hypnogram: {np.array_equal(drawn['dreamento'], dreamento)}",
                   f"  ezscore-f figure stages equal the pinned hypnogram (artifact epochs aside): {np.array_equal(drawn['nidra'], ezscore)}",
                   f"  Dreamento: {describe(drawn['dreamento'], *sig)}",
                   f"  ezscore-f: {describe(drawn['nidra'], *sig)}"]
        whole = sorted((private / "derivatives" / session / "ezscore-whole").glob("*_hypnogram.csv"))
        if whole:
            hyp = stages.load(private, {"session": session, "scorer": "ezscore",
                                        "hypnogram": whole[0].relative_to(private).as_posix()})
            report.append(f"  ezscore-f, one block: {describe(hyp, *sig)}")
    if not a.only:
        scorer_comparison_figure(out)
        report.append("fig4.png written")
    (out / "figures_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n" + "\n".join(report) + f"\n-> {out}")


if __name__ == "__main__":
    main()
