"""Run the Analyze! button of the unmodified OfflineDreamento (HDRecorder mode, autoscoring) without the GUI.

Saves the figures and the DreamentoScorer hypnogram to <root>/derivatives/sub-X/ses-YY/rerun/. --ezscore draws the
ezscore-f stages instead (Artifact probability left out, the five stage probabilities rescaled to sum to 1).
Usage: .venv-dreamento\\Scripts\\python tools\\run_dreamento.py <EEG L.edf> [...] [--dreamento DIR]
       [--window SECONDS] [--ezscore] [--out DIR --name NAME]
"""
import argparse, os, runpy, sys, time, traceback
from pathlib import Path

os.environ.setdefault("OUTDATED_IGNORE", "1")  # no update check by yasa (versions are pinned)
import numpy as np


def out_dir(edf, sub="rerun"):
    parts = list(edf.parent.parts)
    if "raw" not in parts:
        return edf.parent / sub
    parts[parts.index("raw")] = "derivatives"
    return Path(*parts) / sub


def ezscore_stages(edf):
    """ezscore-f stages in Dreamento's coding (W0 N1 1 N2 2 SWS3 REM4) and the five stage probabilities."""
    import pandas as pd
    files = sorted(out_dir(edf, "ezscore").glob("*_hypnodensity.csv"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one ezscore hypnodensity for {edf}, found {len(files)}")
    p = np.loadtxt(files[0], delimiter=",", skiprows=1)[:, 1:]          # Wake, N1, N2, N3, REM, Art
    hyp = np.loadtxt(str(files[0]).replace("_hypnodensity.csv", "_hypnogram.csv"), skiprows=1, dtype=int)
    col = np.select([hyp == 5, hyp == 6], [4, 5], hyp)                    # hypnogram code -> probability column
    if len(hyp) != len(p) or (p[np.arange(len(p)), col] < p.max(1) - 1e-6).any():   # ties within the 6-decimal export
        raise ValueError(f"{files[0].name}: probabilities and hypnogram disagree")
    art = np.flatnonzero(hyp == 6)
    if art.size and art.tolist() != [0]:   # Dreamento's hypnogram line never draws epoch 0
        raise ValueError(f"artifact epochs {art.tolist()} would be drawn as a stage; not supported")
    stage5 = p[:, :5] / p[:, :5].sum(1, keepdims=True)
    y = np.where(hyp == 6, stage5.argmax(1), col)
    proba = pd.DataFrame(stage5, columns=["Wake", "N1", "N2", "SWS", "REM"])
    print(f"ezscore-f stages from {files[0].name} ({len(y)} epochs, artifact epochs {art.tolist()})", flush=True)
    return y, proba


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edf", nargs="+", type=Path)
    ap.add_argument("--dreamento", type=Path, default=Path(os.environ.get("DREAMENTO_DIR", "dreamento")),
                    help="folder with OfflineDreamento.py and DreamentoScorer/ (default: $DREAMENTO_DIR or ./dreamento)")
    ap.add_argument("--window", type=float, metavar="SECONDS",
                    help="centre the 30-s zoom panels on this time, as a click on the spectrogram does in the GUI")
    ap.add_argument("--ezscore", action="store_true", help="draw the ezscore-f stages instead of running DreamentoScorer")
    ap.add_argument("--out", type=Path, help="save the figure as <out>/<name>.png and write no hypnogram")
    ap.add_argument("--name", help="figure file name without .png (with --out, one session only)")
    a = ap.parse_args()
    edfs = [p.resolve() for p in a.edf]
    if a.out is not None:
        if not a.name or len(edfs) != 1:
            ap.error("--out needs --name and exactly one EDF")
        a.out = a.out.resolve()
    elif a.ezscore:
        ap.error("--ezscore needs --out and --name, so that the DreamentoScorer outputs are not overwritten")

    # DreamentoScorer is loaded through a relative path and imports its own 'entropy' package.
    a.dreamento = a.dreamento.resolve()
    os.chdir(a.dreamento)
    sys.path[:0] = [str(a.dreamento), str(a.dreamento / "DreamentoScorer")]
    print(f"Loading Dreamento from {a.dreamento} (numba compiles on import; ~30-60 s with no output) ...", flush=True)
    ns = runpy.run_path(str(a.dreamento / "OfflineDreamento.py"), run_name="offline_dreamento")
    print(f"Loaded. Processing {len(edfs)} session(s).", flush=True)
    import matplotlib.pyplot as plt
    plt.switch_backend("Agg")  # OfflineDreamento selects TkAgg; draw off-screen instead

    OfflineDreamento = ns["OfflineDreamento"]

    # Without a window, Dreamento's pop-ups would block, so they are printed.
    from tkinter import messagebox
    for name in ("showinfo", "showwarning", "showerror"):
        setattr(messagebox, name, lambda title=None, message=None, _n=name, **kw: print(f"[{_n}] {title}:\n{message}", flush=True))

    # post_scoring_N1 (REM before the first N2 -> N1) fails without N2. Same rule then: all REM -> N1.
    post_scoring_N1 = OfflineDreamento.post_scoring_N1

    def post_scoring_N1_no_N2_safe(self, *args, **kwargs):
        if 2 in np.asarray(self.y_pred):
            return post_scoring_N1(self, *args, **kwargs)
        rem = np.asarray(self.y_pred) == 4
        print(f"no N2 in this session: {rem.sum()} REM epochs relabelled N1 (Dreamento's REM-before-first-N2 rule)")
        self.y_pred[rem] = 1
        self.y_pred_post_processed = self.y_pred

    OfflineDreamento.post_scoring_N1 = post_scoring_N1_no_N2_safe

    # retrieve_sleep_statistics indexes the first sleep epoch and crashes when the whole session is wake.
    retrieve_sleep_statistics = OfflineDreamento.retrieve_sleep_statistics

    def retrieve_sleep_statistics_no_sleep_safe(self, hypno, sf_hyp=1 / 30, *args, **kwargs):
        if (np.asarray(hypno) > 0).any():
            return retrieve_sleep_statistics(self, hypno, sf_hyp, *args, **kwargs)
        print("no sleep epoch in this session: all epochs scored Wake", flush=True)
        self.stats = {"TIB": len(hypno) / (60 * sf_hyp), "TST": 0.0, "SOL": float("nan")}

    OfflineDreamento.retrieve_sleep_statistics = retrieve_sleep_statistics_no_sleep_safe

    if a.ezscore:
        def autoscoring_ezscore(self, *args, **kwargs):
            self.y_pred, self.y_pred_proba = ezscore_stages(edf)
            self.sleep_stats = self.retrieve_sleep_statistics(hypno=self.y_pred, sf_hyp=1 / 30,
                                                              sleep_stages=[0, 1, 2, 3, 4])

        OfflineDreamento.autoscoring = autoscoring_ezscore
    module_globals = OfflineDreamento.Apply_button.__globals__  # where the Browse button stores its file list
    root = ns["Tk"]()
    root.withdraw()

    failed = []
    for edf in edfs:
        print(f"\n=== [{edfs.index(edf) + 1}/{len(edfs)}] {edf.parent.parent.name}/{edf.parent.name} ===", flush=True)
        plt.close("all")
        gui = OfflineDreamento(root)
        gui.analysis_signal_options.set(3)  # HDRecorder only
        gui.is_autoscoring.set(1)
        module_globals["hypnodyne_files_list"] = [str(edf)]
        t0 = time.time()
        try:
            gui.Apply_button()
        except Exception as e:
            failed.append(f"{edf.parent.parent.name}/{edf.parent.name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"FAILED: {failed[-1]}", flush=True)
            continue

        if a.window is not None:
            # as a click on the spectrogram (onclick_ZMaxHypnodyneOnly)
            t = np.floor(a.window)
            for num in plt.get_fignums():
                for ax in plt.figure(num).axes:
                    if tuple(ax.get_xlim()) == (0.0, 30.0):
                        ax.set_xlim(t - 15, t + 15)
                    elif "Freq" in ax.get_ylabel():
                        ax.plot([a.window, a.window], [0.3, 40], color="black")
                        ax.set_ylim(0.1, 25)

        if a.out is not None:
            a.out.mkdir(parents=True, exist_ok=True)
            nums = plt.get_fignums()
            for i, num in enumerate(nums, 1):
                path = a.out / (f"{a.name}.png" if len(nums) == 1 else f"{a.name}_{i}.png")
                fig = plt.figure(num)
                fig.savefig(path, dpi=300, bbox_inches="tight")
                print(f"figure saved: {path}")
                if a.window is not None:
                    from matplotlib.transforms import Bbox
                    fig.canvas.draw()
                    r = fig.canvas.get_renderer()
                    t = np.floor(a.window)
                    night = [ax for ax in fig.axes if ax.get_visible() and tuple(ax.get_xlim()) != (t - 15, t + 15)]
                    box = Bbox.union([ax.get_tightbbox(r) for ax in night]).transformed(fig.dpi_scale_trans.inverted())
                    path = a.out / f"{path.stem}-overview.png"
                    fig.savefig(path, dpi=300, bbox_inches=box.padded(0.05))
                    print(f"figure saved: {path} ({len(night)} whole-recording panels)")
            np.savetxt(a.out / f"{a.name}_stages.txt", np.asarray(gui.stage_autoscoring), fmt="%d",
                       header="stages drawn in the figure, 30-s epochs, W=0 N1=1 N2=2 N3=3 REM=4")
            print(f"done in {time.time() - t0:.0f} s")
            continue

        out = out_dir(edf)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for i, num in enumerate(plt.get_fignums(), 1):
            path = out / f"figure{i}_{stamp}.png"
            plt.figure(num).savefig(path, dpi=300, bbox_inches="tight")
            print(f"figure saved: {path}")
        path = out / f"hypnogram_{stamp}.txt"
        np.savetxt(path, np.asarray(gui.stage_autoscoring), fmt="%d",
                   header=f"DreamentoScorer autoscoring, 30-s epochs, W=0 N1=1 N2=2 N3=3 REM=4\nsource: {edf.parent.name}")
        if getattr(gui, "stats", None) is not None:
            (out / f"sleep_metrics_{stamp}.txt").write_text(str(gui.stats), encoding="utf-8")
        stages, counts = np.unique(gui.stage_autoscoring, return_counts=True)
        print(f"hypnogram saved: {path}  ({len(gui.stage_autoscoring)} epochs; "
              + ", ".join(f"{'W N1 N2 N3 REM'.split()[int(s)]}={c}" for s, c in zip(stages, counts)) + ")")
        print(f"done in {time.time() - t0:.0f} s")
    root.destroy()
    print(f"\n{len(edfs) - len(failed)}/{len(edfs)} sessions processed" + "".join(f"\n  FAILED {f}" for f in failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
