"""Single-trial noise SD of the MMN statistic by stage, from pseudo-trials in the pilot recordings (for the power calculation).

The onsets come from one generator seeded 0, drawn in the session order of results/session_summary.csv.
Outputs: results/pilot_noise.csv
Usage: python -I analysis/pilot_noise.py <private_root>
"""
import csv, sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script folder from sys.path
import stages
from stages import EPOCH_S, FS

STAGES = {"W": stages.W, "N1": stages.N1, "N2": stages.N2, "N3": stages.N3}
TRIALS_PER_STAGE, MIN_EPOCHS = 2000, 4
BASE, WIN = (-0.100, 0.0), (0.125, 0.225)
REJECT_UV, EPOCH_WIN = 100.0, (-0.100, 0.400)
REPO = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def window(onsets, a, b):
    """Sample indices from a to b seconds after each onset."""
    return onsets[:, None] + np.arange(int(a * FS), int(b * FS))[None, :]


def main(private):
    sessions = stages.analysed_sessions()
    pinned = {r["session"]: r for r in stages.read_sessions(private)}
    sos = butter(4, [1, 30], "band", fs=FS, output="sos")
    rows = []
    for s in sessions:
        hyp = stages.load(private, pinned[s])
        eeg, _ = stages.read_eeg(private / "raw" / s, len(hyp))
        ok = ~stages.bad_epochs(eeg).any(0)
        x = sosfiltfilt(sos, eeg.mean(0).ravel())
        for st, k in STAGES.items():
            epochs = np.flatnonzero((hyp == k) & ok)
            if epochs.size == 0:
                continue
            e = rng.choice(epochs, TRIALS_PER_STAGE)
            onsets = e * EPOCH_S * FS + rng.integers(int(0.2 * FS), (EPOCH_S - 1) * FS, TRIALS_PER_STAGE)
            base = x[window(onsets, *BASE)].mean(1, keepdims=True)
            keep = np.abs(x[window(onsets, *EPOCH_WIN)] - base).max(1) <= REJECT_UV
            stat = (x[window(onsets, *WIN)].mean(1) - base[:, 0])[keep]
            rows.append({"session": s, "stage": st, "epochs": int(epochs.size), "sd_uV": round(float(stat.std()), 3),
                         "rejected_pct": round(100 * (1 - keep.mean()), 1)})
            print(f"{s} {st:3} {epochs.size:3d} epochs  single-trial noise SD {stat.std():6.2f} uV, rejected {100 * (1 - keep.mean()):4.1f}%")
    out = REPO / "results" / "pilot_noise.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for st in STAGES:
        v = [r["sd_uV"] for r in rows if r["stage"] == st and r["epochs"] >= MIN_EPOCHS]
        if v:
            print(f"{st:3}: median {np.median(v):.2f} uV [range {min(v):.2f}-{max(v):.2f}] over {len(v)} sessions")
    print(f"-> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
