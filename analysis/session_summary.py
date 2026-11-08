"""Sleep-onset summary and inclusion of the Yoga Nidra sessions (Table 2 of the main text).

Included: known audio version, recording covers the second cue plus 5 min, at most 20% bad epochs, hypnogram pinned.
Outputs: results/session_summary.csv
Usage: python -I analysis/session_summary.py <private_root>
"""
import csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script folder from sys.path
import stages
from stages import EPOCH_S, STAGES

CUES_MIN = {"spanish": (31 + 2 / 60, 40 + 12 / 60)}
MAX_BAD = 0.20
AFTER_CUE_S = 600
REPO = Path(__file__).resolve().parents[1]


def asleep_after(hyp, cue_s):
    """Percentage of scored time asleep in [cue, cue + 10 min), or None if the recording ends before."""
    if len(hyp) * EPOCH_S < cue_s + AFTER_CUE_S:
        return None
    t = np.arange(len(hyp)) * EPOCH_S
    overlap = np.clip(np.minimum(t + EPOCH_S, cue_s + AFTER_CUE_S) - np.maximum(t, cue_s), 0, None)
    overlap = overlap * (hyp != stages.UNSCORED)
    if overlap.sum() == 0:
        return None
    return round(100 * (overlap * stages.is_sleep(hyp)).sum() / overlap.sum(), 0)


def first_run(mask, length=1):
    """Index of the first epoch that starts `length` consecutive True epochs, or None."""
    run = np.convolve(mask.astype(int), np.ones(length, int), "valid") == length
    idx = np.flatnonzero(run)
    return int(idx[0]) if idx.size else None


def summarise(private, row):
    s, audio = row["session"], row["audio"]
    out = {"session": s, "audio": audio, "scorer": row.get("scorer", "")}
    hyp = stages.load(private, row)
    if hyp is None:
        return {**out, "included": False, "reason": "no hypnogram pinned in sessions.csv"}
    eeg, minutes = stages.read_eeg(private / "raw" / s, len(hyp))
    bad_frac = stages.bad_epochs(eeg).any(0).mean()
    cues = CUES_MIN.get(audio)

    onset = first_run(stages.is_sleep(hyp), 3)
    n2 = first_run(hyp == stages.N2)
    out.update({
        "duration_min": round(minutes, 1),
        "bad_epochs_pct": round(100 * bad_frac, 1),
        "sleep_onset_min": None if onset is None else onset * EPOCH_S / 60,
        "first_N2_min": None if n2 is None else n2 * EPOCH_S / 60,
        **{f"{st}_min": int((hyp == i).sum()) * EPOCH_S / 60 for i, st in enumerate(STAGES)},
    })
    if cues:
        for k, c in enumerate(cues, 1):
            e = int(c * 60 // EPOCH_S)
            out[f"stage_at_cue{k}"] = stages.name(hyp[e]) if e < len(hyp) else "after end"
            out[f"asleep_10min_after_cue{k}_pct"] = asleep_after(hyp, c * 60)

    reasons = []
    if cues is None:
        reasons.append("audio version unknown")
    elif minutes < cues[1] + 5:
        reasons.append(f"recording shorter than {cues[1] + 5:.1f} min")
    if bad_frac > MAX_BAD:
        reasons.append(f"{100 * bad_frac:.0f}% bad epochs")
    return {**out, "included": not reasons, "reason": "; ".join(reasons)}


def main(private):
    rows = [summarise(private, r) for r in stages.read_sessions(private, "yoga_nidra")]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = REPO / "results"
    out.mkdir(exist_ok=True)
    with open(out / "session_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    inc = [r for r in rows if r["included"]]
    print(f"{len(rows)} sessions, {len(inc)} included")
    for r in rows:
        if not r["included"]:
            print(f"  excluded {r['session']}: {r['reason']}")
    if inc:
        def med(k):
            v = [r[k] for r in inc if r.get(k) is not None]
            return f"{np.median(v):.1f} [{min(v):.1f}-{max(v):.1f}] (n={len(v)})" if v else "n/a"
        for k in ["duration_min", "bad_epochs_pct", "sleep_onset_min", "first_N2_min"] + [f"{s}_min" for s in STAGES]:
            print(f"  {k:<16} median [range] {med(k)}")
        for k in ("stage_at_cue1", "stage_at_cue2"):
            counts = {st: sum(r.get(k) == st for r in inc) for st in STAGES}
            print(f"  {k}: " + ", ".join(f"{st}={n}" for st, n in counts.items() if n))
    print(f"-> {out / 'session_summary.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
