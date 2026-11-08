"""Shared by the pilot scripts: stage coding (UNSCORED = artifact), hypnogram loading and the bad-epoch rule."""
import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

W, N1, N2, N3, REM, UNSCORED = 0, 1, 2, 3, 4, -1
STAGES = ["W", "N1", "N2", "N3", "REM"]
SLEEP = (N1, N2, N3, REM)
EPOCH_S = 30
FS = 256
FLAT_UV, CLIP_UV = 0.5, 1500

SCORER_CODES = {
    "dreamento": {0: W, 1: N1, 2: N2, 3: N3, 4: REM},
    "ezscore": {0: W, 1: N1, 2: N2, 3: N3, 5: REM, 6: UNSCORED},   # NIDRA 0.2.3: 4 is unused
}


def name(code):
    return "unscored" if code == UNSCORED else STAGES[code]


def is_sleep(hyp):
    return np.isin(hyp, SLEEP)


def bad_epochs(x):
    """Flat or clipped epochs of EEG in uV, shaped (..., epochs, samples)."""
    return (x.std(-1) < FLAT_UV) | (np.ptp(x, -1) > CLIP_UV)


def read_eeg(raw_dir, n_epochs):
    """EEG L and EEG R in uV, shaped (2, n_epochs, EPOCH_S * FS), and the shorter channel's length in min.
    The complete 30-s epochs of each channel must equal the hypnogram's n_epochs."""
    import mne
    out, minutes = [], []
    for ch in ("EEG L", "EEG R"):
        raw = mne.io.read_raw_edf(Path(raw_dir) / f"{ch}.edf", verbose="ERROR")
        if raw.info["sfreq"] != FS:
            raise ValueError(f"{raw_dir}/{ch}.edf: sampling rate {raw.info['sfreq']} Hz, expected {FS}")
        n = raw.n_times // (EPOCH_S * FS)
        if n != n_epochs:
            raise ValueError(f"{raw_dir}/{ch}.edf: {n} complete epochs, hypnogram has {n_epochs}")
        out.append(raw.get_data()[0, : n_epochs * EPOCH_S * FS].reshape(n_epochs, EPOCH_S * FS) * 1e6)
        minutes.append(raw.n_times / FS / 60)
    return np.stack(out), min(minutes)


def _read_codes(path):
    """One integer per line; '#' lines and a header are skipped, and several fields on a line are refused."""
    codes = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            codes.append(int(line))
        except ValueError:
            if codes or len(line.replace(",", " ").split()) > 1:
                raise ValueError(f"{path}: expected one integer stage code per line, got {line!r}")
    return np.array(codes, dtype=int)


def load(private, row):
    """Hypnogram of a sessions.csv row in the internal coding, or None if none is pinned."""
    if not row.get("hypnogram"):
        return None
    scorer = row["scorer"]
    if scorer not in SCORER_CODES:
        raise ValueError(f"{row['session']}: unknown scorer {scorer!r}")
    path = Path(private) / row["hypnogram"]
    raw = _read_codes(path)
    mapping = SCORER_CODES[scorer]
    unknown = sorted(set(raw.tolist()) - set(mapping))
    if unknown:
        raise ValueError(f"{path}: codes {unknown} are not defined for scorer {scorer!r}")
    return np.array([mapping[c] for c in raw], dtype=int)


def read_sessions(private, group=None):
    with open(Path(private) / "sessions.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if group is None or r["group"] == group]


def analysed_sessions():
    """Sessions included by session_summary.py, in the order of results/session_summary.csv."""
    with open(REPO / "results" / "session_summary.csv", newline="", encoding="utf-8") as f:
        return [r["session"] for r in csv.DictReader(f) if r["included"] == "True"]
