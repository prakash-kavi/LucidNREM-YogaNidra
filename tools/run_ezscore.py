"""Sleep staging of the pilot sessions with ezscore-f (model ez6moe) through NIDRA 0.2.3.

Outputs: <private>/derivatives/<session>/ezscore/ (ezscore-whole/ with --whole), with a provenance.json.
A scored session is skipped only if its epoch count and the hashes of both EEG files and the model are unchanged.
Usage: .venv-nidra\\Scripts\\python -I tools\\run_ezscore.py <private_root> [--whole] [--force] [--session sub-X/ses-YY ...]
"""
import argparse, hashlib, json, sys, time
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))   # -I drops the script folder from sys.path
import stages

import mne
import numpy as np
import NIDRA
from NIDRA import utils as nidra_utils
from NIDRA.forehead_scorer import ForeheadScorer

MODEL = "ez6moe"

# NIDRA 0.2.3 scores 100-epoch blocks plus a remainder block and fails when there is no full block. Only then
# (or with --whole) is the recording scored as one block, with NIDRA's own call (plan Amendment 1).
_nidra_predict = ForeheadScorer._predict
ONE_BLOCK = {"whole": False, "used": False}


def _predict_one_block(self):
    if self.num_full_seqs > 0 and not ONE_BLOCK["whole"]:
        return _nidra_predict(self)
    d = self.processed_data
    ep = d.reshape(-1, *d.shape[2:])
    ep = ep[~np.isnan(ep.sum(axis=(1, 2)))]
    print(f"Prediction started... ({len(ep)} epochs scored as one block, plan amendment 1)")
    self.raw_predictions = self.session.run(
        None, {self.input_name: np.expand_dims(ep, 0).astype(np.float32)})[0].reshape(-1, 6)
    ONE_BLOCK["used"] = True
    print("Prediction successful.")


ForeheadScorer._predict = _predict_one_block


def digest(path, algo):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_inputs(raw_l, model_file):
    """What a cached result must match: epoch count, hashes of both EEG channels and of the model."""
    r = mne.io.read_raw_edf(raw_l, verbose="ERROR")
    return {"epochs": int(r.n_times / r.info["sfreq"] // stages.EPOCH_S),
            "model_sha256": digest(model_file, "sha256"),
            "input_eegL_sha1": digest(raw_l, "sha1"),
            "input_eegR_sha1": digest(raw_l.with_name("EEG R.edf"), "sha1")}


def validated_hypnogram(private, session, out, n_expected):
    files = sorted(out.glob("*_hypnogram.csv"))
    if len(files) != 1:
        raise ValueError(f"expected one hypnogram in {out}, found {len(files)}")
    loaded = stages.load(private, {"session": session, "scorer": "ezscore",
                                    "hypnogram": files[0].relative_to(private).as_posix()})
    if len(loaded) != n_expected:
        raise ValueError(f"{files[0].name} has {len(loaded)} epochs, expected {n_expected}")
    if not files[0].with_name(files[0].name.replace("_hypnogram", "_hypnodensity")).exists():
        raise ValueError(f"no hypnodensity next to {files[0].name}")
    return loaded


def score_session(private, session, model_file, force, whole):
    raw_l = private / "raw" / session / "EEG L.edf"
    out = private / "derivatives" / session / ("ezscore-whole" if whole else "ezscore")
    prov = out / "provenance.json"
    if not (raw_l.exists() and raw_l.with_name("EEG R.edf").exists()):
        return "FAILED: EEG L.edf or EEG R.edf missing"
    now = current_inputs(raw_l, model_file)
    n_expected = now["epochs"]
    if prov.exists() and not force:
        saved = json.loads(prov.read_text(encoding="utf-8"))
        changed = [k for k in now if saved.get(k) != now[k]]
        if changed:
            return f"FAILED: cached result does not match the current {', '.join(changed)}; rerun with --force"
        validated_hypnogram(private, session, out, n_expected)
        return "skipped (scored before, same inputs and model, validated)"
    out.mkdir(parents=True, exist_ok=True)
    prov.unlink(missing_ok=True)

    ONE_BLOCK.update(whole=whole, used=False)
    result = NIDRA.scorer(type="forehead", input=str(raw_l), output=str(out), model=MODEL,
                          hypnogram=True, hypnodensity=True, plot=True).score()
    if result is None or result[0] is None:
        return "FAILED: NIDRA returned no stages (see its message above)"
    hyp = result[0]
    if len(hyp) != n_expected:
        return f"FAILED: {len(hyp)} epochs, recording has {n_expected}"
    loaded = validated_hypnogram(private, session, out, n_expected)

    prov.write_text(json.dumps({
        "scorer": "ezscore-f via NIDRA", "nidra_version": version("nidra"), "model": MODEL,
        "model_file": model_file.name, "input": f"raw/{session}/EEG L.edf + EEG R.edf",
        **now, "scored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "blocks": ("whole recording as one block (robustness check, plan amendment 1)" if whole
                   else "whole recording as one block (< 100 epochs, plan amendment 1)" if ONE_BLOCK["used"]
                   else "NIDRA unchanged: 100-epoch blocks + remainder block"),
    }, indent=2), encoding="utf-8")
    counts = {stages.name(c): int((loaded == c).sum()) for c in sorted(set(loaded.tolist()))}
    return f"ok, {n_expected} epochs {counts}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("private", type=Path)
    ap.add_argument("--whole", action="store_true", help="score each recording as one block (robustness check)")
    ap.add_argument("--force", action="store_true", help="rescore sessions that are already scored")
    ap.add_argument("--session", nargs="+", help="only these sessions (default: all rows of sessions.csv)")
    a = ap.parse_args()
    private = a.private.resolve()

    model_file = nidra_utils.get_model_path(f"{MODEL}.onnx")
    if not model_file.exists():
        sys.exit(f"Model not found: {model_file}. Download the models first (see README.md).")

    sessions = [r["session"] for r in stages.read_sessions(private)]
    if a.session:
        sessions = [s for s in sessions if s in a.session]
    results = {}
    for s in sessions:
        print(f"\n=== {s}", flush=True)
        try:
            results[s] = score_session(private, s, model_file, a.force, a.whole)
        except Exception as e:
            results[s] = f"FAILED: {type(e).__name__}: {e}"
    print("\n=== Summary")
    for s, msg in results.items():
        print(f"{s:14} {msg}")
    failed = [s for s, m in results.items() if m.startswith("FAILED")]
    print(f"{len(results) - len(failed)} of {len(results)} sessions ok" + (f", failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
