"""Epoch-by-epoch agreement of ezscore-f and DreamentoScorer, artifact epochs left out (Learnings from the Pilot).

Outputs: <private>/derivatives/comparison/scorer_agreement.csv, confusion_matrix.csv
Usage: python -I analysis/compare_scorers.py <private_root>
"""
import csv, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I drops the script folder from sys.path
import stages


def confusion(a, b):
    m = np.zeros((len(stages.STAGES),) * 2)
    np.add.at(m, (a, b), 1)
    return m


def agreement(m):
    """Proportion of agreement and Cohen's kappa of a confusion matrix."""
    n = m.sum()
    po = np.trace(m) / n
    pe = (m.sum(0) * m.sum(1)).sum() / n ** 2
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main(private):
    rows = {r["session"]: r for r in stages.read_sessions(private)}
    sessions = stages.analysed_sessions() + [s for s, r in rows.items() if r["group"] == "pretrial"]
    out, pooled = [], np.zeros((len(stages.STAGES),) * 2)
    for s in sessions:
        r = rows[s]
        ez = stages.load(private, r)
        dr = stages.load(private, {"session": s, "scorer": "dreamento", "hypnogram": r["hypnogram_dreamento"]})
        if len(ez) != len(dr):
            raise ValueError(f"{s}: {len(ez)} vs {len(dr)} epochs")
        keep = ez != stages.UNSCORED
        m = confusion(ez[keep], dr[keep])
        pooled += m
        po, k = agreement(m)
        out.append({"session": s, "group": r["group"], "epochs": len(ez), "artifact_epochs": int((~keep).sum()),
                    "agreement_pct": round(100 * po, 1), "kappa": round(k, 3)})

    dest = Path(private) / "derivatives" / "comparison"
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "scorer_agreement.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    with open(dest / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ezscore-f \\ DreamentoScorer"] + stages.STAGES)
        for st, counts in zip(stages.STAGES, pooled):
            w.writerow([st] + [int(v) for v in counts])

    for o in out:
        print(f"{o['session']:14} {o['group']:10} {o['epochs']:4d} epochs  agreement {o['agreement_pct']:5.1f}%  kappa {o['kappa']:6.3f}")
    yn = [o for o in out if o["group"] == "yoga_nidra"]
    ks, ag = [o["kappa"] for o in yn], [o["agreement_pct"] for o in yn]
    print(f"\nYoga Nidra sessions (n={len(yn)}): agreement median {np.median(ag):.1f}% [{min(ag):.1f}-{max(ag):.1f}], "
          f"kappa median {np.median(ks):.2f} [{min(ks):.2f}-{max(ks):.2f}]")
    po, k = agreement(pooled)
    print(f"pooled (all {len(out)} sessions): agreement {100 * po:.1f}%, kappa {k:.2f}")
    print("pooled confusion matrix, rows ezscore-f, columns DreamentoScorer:")
    print("       " + " ".join(f"{s:>5}" for s in stages.STAGES))
    for st, counts in zip(stages.STAGES, pooled):
        print(f"  {st:4} " + " ".join(f"{int(v):5d}" for v in counts))
    print(f"-> {dest}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
