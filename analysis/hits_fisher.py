"""Smallest number of protocol windows with a hit for p <= 0.05, one-sided Fisher exact test (LRLR hits, main text).

Outputs: results/hits_fisher.csv, results/hits_fisher_curves.csv (curves of Figure 6C)
Usage: python -I analysis/hits_fisher.py
"""
import csv
from fractions import Fraction
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = (3, 5, 10, 20)
CONTROL = ((20, 0), (20, 1), (50, 0), (50, 1), ("equal", 0), ("equal", 1))   # "equal": H2, one sham per cue
CURVE_PROTOCOL = range(1, 41)                                # Figure 6C: 1 to 40 protocol windows
CURVE_CONTROL = ((20, 0), (20, 1), (50, 0), ("equal", 0))    # Figure 6C series
ALPHA = 0.05


def p_greater(k, n, c, m):
    """One-sided Fisher p for k hits in n protocol windows against c hits in m control windows."""
    total, hits = n + m, k + c          # exact, so p = 0.05 is not lost to rounding
    return Fraction(sum(comb(hits, i) * comb(total - hits, n - i) for i in range(k, min(hits, n) + 1)),
                    comb(total, n))


def min_hits(n, c, m):
    """Smallest k with p <= alpha, or None if no k in 0..n reaches it."""
    return next((k for k in range(n + 1) if p_greater(k, n, c, m) <= ALPHA), None)


def main():
    rows = []
    for n in PROTOCOL:
        row = {"protocol_windows": n}
        for m, c in CONTROL:
            mm = n if m == "equal" else m
            k = min_hits(n, c, mm)
            row[f"control_{m}_hits_{c}"] = k
            row[f"p_control_{m}_hits_{c}"] = None if k is None else round(float(p_greater(k, n, c, mm)), 4)
        rows.append(row)
        print(f"{n} protocol windows:", "  ".join(f"{c}/{m} control: >= {row[f'control_{m}_hits_{c}']}"
                                                  for m, c in CONTROL))
    with open(REPO / "results" / "hits_fisher.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("-> results/hits_fisher.csv")
    curves()


def curves():
    """Smallest number of hits for p <= 0.05 against the number of protocol windows (Figure 6C)."""
    rows = []
    for n in CURVE_PROTOCOL:
        for m, c in CURVE_CONTROL:
            mm = n if m == "equal" else m
            rows.append({"protocol_windows": n, "control_windows": m, "control_hits": c, "min_hits": min_hits(n, c, mm)})
    with open(REPO / "results" / "hits_fisher_curves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("-> results/hits_fisher_curves.csv")


if __name__ == "__main__":
    main()
