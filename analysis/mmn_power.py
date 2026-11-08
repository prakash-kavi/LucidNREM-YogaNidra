"""Power calculation for the MMN on a sleep wearable, in closed form (main text, Data Analysis).

Outputs: results/mmn_power.csv, results/mmn_power_curves.csv (curves of Figure 6A and 6B)
Usage: python -I analysis/mmn_power.py
"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[1]

PARAMS = {
    "wake_mmn_uV": 1.86,          # ERP CORE mean amplitude 125-225 ms, FCz, intensity deviants p = .2 (Kappenman et al., 2021)
    "wearable_retention": 0.54,   # 46% MMN effect-size loss, unilateral ear-EEG vs cap (upper bound; Meiser & Bleichner, 2022)
    "peak_ms": 188, "width_ms": 35,                    # ERP CORE peak latency; waveform for the jitter model
    "window_ms": (125, 225), "baseline_ms": (-100, 0),  # ERP CORE measurement window
    "deviant_p": 0.2,             # 2 deviants per 10 tones; upper end of 0.05-0.20 (Duncan et al., 2009), as ERP CORE
    "soa_s": 0.6,                 # within ISI 500-1000 ms (Duncan et al., 2009)
    "calibration_deviants": 400,  # Day 1: 20 min = 2000 tones
    "nightly_deviants": 100,      # nightly 5-min wake calibration, pooled with Day 1
    "jitter_ms": 10,              # main analysis; 30 ms as sensitivity
    "alpha": 0.05,
}
RETENTION = {"W": 1.0, "N2": 0.6}   # preserved MMN, fraction of the wake MMN (assumption); a non-lucid MMN is zero
CURVE_RETENTIONS = (0.4, 0.6, 0.8, 1.0)  # fractions shown in Figure 6B (results/mmn_power_curves.csv)
DEV_PER_MIN = 60 / PARAMS["soa_s"] * PARAMS["deviant_p"]          # 20 deviants/min
VAR_FACTOR = 1 + PARAMS["deviant_p"] / (1 - PARAMS["deviant_p"])  # deviant mean vs mean of all standards
STIM_MIN = (5, 10, 15)  # minutes of successful blocks in stable N2 per night (one or two 5-min episodes); 15 is a sensitivity calculation

# Pilot single-trial noise SD (uV) and number of sessions, from analysis/pilot_noise.py. N3 is a sensitivity value for N2.
PILOT_NOISE = {"W": (8.03, 9), "N2": (8.12, 4), "N3": (12.21, 1)}


def jitter_factor(jitter_ms):
    """Reduction of the window-mean MMN when the waveform is smeared by Gaussian timing jitter."""
    t = np.linspace(-200, 600, 4001)
    def stat(w):
        y = -(PARAMS["width_ms"] / w) * np.exp(-0.5 * ((t - PARAMS["peak_ms"]) / w) ** 2)
        a, b = PARAMS["window_ms"]
        c, d = PARAMS["baseline_ms"]
        return y[(t >= a) & (t < b)].mean() - y[(t >= c) & (t < d)].mean()
    return stat(np.hypot(PARAMS["width_ms"], jitter_ms)) / stat(PARAMS["width_ms"])


def effect(stage, jitter_ms=PARAMS["jitter_ms"], retention=None):
    r = RETENTION[stage] if retention is None else retention
    return PARAMS["wake_mmn_uV"] * PARAMS["wearable_retention"] * r * jitter_factor(jitter_ms)


def se(n_dev, sd):
    return sd * np.sqrt(VAR_FACTOR / n_dev)


def power(n_dev, eff, sd):
    n_dev = np.asarray(n_dev, dtype=float)
    return norm.cdf(eff / se(n_dev, sd) - norm.ppf(1 - PARAMS["alpha"]))


def n_for(eff, sd, target=0.8):
    if eff <= 0:
        return None
    return int(np.ceil((norm.ppf(1 - PARAMS["alpha"]) + norm.ppf(target)) ** 2 * VAR_FACTOR * (sd / eff) ** 2))


def main():
    sd = {st: v[0] for st, v in PILOT_NOISE.items()}
    print("noise SD (uV), sessions:", PILOT_NOISE, f"| {DEV_PER_MIN:.0f} deviants/min")
    rows = []

    # deviants for 80% power with the stage's own noise, and N2 with the N3 noise as sensitivity
    for st, noise in (("W", "W"), ("N2", "N2"), ("N2", "N3")):
        for j in (0, 30):
            e = effect(st, j)
            n80 = n_for(e, sd[noise])
            rows.append({"analysis": "deviants for 80% power", "stage": st, "noise": noise, "jitter_ms": j,
                         "effect_uV": round(e, 3), "noise_sd_uV": sd[noise], "deviants_for_80pct": n80,
                         "minutes_for_80pct": round(n80 / DEV_PER_MIN, 1)})
            print(f"{st:3} ({noise} noise, {j:2d}-ms jitter) effect {e:.2f} uV: 80% power at {n80} deviants "
                  f"({n80 / DEV_PER_MIN:.0f} min)")

    # H1: wake calibration
    e_w = effect("W")
    cal = PARAMS["calibration_deviants"]
    nightly = PARAMS["nightly_deviants"]
    p_cal = float(power(cal, e_w, sd["W"]))
    p_cal1 = float(power(cal + nightly, e_w, sd["W"]))
    se_cal, se_cal20 = se(cal, sd["W"]), se(cal + 20 * nightly, sd["W"])
    p_night_cal = float(power(nightly, e_w, sd["W"]))
    print(f"H1 Day 1 calibration ({cal} deviants): power {p_cal:.2f}, SE {se_cal:.2f} uV; after the first night "
          f"({cal + nightly}): power {p_cal1:.2f}; with 20 nightly calibrations ({cal + 20 * nightly}): SE "
          f"{se_cal20:.2f} uV; one nightly run alone: power {p_night_cal:.2f}")
    rows.append({"analysis": "H1 wake calibration", "stage": "W", "power_day1": round(p_cal, 3),
                 "power_day1_plus_1_night": round(p_cal1, 3), "se_day1_uV": round(se_cal, 3),
                 "se_day1_plus_20_nights_uV": round(se_cal20, 3), "power_single_nightly_run": round(p_night_cal, 3)})

    # per-night power in stable N2
    for m in STIM_MIN:
        n = m * DEV_PER_MIN
        p_n2 = float(power(n, effect("N2"), sd["N2"]))
        p_n3 = float(power(n, effect("N2"), sd["N3"]))
        p_j30 = float(power(n, effect("N2", 30), sd["N2"]))
        print(f"stable N2 {m:2d} min ({n:.0f} deviants): per-night power, preserved MMN {p_n2:.2f} (N3 noise {p_n3:.2f}, "
              f"30-ms jitter {p_j30:.2f}); MMN of zero: positive tests {PARAMS['alpha']:.2f}; SE {se(n, sd['N2']):.2f} uV")
        rows.append({"analysis": "per-night power, stable N2", "stim_min": m, "stim_deviants": int(n),
                     "power_lucid_N2_noise": round(p_n2, 3), "power_lucid_N3_noise": round(p_n3, 3),
                     "power_lucid_30ms_jitter": round(p_j30, 3), "false_positive_nonlucid": PARAMS["alpha"],
                     "se_uV": round(float(se(n, sd["N2"])), 3)})

    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(REPO / "results" / "mmn_power.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("-> results/mmn_power.csv")
    curves(sd)


def curves(sd):
    """Power curves for Figure 6: wake calibration against deviants, and stable N2 against minutes of successful
    blocks per night for several assumed retentions of the wake MMN, with the N2 and the N3 noise."""
    rows = []
    e_w = effect("W")
    for n in range(20, 1001, 20):
        rows.append({"analysis": "wake calibration", "deviants": n, "power": round(float(power(n, e_w, sd["W"])), 4)})
    for r in CURVE_RETENTIONS:
        e = effect("N2", retention=r)
        for noise in ("N2", "N3"):
            for m2 in range(1, 61):  # 0.5 to 30 min in steps of 0.5
                m = m2 / 2
                rows.append({"analysis": "stable N2", "retention": r, "noise": noise, "stim_min": m,
                             "deviants": int(m * DEV_PER_MIN),
                             "power": round(float(power(m * DEV_PER_MIN, e, sd[noise])), 4)})
    fields = ["analysis", "retention", "noise", "stim_min", "deviants", "power"]
    with open(REPO / "results" / "mmn_power_curves.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("-> results/mmn_power_curves.csv")


if __name__ == "__main__":
    main()
