# Tracking Conscious Entry into NREM Sleep (Yoga Nidra) Using a Citizen Neuroscience Framework

Code and computed results for the Methods article by Kavi, Zerr and Patow, submitted to *Frontiers in Psychology* (Research Topic *Methodological Issues in Consciousness Research, Volume IV*).

**Participant data are not shared**, to protect participant privacy. The repository has two parts.

1. **Power calculation and statistics.** Code and results for the MMN power calculation and for the number of LRLR hits needed for significance (Anticipated Results and Figure 6 of the main text). These run without any participant data.
2. **Pilot analysis.** The scripts used on the pilot recordings. They are shared as code only, because they need the private recordings.

## Files

### Power calculation and statistics (runs without participant data)

| File | Purpose |
|---|---|
| `analysis/mmn_power.py` | Power to detect the MMN on a sleep wearable (Anticipated Results). All parameters are set at the top of the script |
| `analysis/hits_fisher.py` | Smallest number of protocol windows with a hit for p ≤ 0.05, one-sided Fisher exact test (Anticipated Results) |
| `analysis/plot_fig6.py` | Figure 6, drawn from the two `*_curves.csv` files |
| `results/` | `mmn_power.csv` and `hits_fisher.csv`, the values reported in the text, and `mmn_power_curves.csv` and `hits_fisher_curves.csv`, the curves of Figure 6 |
| `requirements.txt` | Pinned Python 3.9 packages for these three scripts |

### Pilot analysis (code only)

| File | Purpose |
|---|---|
| `tools/run_ezscore.py` | Sleep stages from ezscore-f (model ez6moe) in NIDRA 0.2.3, the scorer reported in the paper |
| `tools/run_dreamento.py` | Runs Dreamento without its graphical interface. Scores and plots with DreamentoScorer, or with `--ezscore` plots the ezscore-f stages |
| `analysis/session_summary.py` | Table 2 of the main text, and which sessions are analyzed |
| `analysis/pilot_noise.py` | The pilot noise values used in the power calculation |
| `analysis/compare_scorers.py` | Agreement between ezscore-f and DreamentoScorer (Learnings from the Pilot) |
| `tools/make_scorer_figures.py` | Figures 2, 3A, 3B and 4 |
| `analysis/stages.py` | Shared by the pilot scripts. Maps each scorer's stage codes to one coding and loads the hypnograms |
| `analysis/scorer_comparison_plan.md` | Plan for rescoring with ezscore-f, written before scoring, with a dated amendment |
| `env/requirements-nidra.txt` | Pinned Python 3.11 environment for NIDRA |
| `env/requirements-offlinedreamento.txt` | Pinned Python 3.9 environment for Dreamento and the pilot scripts |

## Reproduce the results without participant data

The commands below are for Linux and macOS. On Windows, use `\Scripts\` instead of `/bin/`.

```
python -m venv .venv                          # Python 3.9
.venv/bin/pip install -r requirements.txt
.venv/bin/python -I analysis/mmn_power.py     # -> results/mmn_power.csv, results/mmn_power_curves.csv
.venv/bin/python -I analysis/hits_fisher.py   # -> results/hits_fisher.csv, results/hits_fisher_curves.csv
.venv/bin/python -I analysis/plot_fig6.py     # -> paper/figures/fig6.png
```

Both calculations are deterministic, with no simulation (closed-form power and an exact Fisher test), so the CSV files are reproduced exactly. The published results were computed with the package versions in `requirements.txt`.

The only numbers taken from the participants' recordings are the pilot noise values (`PILOT_NOISE` in `mmn_power.py`). They are reported in the Anticipated Results section and were computed by `analysis/pilot_noise.py`.

## Pilot analysis

### Private data folder

The pilot scripts read a private folder, `<private>`, of pseudonymized ZMax recordings.

```
<private>/
  sessions.csv                         one row per session
  raw/sub-X/ses-YY/                    EEG L.edf, EEG R.edf, ... (HDRecorder EDF export)
  derivatives/sub-X/ses-YY/ezscore/    ezscore-f stages
  derivatives/sub-X/ses-YY/rerun/      DreamentoScorer stages
```

`sessions.csv` has the columns `session, group, audio, scorer, hypnogram, hypnogram_dreamento, figure, note`. For each session, it pins the hypnogram that is reported (ezscore-f) and the DreamentoScorer hypnogram used for comparison.

### Steps

Run the steps in this order.

**1. Score the sleep stages with ezscore-f** (Python 3.11). NIDRA downloads its models only from its graphical interface, so the third line downloads them once.

```
uv venv --python 3.11 .venv-nidra
uv pip install --python .venv-nidra -r env/requirements-nidra.txt
.venv-nidra/bin/python -I -c "import logging; from NIDRA import utils; utils.download_assets('models', logging.getLogger())"
.venv-nidra/bin/python -I tools/run_ezscore.py <private>
.venv-nidra/bin/python -I tools/run_ezscore.py <private> --whole   # robustness check: each recording as one block
```

**2. Score the sleep stages with DreamentoScorer**, for the comparison (Python 3.9). Each hypnogram is written to `<private>/derivatives/sub-X/ses-YY/rerun/hypnogram_<timestamp>.txt` and is then pinned in `sessions.csv`.

```
uv venv --python 3.9 .venv-dreamento
uv pip install --python .venv-dreamento -r env/requirements-offlinedreamento.txt
.venv-dreamento/bin/python tools/run_dreamento.py <private>/raw/sub-X/ses-YY/"EEG L.edf" ... --dreamento <dreamento_dir>
```

**3. Analyze.** Run `session_summary.py` first, because the other two scripts read its output. These outputs stay local and are not shared.

```
.venv-dreamento/bin/python -I analysis/session_summary.py <private>   # Table 2
.venv-dreamento/bin/python -I analysis/pilot_noise.py <private>       # noise values for the power calculation
.venv-dreamento/bin/python -I analysis/compare_scorers.py <private>   # scorer agreement
```

**4. Draw the figures.** Figures 2, 3A, 3B and 4 are written to `<private>/derivatives/figures/`, with a report that checks the drawn stages against the pinned hypnograms.

```
.venv-dreamento/bin/python -I tools/make_scorer_figures.py <private> --dreamento <dreamento_dir>
```

### Notes

- **NIDRA** (https://github.com/paulzerr/nidra) is used unchanged, with one exception. NIDRA 0.2.3 scores a recording in blocks of 100 epochs and fails on recordings shorter than that, so those recordings are scored as one block (Amendment 1 of `analysis/scorer_comparison_plan.md`).
- **Dreamento** (https://github.com/dreamento/dreamento) is used unmodified. The paper used the `OfflineDreamento.py` with SHA-1 `3e699dd9dc7e0fb056c915f99676ea168c53c915`.
- `<dreamento_dir>` is the folder that contains `OfflineDreamento.py` and `DreamentoScorer/`. For `run_dreamento.py`, it can also be set with the `DREAMENTO_DIR` environment variable.
- With `--ezscore`, `run_dreamento.py` replaces only Dreamento's autoscoring call with the ezscore-f stages.
- `run_dreamento.py` handles two cases in which Dreamento crashes. In a session without N2, Dreamento's rule that relabels REM before the first N2 as N1 is applied to all REM epochs. A session without sleep is recorded as all wake.
- Dreamento opens a window even when it draws off-screen, so steps 2 and 4 need a display. On a Linux server, run them under `xvfb-run`.

## License

- Code: Apache License 2.0 (`LICENSE`), copyright 2026 Prakash Kavi, Paul Zerr and Gustavo Patow.
- Computed results in `results/` (the CSV tables): Creative Commons Attribution 4.0 International (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).

To cite, see `CITATION.cff`.
