# Rescoring the pilot with ezscore-f: plan fixed before scoring

Decided 2026-11-03, before any ezscore-f hypnogram of the pilot recordings existed.

## Decision

ezscore-f (model ez6moe, run through NIDRA) replaces DreamentoScorer as the scorer the paper reports for the pilot, whatever the comparison shows. Agreement with DreamentoScorer is reported. DreamentoScorer is no longer used for staging, and its post-scoring rule that relabels REM before the first N2 as N1 is not applied to ezscore-f.

## Scoring

- Sessions: the 10 analyzed Yoga Nidra sessions and the pretrial session of Figure 2, as listed in `<private>/sessions.csv`.
- Input: the two forehead EEG channels (`EEG L.edf`, `EEG R.edf`) of the HDRecorder export, unchanged.
- Output: `<private>/derivatives/<session>/ezscore/`. The NIDRA version and model are recorded.
- Codes are mapped in `analysis/stages.py` (REM 5 → 4, Artifact 6 → unscored). Unscored epochs are never sleep, so they interrupt the three consecutive sleep epochs that define sleep onset.
- The analysis inclusion criteria are unchanged.

## Comparison with DreamentoScorer (reported, never used to choose the scorer)

1. Agreement: epoch-by-epoch agreement and Cohen's kappa for each session, and a pooled confusion matrix.
2. Stage of the signal epochs: Figure 2 (2900–2910 s), Figure 3A (2938–2944 s), Figure 3B (2505–2507 s).
3. Table 2: sleep onset, first N2, minutes per stage, stage at each cue, sleep in the 10 min after each cue.
4. Pilot noise by stage (`analysis/pilot_noise.py`). If it changes, `analysis/mmn_power.py` is rerun and Supplementary Section S1 is updated.

## Reporting rules

- All pilot numbers and stage labels follow ezscore-f as they come out, including any that weaken a claim.
- Figures 2 and 3 keep the Dreamento layout. Only the hypnogram and stage-probability panels are redrawn from ezscore-f.
- The pilot section is changed only where the scorer forces it: numbers, stage labels, the Methods paragraph on scoring, and the figure panels.

## Amendment 1 (2026-11-03, after scoring the sessions)

Made after the ezscore-f stages of all sessions, including those of the figure sessions, had been seen.

- NIDRA 0.2.3 scores a recording in blocks of 100 epochs and scores the remaining epochs as a separate block. With fewer than 100 epochs the first block is empty and NIDRA fails (Table 2, sessions 7 and 10, with 96 and 91 epochs). For these recordings only, the whole recording is scored as one block, exactly as NIDRA scores a remainder block. Every other recording is scored by NIDRA unchanged, with its 100-epoch blocks (`tools/run_ezscore.py`, recorded in each provenance.json).
- The stages can depend on this blocking. In the session of Figure 3A, the epochs after the first 100 are REM when scored as an 11-epoch remainder block and N1 or N2 when the recording is scored as one block. As a robustness check, the stage of each signal epoch and of its neighboring epochs is also reported with the whole recording scored as one block (`tools/run_ezscore.py --whole`, stages in `tools/make_scorer_figures.py`'s report).
- The developer of NIDRA is asked which use of ezscore-f was validated, and about the failure with fewer than 100 epochs.
