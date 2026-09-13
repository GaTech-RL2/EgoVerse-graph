# Results snapshot, 2026-09-13

Every protocol rollout of the UNITE cotrain campaign (sweeps 2 and 3, variants, replicates, CFG sweep), horizon revision 3.

| file | what |
|---|---|
| `summary_single_readings.tsv` | one line per (row, step, embodiment, level, CFG variant): mean peak, SR@0.80/0.95, never-moved, budget, canonical-40 mean; single readings written by `monitor_ct3.py` |
| `replicates_canonical40_vs_pooled80.txt` | 8-replicate means and paired tests for the headline cells, on canonical seeds 0-39 and on the pooled 0-79 (`paired40.py`) |
| `replicates_pooled80_means.txt`, `replicates_pooled80_paired.txt` | the 9-11 tables as first published (80 seeds) (`pool_reps.py`, `paired.py`) |
| `cfg_sweep_2026-09-12.txt`, `cfg_sweep_paired.txt` | CFG 1.0 / 1.5 / 2.0 / 3.0 / 4.0 / 6.0 on the best checkpoints, canonical 40 seeds x 3 replicates, OEC-56 per CFG (`cfg_final_summary.py`, `cfg_paired.py`) |
| `oec_summary_single_readings.txt` | OEC-56 (30 levels x 5 seeds) per row and checkpoint |
| `peaks_export.json.gz` | every episode: `{"row|step|emb|level|variant": {seed: peak}}` for all level-0 and OEC results (`export_peaks.py`) |

Seeds: the protocol's canonical set is level 0 seeds 0-39 plus the OEC-56 bank. The 9-11 headline tables pooled seeds 0-79 (the 40-79 block is a labelled extension); `replicates_canonical40_vs_pooled80.txt` restates them on 0-39. Both agree in sign; the 40-seed error bars are about 1.4x wider.

Raw JSONs and checkpoints live on ICE (`~/scratch/rollouts/CT2-rev3/`, cedar run dirs) as listed in `../unite_cotrain_handoff_2026-09-10.md`.
