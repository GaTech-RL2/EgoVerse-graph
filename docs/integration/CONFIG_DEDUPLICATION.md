# Model configuration consolidation

The complete A/#197, B/#159 and C/#160 consolidation is documented in
[the integration report](CONSOLIDATION_20261001.md). This supersedes the earlier
access-blocked first-layer report. All three pinned sources are available and
preserved; the integrated code passes 1,993 CPU tests and 439 config contexts.
Real-weight/data GPU acceptance and destination merge remain gated.

The [complete disposition table](MODEL_DISPOSITIONS_20261001.md) covers all
29 A, 55 B and 44 C model entries. The 67 unique command paths remain available,
with ten copied definitions replaced by aliases or focused overrides:

| Old command | Shared definition / explicit difference |
| --- | --- |
| `hpt_bc_flow_aria` | `hpt_bc_flow_human`; exact alias |
| `hpt_cotrain_scale_flow_shared_head` | `hpt_cotrain_mecka_flow_shared_head`; exact alias |
| `pi0.5_bc_scale` | `pi0.5_bc_mecka`; exact alias |
| `pi0.5_cotrain_mecka_scale` | `pi0.5_bc_mecka`; exact alias |
| `pi05/pi0.5_ft_abc_eva_6d` | `pi05/pi0.5_bc_abc_eva_6d`; exact alias |
| `abc_arc/hpt_abc_cotrain_180M_qwen_pooled` | Matching 300M definition; flow 384 × 10 |
| `abc_arc/hpt_abc_cotrain_180M_qwen_pooled_arc_D40_M100` | Matching 300M ARC definition; flow 384 × 10 |
| `e1/hpt_flow_wrists_ft_arcdur` | FT definition; action width 16, proprio remains 14 |
| `e1/hpt300_flow_wrists_arcdur` | FT definition; trunk 840 × 19, 10 heads, flow 320 × 6 |
| `e1/dp300pt_wrists_arcdur` | DP-300M definition; pretrained ResNet with original BatchNorm |

E1 observation/codec declarations are shared in `model_contract/e1_wrists`.
Visual graph declarations are shared in `model_contract/visual_bimanual`.
Absolute Hydra defaults with `_here_` preserve the original model namespace.
These are explicit architecture presets, not hidden source-name dispatch.

The source union contains 67 model paths. Taking the longest source definition
at each path gives 14,071 model YAML lines; the destination has 11,784 plus 592
lines of shared declarations. The five exact B aliases alone remove 1,356
duplicated lines. Counts describe the surface, not semantic equivalence.

The first B-only layer compared every non-Hydra field in 254 composed contexts
with a fixed clock. Its historical receipt remains in
`evidence/config-deduplication-20261001.json`; its old access/test blockers are
superseded by [the assembled receipt](evidence/consolidation/validation.json).
Six E1 model consolidations preserve resolved pipeline/optimizer/scheduler
values exactly, apart from declared stage identifiers. All 64 C grid cells
retain networks, schedules and normalization, after making the former
sample-mean reduction default explicit in two recipes.
