# Source PR provenance

Every head below is an ancestor of integration code commit
`96df46f1427cd08e0dd64ac534af133ee37b193c`. Source PRs and branches remain intact.
The [machine ledger](evidence/consolidation/source-pr-ledger.json) records full SHAs,
base/head relationships, destinations, removed historical paths and check URLs.
Paths removed by earlier A ancestors retain their recorded migrations; they are
not silently reintroduced over newer visual recipes.

| PR | Capability | Destination / disposition |
| --- | --- | --- |
| [#161](https://github.com/GaTech-RL2/EgoVerse-graph/pull/161) | feat(arc): preserve native source windows and timed reconstruction | embodiment helpers and lab codec; native windows retained |
| [#162](https://github.com/GaTech-RL2/EgoVerse-graph/pull/162) | feat(hpt): add separate clock heads and shared-stem mixtures | HPT clock heads and mixtures; newer canonical visual callers |
| [#163](https://github.com/GaTech-RL2/EgoVerse-graph/pull/163) | feat(eval): unify execution-capped open-loop metrics and videos | OpenLoopSimEval and shared video lifecycle |
| [#164](https://github.com/GaTech-RL2/EgoVerse-graph/pull/164) | feat(eval): add offline checkpoint validation sweeps | scripts/eval/validate_checkpoint_sweep.py |
| [#165](https://github.com/GaTech-RL2/EgoVerse-graph/pull/165) | feat(bc): add RL2 stationery and ABC towels campaign recipes | robot_bc visual and retained legacy ABC experiments |
| [#166](https://github.com/GaTech-RL2/EgoVerse-graph/pull/166) | feat(arc): add separate bimanual translation and rotation clocks | hybrid codec with later per-arm corrections |
| [#167](https://github.com/GaTech-RL2/EgoVerse-graph/pull/167) | feat(bc): add human-bimanual MECKA baseline and hybrid recipes | human_bc visual recipes; native human ARC sampling |
| [#168](https://github.com/GaTech-RL2/EgoVerse-graph/pull/168) | feat(bc): add visual-only ABC multitask and 30Hz distance calibration | ABC visual multitask recipes and calibration |
| [#169](https://github.com/GaTech-RL2/EgoVerse-graph/pull/169) | feat(train): preserve W&B identity on restart without conflating offline eval | trainHydra W&B identity and resume tests |
| [#170](https://github.com/GaTech-RL2/EgoVerse-graph/pull/170) | Add distance-budgeted episode DTW with bidirectional timing alignment | distance_budget_dtw.py; later M28 corrections retained |
| [#171](https://github.com/GaTech-RL2/EgoVerse-graph/pull/171) | Fix visual BC targets and add language-free robot and human recipes | canonical visual recipe targets and independent BC tests |
| [#172](https://github.com/GaTech-RL2/EgoVerse-graph/pull/172) | fix(eval): canonical open-loop video and visual ABC recipes | canonical robot_bc/human_bc paths and full-frame videos |
| [#173](https://github.com/GaTech-RL2/EgoVerse-graph/pull/173) | feat(arc): unify race, multistream, and joint-distance hybrid modes | race/multistream/joint_distance contract |
| [#177](https://github.com/GaTech-RL2/EgoVerse-graph/pull/177) | fix(stems): initialize HPT vision encoders from pretrained ResNet | pretrained stem initialization with later MLP/jitter extension |
| [#189](https://github.com/GaTech-RL2/EgoVerse-graph/pull/189) | perf(arc): speed up hybrid ARC tokenization and horizon setup | vectorized tokenizer and horizon helpers |
| [#190](https://github.com/GaTech-RL2/EgoVerse-graph/pull/190) | fix(arc): preserve hold tokens and the YAM source window | held streams and bounded YAM source buffers |
| [#185](https://github.com/GaTech-RL2/EgoVerse-graph/pull/185) | fix(arc): give each arm its own rotation clock and race it | per-arm rotation clocks; obsolete shared-clock methods deleted |
| [#191](https://github.com/GaTech-RL2/EgoVerse-graph/pull/191) | feat(arc): default the bimanual arc token to the wide velocity layout | wide M28 default with explicit stacked compatibility |
| [#186](https://github.com/GaTech-RL2/EgoVerse-graph/pull/186) | fix(arc): accept either velocity layout in both arc evaluators | both-layout evaluation through the same codec |
| [#187](https://github.com/GaTech-RL2/EgoVerse-graph/pull/187) | feat(arc): point the arc model and experiment configs at the wide token | wide visual model/config contract; legacy stacked B paths retained |
| [#193](https://github.com/GaTech-RL2/EgoVerse-graph/pull/193) | fix(arc): repair M28 DTW and per-arm rotation caps | per-arm rotation caps and M28 DTW |
| [#196](https://github.com/GaTech-RL2/EgoVerse-graph/pull/196) | feat(hpt): add ABC towel shape and velocity ablations | shape/velocity shared, split and MoT ablations |
| [#197](https://github.com/GaTech-RL2/EgoVerse-graph/pull/197) | feat(hpt): add ResNet MLP image stems with training color jitter | ResNet+MLP+jitter; saved direct-ResNet checkpoints require original config |
| [#150](https://github.com/GaTech-RL2/EgoVerse-graph/pull/150) | Introduce model-owned inference and data/evaluation contracts | model-owned inference/data/evaluator interfaces |
| [#151](https://github.com/GaTech-RL2/EgoVerse-graph/pull/151) | Restore HPT recipe parity through configured graph stages | HPT graph parity and retained recipes |
| [#152](https://github.com/GaTech-RL2/EgoVerse-graph/pull/152) | Restore PI recipes and shared token diagnostics | PI parity and shared diagnostics; grouped backend calls |
| [#153](https://github.com/GaTech-RL2/EgoVerse-graph/pull/153) | Restore recorded-data visualization and conversion tools | recorded-data tools and package entrypoints |
| [#154](https://github.com/GaTech-RL2/EgoVerse-graph/pull/154) | Bind checkpoints to inference and preprocessing contracts | bound checkpoint/inference session and immutable preprocessing |
| [#155](https://github.com/GaTech-RL2/EgoVerse-graph/pull/155) | Enforce recursive configuration and integration CI gates | recursive constructor/config, CPU, wheel and CI gates |
| [#156](https://github.com/GaTech-RL2/EgoVerse-graph/pull/156) | Validate retained model recipes on source-pinned OSMO inputs | source-pinned OSMO harness; assembled GPU execution pending authorization |
| [#157](https://github.com/GaTech-RL2/EgoVerse-graph/pull/157) | Preserve source parity and reject incompatible model data | source parity and incompatible data rejection |
| [#158](https://github.com/GaTech-RL2/EgoVerse-graph/pull/158) | Bind deployment recipes to declared frames and token timing | deployment contracts and explicit codec adapters |
| [#159](https://github.com/GaTech-RL2/EgoVerse-graph/pull/159) | Verify legacy trunk initialization and capability dispositions | trunk parameter/output/loss/gradient parity retained |
| [#160](https://github.com/GaTech-RL2/EgoVerse-graph/pull/160) | YAM arc BC: canonical ARC stack + stationery/towels grid, consolidated on current main | weighted/homogeneous training, pooled stats, weights-only FT, group sampler, 64 grids and manifests |

The refreshed source metadata had no submitted human reviews or unresolved review
threads. Seventeen failed automation jobs were inspected individually: ten wiki
repository/API 404s, five missing Anthropic credential failures, and two missing
Graphite base-ref failures. These are not passing code checks or review approvals.
Fresh integration CI and real-weight/data GPU gates remain separate.
