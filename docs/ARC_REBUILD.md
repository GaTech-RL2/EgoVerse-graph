# ARC arrival-clock rebuild

Main review stack: [Graphite #311](https://app.graphite.com/github/pr/GaTech-RL2/EgoVerse-graph/311).
Optional projection extension: [#317](https://github.com/GaTech-RL2/EgoVerse-graph/pull/317); it is not an ancestor of the main stack tip.

Base main remains `a525d73897bec64397048e7c8897fb886555567b`. Actual main, historical #263–#279, and existing training runs are untouched. This is a source review stack, not a cluster deployment or evidence of robot performance.

## Review order

The same PR identities are retained. Numeric branch prefixes are historical;
follow these group labels and PR parents. Evaluators now have their own **Eval**
section. Projection is no longer part of Cartesian core creation on this stack.

### Preparation

| Level / PR | Change |
| --- | --- |
| 1 / [#284](https://github.com/GaTech-RL2/EgoVerse-graph/pull/284) | remove committed tests and retain configuration audits |
| 2 / [#312](https://github.com/GaTech-RL2/EgoVerse-graph/pull/312) | Withdraw inherited Cartesian ARC and preserve baseline-only integrations |

### ARC

| Level / PR | Change |
| --- | --- |
| 3 / [#290](https://github.com/GaTech-RL2/EgoVerse-graph/pull/290) | share rotation travel and first-crossing interpolation primitives |
| 4 / [#297](https://github.com/GaTech-RL2/EgoVerse-graph/pull/297) | Create translation ARC with measured-proprio incoming waypoint rates |
| 5 / [#313](https://github.com/GaTech-RL2/EgoVerse-graph/pull/313) | Add independent rotation clocks and rotation-only supports |
| 6 / [#314](https://github.com/GaTech-RL2/EgoVerse-graph/pull/314) | Add independent event-based gripper clocks |
| 7 / [#315](https://github.com/GaTech-RL2/EgoVerse-graph/pull/315) | Add distance-resampled gripper clocks |
| 8 / [#316](https://github.com/GaTech-RL2/EgoVerse-graph/pull/316) | Add absolute arrival seconds and anchor-aware monotone decoding |
| 9 / [#318](https://github.com/GaTech-RL2/EgoVerse-graph/pull/318) | Add first-moving-stream termination without synthetic hold timers |

### Pipeline

| Level / PR | Change |
| --- | --- |
| 10 / [#281](https://github.com/GaTech-RL2/EgoVerse-graph/pull/281) | preserve native action windows before interpolation |
| 11 / [#287](https://github.com/GaTech-RL2/EgoVerse-graph/pull/287) | align cameras proprioception and actions on one control clock |
| 12 / [#288](https://github.com/GaTech-RL2/EgoVerse-graph/pull/288) | preserve measured ARC anchors and metric-frame observation metadata |
| 13 / [#289](https://github.com/GaTech-RL2/EgoVerse-graph/pull/289) | bind post-tokenization normalization to exact arrival contracts |
| 14 / [#291](https://github.com/GaTech-RL2/EgoVerse-graph/pull/291) | resolve independent distance-bounded next-frame source windows |
| 15 / [#292](https://github.com/GaTech-RL2/EgoVerse-graph/pull/292) | pass explicit native decoder context and execution prefixes |
| 16 / [#282](https://github.com/GaTech-RL2/EgoVerse-graph/pull/282) | accept existing token sequences without an extra axis |
| 17 / [#283](https://github.com/GaTech-RL2/EgoVerse-graph/pull/283) | add optional clean-sample bounds during integration |
| 18 / [#285](https://github.com/GaTech-RL2/EgoVerse-graph/pull/285) | make ResNet ImageNet normalization explicit and opt-in |
| 19 / [#286](https://github.com/GaTech-RL2/EgoVerse-graph/pull/286) | add ABC-DiT through shared graph observation and flow stages |
| 20 / [#293](https://github.com/GaTech-RL2/EgoVerse-graph/pull/293) | preserve W&B identity across automatic Lightning resumes |
| 21 / [#301](https://github.com/GaTech-RL2/EgoVerse-graph/pull/301) | replace campaign-named image stem with shared ResNet-MLP implementation |

### Eval

| Level / PR | Change |
| --- | --- |
| 22 / [#298](https://github.com/GaTech-RL2/EgoVerse-graph/pull/298) | score world-frame arrival-clock full-episode DTW |
| 23 / [#299](https://github.com/GaTech-RL2/EgoVerse-graph/pull/299) | evaluate next-frame open-loop chunks and videos from measured anchors |

### Robot

| Level / PR | Change |
| --- | --- |
| 24 / [#300](https://github.com/GaTech-RL2/EgoVerse-graph/pull/300) | decode measured-proprio incoming ARC for robot execution |
| 25 / [#294](https://github.com/GaTech-RL2/EgoVerse-graph/pull/294) | add bounded rollout IK recovery and selectable endpoint solvers |
| 26 / [#295](https://github.com/GaTech-RL2/EgoVerse-graph/pull/295) | record demonstration-compatible rollout episodes and tempo summaries |
| 27 / [#296](https://github.com/GaTech-RL2/EgoVerse-graph/pull/296) | add standalone camera alignment diagnostics |
| 28 / [#306](https://github.com/GaTech-RL2/EgoVerse-graph/pull/306) | connect bound controls recording and reset guides to the rollout dashboard |

### Config

| Level / PR | Change |
| --- | --- |
| 29 / [#302](https://github.com/GaTech-RL2/EgoVerse-graph/pull/302) | factor reusable model bases and HPT PI component groups |
| 30 / [#303](https://github.com/GaTech-RL2/EgoVerse-graph/pull/303) | organize simulation recipes under arc_sim and update launcher selectors |
| 31 / [#304](https://github.com/GaTech-RL2/EgoVerse-graph/pull/304) | integrate arrival-v2 source offsets and explicit experiment contracts |
| 32 / [#305](https://github.com/GaTech-RL2/EgoVerse-graph/pull/305) | remove remaining E1 campaign runtime and temporary withdrawal support |
| 33 / [#309](https://github.com/GaTech-RL2/EgoVerse-graph/pull/309) | update PI launch examples and integration matrix to canonical config paths |

### Tools

| Level / PR | Change |
| --- | --- |
| 34 / [#307](https://github.com/GaTech-RL2/EgoVerse-graph/pull/307) | add checkpoint sweeps and explicit validation-only logging |
| 35 / [#308](https://github.com/GaTech-RL2/EgoVerse-graph/pull/308) | add source-window measurements and paired ARC representation inspection |
| 36 / [#310](https://github.com/GaTech-RL2/EgoVerse-graph/pull/310) | package ARC inspection assets and lock optional TracIK dependency |

### Docs

| Level / PR | Change |
| --- | --- |
| 37 / [#280](https://github.com/GaTech-RL2/EgoVerse-graph/pull/280) | use the authorized GitHub account for Graphite publication |
| 38 / [#311](https://github.com/GaTech-RL2/EgoVerse-graph/pull/311) | map the feature-sized rebuild and current configuration contracts |

## Behavioral changes in this revision

- Cartesian ARC always uses independent streams. There are no selectable race,
  cumulative/joint-distance or multistream algorithms. Cumulative geometric
  length remains a necessary internal calculation, not a mode.
- The layouts are full separation (M×20), translation/rotation (M×18),
  translation-only and rotation-only (M×16). Projection is optional elsewhere.
- Native ARC source rows start at the next recorded frame, not the observation
  frame. Baseline indexing is unchanged. Native ARC stride must be one.
- Current measured pose is passed explicitly in the action frame. Row zero is
  an incoming speed for PW, or an absolute arrival time for duration, just like
  the later incoming intervals. Decoder queries are dt, 2dt, …, not time zero.
- There is no leading-delay extraction, synthetic 400-frame hold target, dwell
  event allocation or duration floor. PW cannot encode the duration of exactly
  zero-distance intervals; all-static PW uses the caller's explicit horizon.
- Normalization still runs after tokenization. Exact arrival-v2 contracts reject
  old or missing normalization bindings; shapes alone do not establish compatibility.
- Eval and robot adapters supply the same measured native anchor. Context is
  model-declared; no class-name or token-width inference supplies a starting pose.

These are intentional algorithmic changes, not a numerical-parity rebuild of
#279. The earlier rebuild at `2cec3b844046c967f33d326298735d0a482076ee`
preserved that old behavior; its sources and verification remain in Git history.
New training requires fresh compatible statistics and experiment identities.

## Scope and integration

Preparation withdraws only Cartesian/bimanual ARC; independent planar/simulation
codecs remain. Six incremental codec snapshots build incoming-rate translation,
rotation layouts, event grippers, distance grippers, absolute clocks, and
first-stream termination. Unsupported options are absent at each stage.

The configuration migration remains the activation boundary for complete
experiment menus and renamed runtime entry points. Intermediate helper layers
are reviewable components, not production launch candidates. New methods,
configs, normalization and rollout context must be deployed together.

Event grippers still cover changes over the supplied source window, downsample
when supports exceed M and repeat the endpoint to fill M. No new event-count
budget is invented. Their zero-distance PW dwell-loss limitation remains; see
[clock contract](ARC_CLOCK_CONTRACT.md) for details and source-distance caps.

## Canonical paths and checks

- Cartesian codec: `egomimic/rldb/zarr/cartesian_arc.py` (stable import from `arc_length_tokenizer.py`).
- Episode evaluation: `egomimic/eval/open_loop_stream.py` (stable `open_loop_sim.py`).
- Robot decoding: `egomimic/robot/stream_decoder.py` (stable `arc_decoder.py`).
- [Configuration](CONFIGURATION.md), [clock contract](ARC_CLOCK_CONTRACT.md),
  [DTW contract](DISTANCE_BUDGET_DTW.md), [no committed tests](VERIFICATION_POLICY.md).

Verification scripts/fixtures remain outside the application repository. CPU
oracle, frame/context, normalization, source-offset, config and staged-codec
checks do not establish hardware safety or real-robot success. No training job
or physical robot is started by this rebuild.
