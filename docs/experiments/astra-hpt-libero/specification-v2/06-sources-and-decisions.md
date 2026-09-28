# 06 — Sources, provenance, and decisions

## Evidence status

This consolidation reread the recovered plan and inspected local EgoVerse and LIBERO source files. It did not install dependencies, fetch new papers, run the simulator, train HPT, or verify live Astra access. Repository links below are references; local inspection hashes are recorded in [local-inspection.json](sources/local-inspection.json). Local files may differ from the referenced repository revision, so the hashes are necessary and the recorded commits are not complete environment locks.

[Prior source retrieval metadata](sources/prior-retrieval-manifest.json) is copied unchanged from the earlier planning work. Its retrieval claims and paper interpretation are inherited evidence, not fresh verification. Confirm the paper version and equation, upstream compatibility, dependency versions, and actual provider access in W01.

## Primary implementation sources

| Source | Inspected or inherited evidence | Use in this plan |
|---|---|---|
| [EgoVerse](https://github.com/GaTech-RL2/EgoVerse) | Local HPT algorithm, modality/head implementations, flow head, and Hydra examples inspected | Learner architecture and concrete adapter requirements |
| [HPT algorithm](https://github.com/GaTech-RL2/EgoVerse/blob/main/egomimic/algo/hpt.py) | Domain stems, image encoders, action tokens, annotation path, optional checkpoint loading | Single `libero_push` integration and full-parameter initialization audit |
| [HPT networks](https://github.com/GaTech-RL2/EgoVerse/blob/main/egomimic/models/hpt_nets.py) | ResNet defaults to `weights="DEFAULT"`; Qwen/T5 encoders load pretrained models; reusable modality stems | Set image weights to `None`; implement a new random byte language stem |
| [Flow policy](https://github.com/GaTech-RL2/EgoVerse/blob/main/egomimic/models/fm_policy.py) | Beta flow time, interpolation/velocity targets, Euler inference | Seven-action/horizon-ten head; controlled noise/time draws |
| [Denoising policy](https://github.com/GaTech-RL2/EgoVerse/blob/main/egomimic/models/denoising_policy.py) | Base head reduces an unmasked MSE | Required local padding-mask adapter |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | Local task registration/BDDL generation code inspected; broader docs inherited | One bounded procedural simulator integration |
| [Task generation utilities](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/utils/task_generation_utils.py) | `register_task_info`, scene lookup, `generate_bddl_from_task_info` | Compile structured scene/task data into new loadable bundles |
| [Procedural creation walkthrough](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/notebooks/procedural_creation_walkthrough.ipynb) | Inherited source retrieval | W01/W04 reference for scene-template registration and asset integration |
| [robosuite controllers](https://robosuite.ai/docs/modules/controllers.html) | Inherited documentation reference | Verify the pinned stack's actual OSC frame, scaling, and gripper behavior |

The byte language stem, initialization audit, masked loss, custom `AstraPush` primitive assets, paired evaluator, and curriculum coordinator are planned work. They are not claimed to be complete upstream features.

## Research references

- [Requested paper, arXiv:2609.30063](https://arxiv.org/pdf/2609.30063): generator–learner loop and historical-gradient score; its description in this package follows the recovered notes pending W01 re-verification.
- [GenSim](https://arxiv.org/abs/2310.01361): generated simulation tasks and supervision; relevant prior art for task-generation claims.
- [RoboGen](https://arxiv.org/abs/2311.01455): generative simulation/learning pipelines; generating training problems alone is not a sufficient novelty claim.
- [Eureka](https://arxiv.org/abs/2310.12931) and [Text2Reward](https://arxiv.org/abs/2309.11489): reward-generation context. Arbitrary generated rewards are outside this pilot.
- [ALP-GMM](https://arxiv.org/abs/1910.07224), [Prioritized Level Replay](https://arxiv.org/abs/2010.03934), and [PAIRED](https://arxiv.org/abs/2012.02096): curriculum-design context and potential future comparisons. This pilot implements none of these additional arms.

## Decisions consolidated from the supplied conversation

| Earlier draft or ambiguity | Current decision |
|---|---|
| LIBERO versus Isaac / other simulators | LIBERO's MuJoCo/robosuite stack only |
| Broad research program | Small signs-of-life pilot with one training seed and two arms |
| Fixed engineering versus fixed curriculum | Fixed stack/interfaces/budgets; Astra chooses adaptive stage allocation and progression |
| Reset randomization presented as generation | New structural scenes and immutable bundles required |
| OpenVLA/OFT, then pretrained π₀.₅ | EgoVerse HPT with every learner component randomly initialized |
| `pretrained: false` interpreted as sufficient | Explicit vision/language/trunk/head audit; all trainable |
| JAX LoRA / openpi export dependencies | PyTorch full-parameter HPT training and direct HDF5 adapter |
| Large or online gradient feedback experiment | Twelve-candidate post-run diagnostic first |
| Older Astra Reversal method | Historical separate project, not part of this experiment |

New explicit design choices in this consolidation include the random byte-level language stem, fixed HPT/flow dimensions, masked loss adapter, `1e-4` full-parameter learning rate, identity normalization for already normalized actions, common-phase generation-call caps, and a cluster bootstrap over scene templates. These are planning decisions made to remove ambiguity, not empirical findings. Freeze or revise them through a new manifest version before running the study.

## Historical material and source request

The original documents are preserved under the workspace's `docs/astra-vla-curriculum/`, `docs/astra-reversal/`, and `docs/context/` folders. Their import hashes are recorded in `docs/import-manifest.json`. The older ZIP lacks the later implementation manifest and contains superseded learner choices; it is not the current package.

The supplied conversation is copied into [source-request.txt](sources/source-request.txt) so this planning package records why the complete learner must start from scratch. The prior retrieval manifest preserves literature lookup history. The active package intentionally does not duplicate obsolete plans into its reading order.

## Questions resolved by implementation gates

W01 must establish actual source/package locks, HPT/LIBERO integration, offline random initialization, device memory fit, and real Astra access. W02–W04 must establish camera semantics, teacher solvability, genuine scene novelty, and consistent full-state restoration. These are unknowns to measure, not open choices about the simulator or learner. A failed gate is reported explicitly; it does not authorize silently substituting another stack.
