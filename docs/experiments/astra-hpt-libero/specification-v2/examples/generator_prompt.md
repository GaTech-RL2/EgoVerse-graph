# Proposed Astra generation prompt

This prompt template requires an actual provider adapter and typed scene/task/teacher schemas. It does not define a public model ID or a runnable API request.

```text
You design training problems for a randomly initialized EgoVerse HPT robot
learner in a fixed LIBERO pushing simulator. Use only the supplied asset,
scene, instruction, and teacher-skill grammar.

Choose exactly five task templates with positive integer episode allocations.
The total is 75. Allocate among S1 basic pushing, S2 color/target selection,
and S3 spatial-reference selection. You may revisit or skip stages.

For the adaptive arm, use only the supplied TRAINING-CONTROL report and
training archive. Begin with easier tasks when warranted, increase rigor
when evidence supports it, and revisit failures as needed. There is no
mastery threshold or prescribed stage schedule. Explain which measured
failures motivate your allocation and what you predict will improve.

For the uniform arm, obey quotas S1=25, S2=25, S3=25. No learner report is
provided. Do not invent learner performance or infer it from another arm.

Supply each template's SceneSpec, TaskSpec, instruction, and typed teacher
program. Create structurally new scenes within the catalog: renaming tasks,
changing only instruction wording, or sampling another reset coordinate
does not count. At least two new structural templates must survive this round.

Use geometry and teacher-validity errors only to repair your proposals.
Do not redefine success predicates or output executable scene/reward code.
Do not request development or sealed-test tasks, metrics, or frames.
Do not present a predicted outcome as a measured one.

Return only the structured CurriculumDecision accepted by the supplied
schema. Include stage allocations, five templates, rationale, predicted
probe changes, and advance/consolidate/revisit intent.
```

Attach schema versions, approved catalogs, generation-only duplicate summaries, remaining attempt/call/token budgets, and—only for A—the training report and up to six selected training-control frame sequences. Save the assembled request verbatim with all artifact hashes and provider metadata. In seed and commissioning phases, use their separately specified quotas and objectives instead of the branch-round five-template/75-episode instruction.
