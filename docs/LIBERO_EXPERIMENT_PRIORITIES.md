# LIBERO experiment priorities

User-directed order, September 24, 2026:

**GPU allocation:** use L40S for training, recovery and evaluation. The user
requested migration away from H100 on September 24. Preserve uploaded
checkpoints, optimizer/EMA state and existing evaluation artifacts; submit
replacement jobs under new run IDs. Keep the original global batch and total
optimizer budget when changing the number of devices.
See the [verified L40S migration snapshot](results/libero_l40s_migration_20260924.md).

1. **Finish the original ARC versus OAT evaluations.** Complete every fixed
   STK/DUR configuration on Spatial, Object, Goal and LIBERO-10. Close the
   initial-state pairing discrepancies on Spatial and LIBERO-10, investigate
   the anomalous OAT Object result, and publish full-suite and per-task scores
   with checkpoint and protocol evidence. LIBERO-90 remains deferred.
2. **Finish ARC with the released OAT diffusion-policy configuration.** Let
   existing jobs continue. After the first comparison is resolved, give recovery
   and evaluation of this campaign priority. Resume interrupted training from
   verified checkpoints at the original training budget; preserve optimizer
   and EMA state. This is the diffusion-Transformer comparison, separate from
   OAT's autoregressive policy.
3. **Run ARC + OAT.** Train the native OAT tokenizer on ARC supports, then its
   observation policy, for both STK and DUR. The implementation is retained in
   commit `67670b1fdd5199cde9d043f9e51136860041e92b`; GPU validation and training
   are deferred until the first two priorities are complete.

The two hybrid smoke workflows and two dependent full workflows submitted on
September 24 were canceled following this priority change. Their GPU tasks
had not started; only the full workflows' CPU prerequisite waiters were
running. No hybrid training checkpoints were lost. Preserve the original
specifications and submission/cancellation receipts; future launches must use
new run IDs. The 177 focused CPU tests do not constitute GPU smoke validation.

Existing ARC and OAT jobs, datasets, checkpoints and other agents' worktrees
are preserved. New hybrid submissions must remain disabled while this order
is in effect. Queue priority stays within the normal cluster allocation rules.

Current score snapshots (with unresolved issues explicitly marked):

- [Spatial](results/arc_vs_oat_libero_spatial_20260924.md)
- [Object, Goal and LIBERO-10](results/arc_vs_oat_libero_rest_20260924.md)
- [ARC + OAT implementation](LIBERO_ARC_OAT_HYBRID.md)
