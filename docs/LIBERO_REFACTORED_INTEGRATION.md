# LIBERO and refactored Action Flow integration

This isolated branch combines the refactored Action Flow source with the
unpublished LIBERO global-basis P4 source snapshot. It is a source integration,
not a training run or a migration of any existing checkpoint.

- Refactored starting commit and preserved return branch:
  `8e95c44269cc1d924b7137b03ba8ac77f8f425df`,
  `codex/refactored-pre-libero-20260929`.
- LIBERO P4 source snapshot: `d1a575f47bd5360ba5bf4c7c91172af393e4f557`,
  from the task-local clean `source-p4` checkout on Skynet. Its Git export is
  shallow; it is not a published GitHub branch head.
- Recorded LIBERO base: `5ac94e3f7062d587e8d1656c6655cc7550e50a8a`.
  The P4 snapshot tree differs from this base in 26 paths. Commit
  `ec3034fae7149924b971a53077daa4abb7bb0e6a` reconstructs that
  ancestry for this isolated merge only. It must not be presented as the
  original LIBERO commit history.

The two merge conflicts were resolved by retaining both the refactored
loader's seed/compatibility controls and LIBERO's ordered validation controls,
and by retaining the existing embodiment-override constructor contract while
passing specialized resolver arguments. The refactored Action Flow model and
grouped-backward implementation are not edited by the merge.

Before any LIBERO training, separately resolve its exact model/configuration,
dataset and normalization, constructed parameter count, source cleanliness,
runtime, validation contract, optimizer budget, storage, and launch smoke.
In particular, this merge does not turn the 27M OAT Transformer DP into the
~263M PushShapes paper-style conditional U-Net DP.
