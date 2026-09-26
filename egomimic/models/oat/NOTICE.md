# OAT source provenance

The model, tokenizer and perception subpackages were ported from
<https://github.com/Chaoqi-LIU/oat/tree/1da92695ef12c23b7000a0b1a76cab0aef4750e6>.
`UPSTREAM.json` records the source SHA-256 of every transferred file.
Their module imports use the native `egomimic.models.oat` namespace. Network
definitions, parameter names, masks, quantization and numerical operations are
preserved. The base checkpoint methods now load native Pipeline checkpoints;
the OAT workspace/training runtime is not included or required.
Formatting and import cleanup follow this repository's lint rules. A shadowed
duplicate FSQ `__repr__` definition was removed (the effective definition is
unchanged), and unused local names in upstream's normalizer self-test were
marked as unused. These edits do not alter model calculations.

`LICENSE` is the upstream license, including its third-party exception.
Several tokenizer files carry EPFL–Apple Sample Code License (Non-Commercial)
headers; those headers are retained and those files are not relicensed as MIT.
The quantizer retains its attribution to the original FSQ implementation.
Upstream's root license refers to a NOTICE file that is absent from this pinned
revision; its referenced `apple/ml-oat` repository was unavailable during this
port. No replacement license text has been invented.

`factory.py` and `checkpoint.py` are native integration code. Graph adapters,
dataset adapters and benchmark code live in EgoVerse's existing namespaces.
See `docs/ARC_OAT.md` for protocol corrections and numerical validation.
