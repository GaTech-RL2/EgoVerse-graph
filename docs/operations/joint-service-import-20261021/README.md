# Verified operational service import

This import records the exact 23-file deployed operational service named `9a2982a8a611b9d377f7719dbd710161b39e4322` in the Lambda task `joint-modality-ablations-20261020/services/`. Its `FILES.sha256` bytes have SHA256 `1714c1b459dcef0dbe372651f0087e59391231f39f729896b7d75df5ce08d4ae`; every file was verified against that manifest before import.

The name is a deployment identifier. Its Git ancestry was not established, and this commit does not assert that it descends from that identifier as a Git commit. The actual Git parent is the published modality-ablation branch at `255fdaf5137f7043b35df7ef48a3c1b8adc70815`. This creates a reviewable source home for verified operational bytes; it does not alter existing deployed releases or retrofit their provenance.

Only orchestration/verification scripts and their declared dependency closure are imported. No `egomimic` model, data or configuration file changes. Existing scientific checkpoints and new image/proprio requests retain the separately verified frozen source `535d4fc26c4c85cf57bbd2a776cc79cfec085c4e` and its own archive/manifest bindings.
