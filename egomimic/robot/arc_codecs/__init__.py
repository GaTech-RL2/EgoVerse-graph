"""Checkpoint-era ARC codecs, selected explicitly by model-owned decoders.

These immutable compatibility copies do not replace the training tokenizers.
cartesian/e1 are from the station b1ecba63 snapshot; m28/pr193 retain their
original source pins in their headers. Never infer a codec from tensor width.
"""
