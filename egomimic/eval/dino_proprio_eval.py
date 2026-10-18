"""Compatibility evaluator for original DINO recipe prediction keys."""

from egomimic.eval.eval import EvaluationDataRequirements
from egomimic.eval.image_proprio_eval import ImageProprioEval


class DinoProprioEval(ImageProprioEval):
    prediction_key = "dino_proprio_prediction"

    def data_requirements(self):
        # Preserve the archived DINO datamodule contract. Pixel diagnostics use
        # the generic evaluator and its richer metadata declaration.
        return EvaluationDataRequirements(required_keys=("world", "embodiment"))
