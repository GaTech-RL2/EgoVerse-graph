"""Absolute-pose metrics with the existing fixed-seed EnergyScore@32 evaluator."""
from egomimic.eval.libero_comparison_eval import LiberoComparisonEvaluator


class RavensComparisonEvaluator(LiberoComparisonEvaluator):
    def __init__(self, **kwargs):
        super().__init__(embodiment="ravens_ur5", **kwargs)
        self.identity.update(
            action_contract="world absolute XYZ/rotation vector/suction; 20 Hz",
            distance="equal-weight normalized XYZ/rotation-vector/suction RMS over H32",
        )

    def native_metrics(self, prediction, target):
        if prediction.shape != target.shape or prediction.shape[-2:] != (32, 7):
            raise ValueError("Expected equal H32 seven-dimensional continuous chunks")
        error = prediction - target
        return {"reconst_mse": error.square().mean(),
                "reconst_mae": error.abs().mean(),
                "position_mse_m2": error[..., :3].square().mean(),
                "rotation_vector_mse_rad2": error[..., 3:6].square().mean(),
                "suction_mse": error[..., 6].square().mean(),
                "suction_sign_accuracy": ((prediction[..., 6] > 0) == (target[..., 6] > 0)).float().mean(),
                "endpoint_position_error_m": error[:, -1, :3].norm(dim=-1).mean()}
