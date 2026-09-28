"""Released FAST autoregression through the shared graph contract."""

from egomimic.pipeline.core import Stage


class FASTPolicyStage(Stage):
    writes = ("loss/fast_token_ce",)
    writes_by_mode = {"inference": ("pred_action",)}

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.reads = ("actions", *policy.obs_ports)
        self.reads_by_mode = {"inference": tuple(policy.obs_ports)}

    def bind_data_context(self, *, normalizer):
        self.normalizer_state = normalizer.to_state()
        self.data_context = normalizer.tokenizer_context()
        reference = self.policy.action_tokenizer._training_data_context
        if reference is not None:
            normalizer.assert_tokenizer_context(reference)

    def train(self, mode=True):
        super().train(mode)
        self.policy.action_tokenizer.eval()
        return self

    def execute(self, batch, *, mode):
        obs = {key: batch[key] for key in self.policy.obs_ports}
        if mode == "train":
            batch["loss/fast_token_ce"] = self.policy(
                {"action": batch["actions"], "obs": obs}
            )
        else:
            batch["pred_action"] = self.policy.predict_action(obs)["action_pred"]
        return batch

    def forward(self, batch):
        return self.execute(batch, mode="train")
