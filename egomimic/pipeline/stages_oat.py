"""OAT training and inference through the shared Pipeline contract."""

from __future__ import annotations

from egomimic.pipeline.core import Stage


class OATTokenizerStage(Stage):
    reads = ("actions",)
    writes = ("loss/oat_reconstruction",)
    writes_by_mode = {"inference": ("pred_action",)}

    def __init__(self, tokenizer, use_k_tokens=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_k_tokens = use_k_tokens

    def bind_data_context(self, *, normalizer):
        self.normalizer_state = normalizer.to_state()
        self.data_context = normalizer.tokenizer_context()

    def execute(self, batch, *, mode):
        if mode == "train":
            batch["loss/oat_reconstruction"] = self.tokenizer(
                {"action": batch["actions"]}
            )
        else:
            keep = self.use_k_tokens
            if keep is not None and not 1 <= keep <= self.tokenizer.latent_horizon:
                raise ValueError("Invalid OAT prefix length")
            batch["pred_action"] = self.tokenizer.autoencode(
                batch["actions"],
                eval_keep_k=None if keep is None else [keep] * len(batch["actions"]),
            )
        return batch

    def forward(self, batch):
        return self.execute(batch, mode="train")


class OATPolicyStage(Stage):
    writes = ("loss/oat_token_ce",)
    writes_by_mode = {"inference": ("pred_action",)}

    def __init__(self, policy, use_k_tokens=None):
        super().__init__()
        self.policy = policy
        self.use_k_tokens = use_k_tokens
        self.reads = ("actions", *policy.obs_ports)
        self.reads_by_mode = {"inference": tuple(policy.obs_ports)}

    def bind_data_context(self, *, normalizer):
        self.normalizer_state = normalizer.to_state()
        self.data_context = normalizer.tokenizer_context()
        reference = getattr(
            self.policy.action_tokenizer, "_training_data_context", None
        )
        if reference is not None:
            normalizer.assert_tokenizer_context(reference)

    def train(self, mode=True):
        super().train(mode)
        # requires_grad=False alone does not disable dropout/FSQ corruption.
        self.policy.action_tokenizer.eval()
        return self

    def execute(self, batch, *, mode):
        obs = {key: batch[key] for key in self.policy.obs_ports}
        if mode == "train":
            self.policy.action_tokenizer.eval()
            batch["loss/oat_token_ce"] = self.policy(
                {"action": batch["actions"], "obs": obs}
            )
        else:
            if (
                self.use_k_tokens is not None
                and not 1 <= self.use_k_tokens <= self.policy.max_seq_len
            ):
                raise ValueError("Invalid OAT prefix length")
            batch["pred_action"] = self.policy.predict_action(
                obs,
                use_k_tokens=self.use_k_tokens,
            )["action_pred"]
        return batch

    def forward(self, batch):
        return self.execute(batch, mode="train")


class OATObservationStage(Stage):
    """Use the same released vision/state encoder for continuous ARC policies."""

    writes = ("condition",)

    def __init__(self, encoder, obs_keys):
        super().__init__()
        self.encoder = encoder
        self.reads = tuple(obs_keys)

    def forward(self, batch):
        batch["condition"] = self.encoder(
            {key: batch[key] for key in self.reads}
        ).flatten(1)
        return batch
