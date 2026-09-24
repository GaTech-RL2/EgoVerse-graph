"""Verbatim methods from EgoVerse ec5c903c HPTModel, for parity only."""

import torch
from geomloss import SamplesLoss


class LegacyAlignment:
    def resume_from_depth(self, block_outputs, depth):
        """
        Detach at trunk depth and resume trunk forward pass.
        Gradients will only flow from depth upward.
        """
        cut_tokens = block_outputs[depth - 1].detach()

        blocks = self.trunk["trunk"].blocks
        for blk in list(blocks)[depth:]:
            cut_tokens = blk(cut_tokens, attn_mask=None)

        if self.trunk["trunk"].post_transformer_layer is not None:
            cut_tokens = self.trunk["trunk"].post_transformer_layer(cut_tokens)

        return self.postprocess_tokens(cut_tokens)

    def make_custom_cost(self, scaling_mask):
        def custom_cost(x, y):
            cost = 0.5 * (((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(dim=-1))
            return cost * scaling_mask

        return custom_cost

    def compute_ot(
        self, tokens1, tokens2, emb1_actions, emb2_actions, supervised, lambd
    ):
        tokens1 = tokens1.reshape(tokens1.shape[0], -1)
        tokens2 = tokens2.reshape(tokens1.shape[0], -1)

        if not supervised:
            ot_loss_fn = SamplesLoss("sinkhorn", p=2, blur=0.05, truncate=18)
            ot_loss = ot_loss_fn(tokens2, tokens1)
            avg_feature_dist = torch.norm(tokens2 - tokens1, dim=-1).mean()
            return ot_loss, avg_feature_dist
        else:
            B = tokens1.shape[0]
            if not self.ot_6dof:
                emb1_actions = emb1_actions[..., :3]
                emb2_actions = emb2_actions[..., :3]
            if self.use_dtw:
                emb2_delta = emb2_actions
                emb1_delta = emb1_actions
                emb2_expand = emb2_delta.unsqueeze(1).expand(B, B, -1, -1)
                emb1_expand = emb1_delta.unsqueeze(0).expand(B, B, -1, -1)
                pairwise_dist = self.dtw(
                    emb2_expand.reshape(B * B, *emb2_actions.shape[1:]),
                    emb1_expand.reshape(B * B, *emb1_actions.shape[1:]),
                ).view(B, B)
            else:
                emb2_expand = emb2_actions.unsqueeze(1)  # (B, 1, T, D)
                emb1_expand = emb1_actions.unsqueeze(0)  # (1, B, T, D)
                pairwise_dist = ((emb2_expand - emb1_expand) ** 2).mean(
                    dim=(2, 3)
                )  # (B, B) #changed

            labels = torch.argmin(pairwise_dist, dim=1)
            W = torch.ones(B, B).to(self.device)
            W[torch.arange(B), labels] = lambd

            custom_cost_fn = self.make_custom_cost(W)

            ot_loss_fn = SamplesLoss(
                loss="sinkhorn", p=2, blur=0.05, cost=custom_cost_fn, truncate=18
            )

            ot_loss = ot_loss_fn(tokens2, tokens1)
            avg_feature_dist = torch.norm(tokens2 - tokens1, dim=-1).mean()
            return ot_loss, avg_feature_dist
