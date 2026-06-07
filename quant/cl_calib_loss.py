import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CLCalibLoss(nn.Module):
    """
    CL-Calib contrastive loss.
    Reference: "Enhancing Post-training Quantization Calibration through Contrastive Learning".
    """

    def __init__(self, fp_tail_model: nn.Module, tau: float = 0.1,
                 num_neg: Optional[int] = None, normalize: bool = True):
        super().__init__()
        self.fp_tail_model = fp_tail_model
        self.tau = tau
        self.num_neg = num_neg
        self.normalize = normalize

        # Keep the FP tail frozen as the feature mapper g = f[k:]_F.
        for p in self.fp_tail_model.parameters():
            p.requires_grad_(False)
        self.fp_tail_model.eval()

    def _map_features(self, act: torch.Tensor) -> torch.Tensor:
        v = self.fp_tail_model(act)
        if v.dim() > 2:
            v = v.flatten(1)
        if self.normalize:
            v = F.normalize(v, dim=-1, p=2)
        return v

    def forward(self, act_quant: torch.Tensor, v_f: torch.Tensor) -> torch.Tensor:
        B = act_quant.shape[0]
        assert B > 1, "CL-Calib requires batch size > 1 to form negative pairs."

        # Map quantized features into the contrastive space.
        v_q = self._map_features(act_quant)

        # Use externally precomputed FP features.
        sim = torch.mm(v_q, v_f.t()) / self.tau

        # Positive pairs are on the diagonal.
        pos_scores = torch.diagonal(sim)
        pos_loss = -F.logsigmoid(pos_scores).mean()

        # Negative pairs are all off-diagonal entries.
        neg_mask = ~torch.eye(B, dtype=torch.bool, device=act_quant.device)

        if self.num_neg is not None:
            neg_loss_list = []
            for i in range(B):
                neg_sims_i = sim[i][neg_mask[i]]
                if self.num_neg < B - 1:
                    idx = torch.randperm(B - 1, device=act_quant.device)[:self.num_neg]
                    neg_sims_i = neg_sims_i[idx]
                neg_loss_list.append(-F.logsigmoid(-neg_sims_i).mean())
            neg_loss = torch.stack(neg_loss_list).mean()
        else:
            # Use all other samples in the batch as negatives.
            neg_sims = sim[neg_mask]
            neg_loss = -F.logsigmoid(-neg_sims).mean()

        return pos_loss + neg_loss
