import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    InfoNCE loss with a static global memory bank.
    """

    def __init__(self, fp_tail_model: nn.Module, tau: float = 0.1, normalize: bool = True):
        super().__init__()
        self.fp_tail_model = fp_tail_model
        self.tau = tau
        self.normalize = normalize

        # Keep the FP tail frozen as a teacher feature mapper.
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

    def forward(self, act_quant: torch.Tensor, v_f_global: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        :param act_quant: Quantized activations with shape [B, C, H, W].
        :param v_f_global: Precomputed FP features in the global memory bank with shape [N, Dim].
        :param batch_idx: Positive indices for the current batch in the global memory bank.
        """
        # Map quantized activations into the contrastive space.
        v_q = self._map_features(act_quant)  # [B, Dim]

        # Compute similarities between batch features and the global memory bank.
        sim = torch.mm(v_q, v_f_global.t()) / self.tau

        # Cross-entropy selects each sample's matching FP feature as the positive.
        loss = F.cross_entropy(sim, batch_idx)

        return loss
