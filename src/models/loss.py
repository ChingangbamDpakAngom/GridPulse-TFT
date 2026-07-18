import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    def __init__(self, quantiles: list[float] = None):
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.register_buffer("q_tensor", torch.tensor(self.quantiles, dtype=torch.float32))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(-1).expand(-1, -1, len(self.quantiles))

        errors = y_true - y_pred
        losses = torch.max(self.q_tensor * errors, (self.q_tensor - 1) * errors)
        return losses.mean()