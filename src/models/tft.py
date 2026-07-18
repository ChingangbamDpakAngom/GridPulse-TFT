import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalFusionLite(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        horizon: int = 48,
        num_quantiles: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_quantiles = num_quantiles

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.encoder_lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )

        self.decoder = nn.Linear(hidden_dim, horizon * num_quantiles)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        x = self.input_projection(x)

        lstm_out, _ = self.encoder_lstm(x)

        attn_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)

        context = attn_out[:, -1, :]
        context = self.dropout(context)

        quantile_out = self.decoder(context)
        quantile_out = quantile_out.view(batch_size, self.horizon, self.num_quantiles)

        return quantile_out


def get_model(input_dim: int, **kwargs) -> TemporalFusionLite:
    return TemporalFusionLite(input_dim=input_dim, **kwargs)