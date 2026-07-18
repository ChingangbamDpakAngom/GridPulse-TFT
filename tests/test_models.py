import torch

from src.models.dataset import get_feature_columns
from src.models.loss import PinballLoss
from src.models.tft import TemporalFusionLite


def test_model_forward():
    model = TemporalFusionLite(input_dim=15, hidden_dim=32, horizon=48, num_quantiles=3)
    x = torch.randn(2, 168, 15)
    out = model(x)
    assert out.shape == (2, 48, 3)


def test_pinball_loss():
    criterion = PinballLoss([0.1, 0.5, 0.9])
    y_pred = torch.tensor([[[100.0, 110.0, 120.0]] * 48])
    y_true = torch.tensor([[[105.0]] * 48])
    loss = criterion(y_pred, y_true)
    assert loss.item() > 0


def test_feature_columns():
    cols = get_feature_columns()
    assert "intensity_actual" not in cols
    assert "hour_sin" in cols
    assert "lag_1h" in cols
