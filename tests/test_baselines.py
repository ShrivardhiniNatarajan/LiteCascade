import pytest
import torch

from litecascade.eval.profiler import profile_model
from litecascade.models.baselines import B1_MLP, B3_LSTM, B7_DSCNN, B4_BiLSTM, B5_WangBiLSTM, B6_CNN_BiLSTM, B8_Transformer


@pytest.mark.parametrize("model_class", [B1_MLP, B3_LSTM, B4_BiLSTM, B5_WangBiLSTM, B6_CNN_BiLSTM, B7_DSCNN, B8_Transformer])
def test_baselines_forward_and_profile(model_class):
    B = 2
    T = 10
    F = 12

    input_dim = F if model_class != B1_MLP else T * F
    model = model_class(input_dim, hidden_dim=16, num_classes=2)

    x = torch.randn(B, T, F)

    logits = model(x)
    assert logits.shape == (B, 2), f"{model_class.__name__} output shape mismatch."

    # Profile
    x_single = torch.randn(1, T, F)
    prof = profile_model(model, x_single)
    assert prof["params"] > 0
    assert prof["macs"] > 0
