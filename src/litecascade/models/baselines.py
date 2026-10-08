from typing import Any

import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# B1: L1-regularised MLP (SNIPE-style)
# ==========================================
class B1_MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor):
        # x shape: (B, T, F) -> flatten to (B, T*F)
        x = x.view(x.size(0), -1)
        return self.net(x)


# ==========================================
# B2: Tree Ensembles Wrapper
# ==========================================
class B2_Trees:
    """Wrapper to behave similarly to a PyTorch model for evaluation purposes."""

    def __init__(self, model_type: str = "rf", random_state: int = 42):
        if model_type == "rf":
            self.model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=100, random_state=random_state, use_label_encoder=False, eval_metric="logloss"
            )

    def fit(self, X: Any, y: Any):
        self.model.fit(X, y)

    def predict_proba(self, X: Any):
        return self.model.predict_proba(X)

    def predict(self, X: Any):
        return self.model.predict(X)


# ==========================================
# B3: LSTM
# ==========================================
class B3_LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        # Use last timestep
        out = out[:, -1, :]
        return self.fc(out)


# ==========================================
# B4: BiLSTM
# ==========================================
class B4_BiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ==========================================
# B5: BiLSTM (for IPCA + dynamic quant)
# ==========================================
class B5_WangBiLSTM(B4_BiLSTM):
    """Exactly the same architecture as B4, but pipeline expects IPCA input."""

    pass


# ==========================================
# B6: CNN-BiLSTM (Jouhari & Guizani)
# ==========================================
class B6_CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        # Conv1d expects (B, C, L) where C is features, L is time
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # Will halve time dimension
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor):
        # x shape: (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        # Try pool, if sequence length is large enough (T >= 2)
        if x.size(2) >= 2:
            x = self.pool(x)
        # Back to (B, T', C')
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ==========================================
# B7: Depthwise-Separable CNN
# ==========================================
class B7_DSCNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        # Depthwise
        self.depthwise = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim)
        # Pointwise
        self.pointwise = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)
        x = self.gap(x).squeeze(2)
        return self.fc(x)


# ==========================================
# B8: Tiny Transformer Encoder
# ==========================================
class B8_Transformer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_heads: int = 2, num_layers: int = 2, num_classes: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 2, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.input_proj(x)
        x = self.transformer(x)
        # Use mean pooling over time
        x = x.mean(dim=1)
        return self.fc(x)
