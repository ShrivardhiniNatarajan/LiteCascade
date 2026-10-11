import torch
import torch.nn as nn

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * x, dim=1)
        return context

class TeacherCNNBiLSTM(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 11):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.attention = AttentionPooling(hidden_dim=256)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x is (B, T, F)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        context = self.attention(lstm_out)
        return self.fc(context)
