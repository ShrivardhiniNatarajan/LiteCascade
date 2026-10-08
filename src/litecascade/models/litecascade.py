import torch
import torch.nn as nn
import torch.nn.functional as F

class PIPLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.linear(x)

class DSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, 
            padding=padding, dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualInvertedBottleneckTCN(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.use_res_connect = in_channels == out_channels
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_res_connect:
            return x + out
        return out

class LiteCascade(nn.Module):
    def __init__(self, F_in, d, c=24, hidden=24, T_len=10, K=11, variant='R', verify_alerts=False):
        super().__init__()
        self.variant = variant
        self.K = K
        self.F_in = F_in
        self.d = d
        self.c = c
        self.T_len = T_len
        self.verify_alerts = verify_alerts
        
        # PIPLayer
        self.pip = PIPLayer(F_in, d)
        
        # Shared stem
        self.stem = DSBlock(d, c, kernel_size=3, padding=1)
        
        # Sentinel
        self.sentinel_ds = DSBlock(c, c, kernel_size=3, padding=1)
        self.sentinel_fc = nn.Linear(c, K)
        self.T1 = nn.Parameter(torch.ones(1))
        
        # Analyst
        self.analyst_blk1 = ResidualInvertedBottleneckTCN(c, c, expansion=2, dilation=1, padding=1)
        self.analyst_blk2 = ResidualInvertedBottleneckTCN(c, c, expansion=2, dilation=2, padding=2)
        
        if variant == 'R':
            self.gru = nn.GRU(input_size=c, hidden_size=hidden, batch_first=True, bidirectional=True)
            attn_in = hidden * 2
        else:
            self.gru = nn.Identity()
            attn_in = c
            
        self.attn_query = nn.Parameter(torch.randn(1, 1, attn_in))
        self.analyst_fc = nn.Linear(attn_in, K)
        
    def freeze_pip(self, freeze=True):
        for p in self.pip.parameters():
            p.requires_grad = not freeze
            
    def _analyst_forward(self, x):
        # x: (B, c, T)
        x = self.analyst_blk1(x)
        x = self.analyst_blk2(x)
        
        # (B, c, T) -> (B, T, c)
        x = x.transpose(1, 2)
        
        if self.variant == 'R':
            x, _ = self.gru(x)
            
        # Attention pooling
        # x is (B, T, attn_in)
        # attn_query is (1, 1, attn_in)
        weights = torch.matmul(x, self.attn_query.transpose(1, 2)) # (B, T, 1)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1) # (B, attn_in)
        
        return self.analyst_fc(context)

    def forward(self, x, mode="train", tau_b=0.0, tau_m=0.0):
        # x: (B, T, F_in)
        B, T_len, F_in = x.size()
        
        # PIPLayer: (B, T, F_in) -> (B, T, d)
        x = self.pip(x)
        
        # -> (B, d, T)
        x = x.transpose(1, 2)
        
        # Shared stem: (B, d, T) -> (B, c, T)
        x_stem = self.stem(x)
        
        # Sentinel
        x_sent = self.sentinel_ds(x_stem)
        # GAP -> (B, c)
        x_sent_pool = x_sent.mean(dim=-1)
        logits1 = self.sentinel_fc(x_sent_pool)
        
        if mode == "train":
            logits2 = self._analyst_forward(x_stem)
            return logits1 / self.T1, logits2
            
        elif mode == "infer":
            # inference mode
            probs1 = torch.softmax(logits1 / self.T1, dim=-1)
            preds1 = torch.argmax(probs1, dim=-1)
            
            # calculate exit mask
            exit_mask = torch.zeros(B, dtype=torch.bool, device=x.device)
            if self.verify_alerts:
                tau_m = 1.0
                
            for i in range(B):
                p1 = probs1[i]
                pred = preds1[i]
                if pred == 0 and p1[0] >= tau_b:
                    exit_mask[i] = True
                elif pred != 0 and p1.max() >= tau_m:
                    exit_mask[i] = True
                    
            preds = preds1.clone()
            
            # evaluate analyst only on non-exited
            non_exited_idx = torch.nonzero(~exit_mask, as_tuple=True)[0]
            if len(non_exited_idx) > 0:
                x_stem_non_exited = x_stem[non_exited_idx]
                logits2 = self._analyst_forward(x_stem_non_exited)
                preds2 = torch.argmax(logits2, dim=-1)
                preds[non_exited_idx] = preds2
                
            return preds, exit_mask, {}
