import torch
import torch.nn as nn
import torch.nn.functional as F

class CARPooledMLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mlp_ratio)
        self.fc2 = nn.Linear(dim * mlp_ratio, dim)
        self.dropout = nn.Dropout(dropout)
        self.global_summary_proj = nn.Linear(dim, dim)
        self.token_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(1, 1)
        self.mod_signal = None

    def forward(self, x):
        global_summary = x.mean(dim=1, keepdim=True)
        summary_embed = self.global_summary_proj(global_summary)
        token_embed = self.token_proj(x)
        alignment = F.cosine_similarity(token_embed, summary_embed, dim=-1, eps=1e-6).unsqueeze(-1)
        gate = torch.sigmoid(self.gate_proj(alignment))
        self.mod_signal = gate
        y = self.fc2(F.gelu(self.fc1(x)))
        return x + self.dropout(y * gate)
