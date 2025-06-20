import torch
import torch.nn as nn
from .carp_mlp import CARPooledMLP

class MiniUMTBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = CARPooledMLP(dim, mlp_ratio, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input)
        attn_mod_signal = self.attn_gate(attn_input.mean(dim=1)).unsqueeze(-1)
        x = x + self.dropout(attn_output * (1 + attn_mod_signal))
        return self.mlp(self.norm2(x))
