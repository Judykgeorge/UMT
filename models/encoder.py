import torch
import torch.nn as nn
from .patch_embedding import PatchEmbeddingForCIFAR
from .mini_umt_block import MiniUMTBlock

class MiniUMTEncoder(nn.Module):
    def __init__(self, depth=6, dim=128, num_classes=10):
        super().__init__()
        self.embedding = PatchEmbeddingForCIFAR(emb_dim=dim)
        self.blocks = nn.Sequential(*[MiniUMTBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        mod_signals = []
        for block in self.blocks:
            x = block(x)
            mod_signals.append(block.mlp.mod_signal.squeeze(-1))
        x = self.norm(x)
        return self.head(x[:, 0]), mod_signals
