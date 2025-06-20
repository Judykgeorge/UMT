import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class PatchEmbeddingForCIFAR(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.feature_extractor = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        )
        self.projector = nn.Conv2d(256, emb_dim, kernel_size=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embed = None

    def forward(self, x):
        B = x.size(0)
        x = self.feature_extractor(x)
        x = self.projector(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is None or self.pos_embed.shape[1] != x.shape[1]:
            self.pos_embed = nn.Parameter(torch.randn(1, x.shape[1], x.shape[2], device=x.device))
        return x + self.pos_embed
