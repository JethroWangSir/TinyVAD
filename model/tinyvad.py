import torch
import torch.nn as nn
from .patchify import Patchify
from .csp_tiny_layer import CSPTinyLayer

class TinyVAD(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, patch_size, num_blocks):
        super(TinyVAD, self).__init__()
        self.patchify = Patchify(in_channels, hidden_channels, patch_size)
        self.layer1 = CSPTinyLayer(hidden_channels, hidden_channels, num_blocks)
        self.layer2 = CSPTinyLayer(hidden_channels, hidden_channels, num_blocks)
        self.layer3 = CSPTinyLayer(hidden_channels, out_channels, num_blocks)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(out_channels, 1)

    def forward(self, x):
        x = self.patchify(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

if __name__ == "__main__":
    model = TinyVAD(1, 32, 64, 8, 2)
    print(model)
    dummy_input = torch.randn(1, 1, 64, 64)
    output = model(dummy_input)
    print(output)
    