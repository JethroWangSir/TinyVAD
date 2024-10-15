import torch
import torch.nn as nn

class Patchify(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super(Patchify, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size, padding=0, bias=False)
        
    def forward(self, x):
        # x.shape = (batch_size, channels, height, width)
        x = self.conv(x)
        return x

if __name__ == "__main__":
    model = Patchify(1, 32, 8)
    print(model)
    dummy_input = torch.randn(1, 1, 64, 64)
    output = model(dummy_input)
    print(output.shape)
