import torch
import torch.nn as nn

class TinyBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TinyBlock, self).__init__()
        
        # f1: 3x3 depthwise convolution + BatchNorm
        self.f1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # f2: 1x1 grouped pointwise convolutions with 8 groups + ReLU
        self.f2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=8, bias=False),
            nn.ReLU(inplace=True)
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        f1_out = self.f1(x)
        f2_out = self.f2(x + f1_out)
        out = self.shortcut(x) + f1_out + f2_out
        return out

if __name__ == "__main__":
    model = TinyBlock(16, 16)
    print(model)
    dummy_input = torch.randn(1, 16, 8, 8)
    output = model(dummy_input)
    print(output.shape)
