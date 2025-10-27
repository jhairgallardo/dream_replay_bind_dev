import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResizeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, padding=1, padding_mode='reflect', mode='bilinear', bias=False): # bilinear
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias, padding_mode=padding_mode)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, antialias=True)
        x = self.conv(x)
        return x
    
class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        planes = out_planes
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect')
        self.norm2 = nn.GroupNorm(min([32, in_planes//4]), in_planes)
        self.act = nn.Mish()
        if stride == 1: # Not changing spatial size
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect')
            self.norm1 = nn.GroupNorm(min([32, planes//4]), planes)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='reflect'),
                nn.GroupNorm(min([32, planes//4]), planes)
            )
        else: # Changing spatial size (expanded by x2)
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride, padding_mode='reflect')
            self.norm1 = nn.GroupNorm(min([32, planes//4]), planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride, padding_mode='reflect'),
                nn.GroupNorm(min([32, planes//4]), planes)
            )
    def forward(self, x):
        out = self.act(self.norm2(self.conv2(x)))
        out = self.norm1(self.conv1(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out

class Generator_Network(nn.Module):
    def __init__(self,
                 in_planes=192,
                 num_Blocks=[1,1,1,1], 
                 nc=3):
        super().__init__()
        self.in_planes = in_planes
        self.out_act = lambda x: torch.tanh(x)

        # Since the feature tokens start already at 14x14, we make layer 4 to output the same spatial size by doing stride 1.
        # (This is the case for ViT with 196 tokens (patch size of 16 on 224x224 images)
        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=1) # stride 1 to not change spatial size
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=2)

        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2, padding=1, padding_mode='reflect', bias=True) ## 3x3 kernel size
        # init conv1 bias as zero
        nn.init.constant_(self.conv1.conv.bias, 0)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Similar to zero init residual
        for m in self.modules():
            if isinstance(m, BasicBlockDec) and m.norm1.weight is not None:
                nn.init.constant_(m.norm1.weight, 0) # shutdown the main path and only let the residual path pass at init

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockDec(self.in_planes, planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input x is shape (B, V, Timg, Dimg)
        B, V, Timg, Dimg = x.shape
        # Reshape x to (B*V, Timg, Dimg)
        x = x.reshape(B*V, Timg, Dimg)
        # Reshape it to (B*V, Dimg, 14, 14)
        x = x.permute(0, 2, 1) # (B*V, Dimg, Timg)
        h = math.sqrt(Timg) # 14 if Timg = 196
        w = math.sqrt(Timg) # 14 if Timg = 196
        x = x.reshape(x.shape[0], x.shape[1], int(h), int(w)) # (B*V, Dimg, h, w)
        # Now we can pass it through the decoder
        x = self.layer4(x) # (B*V, 256, h, w)
        x = self.layer3(x) # (B*V, 128, 28, 28)
        x = self.layer2(x) # (B*V, 64, 56, 56)
        x = self.layer1(x) # (B*V, 64, 112, 112)
        x = self.out_act(self.conv1(x)) # (B*V, 3, 224, 224)
        # Reshape x to (B, V, 3, 224, 224)
        x = x.reshape(B, V, 3, 224, 224)
        return x