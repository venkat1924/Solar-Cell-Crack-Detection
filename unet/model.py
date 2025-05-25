# elpv_segmentation/model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F # Import F for interpolate

# ... (ConvBNAct, DilatedBottleneck, DecoderBlock, UpBlock classes remain the same) ...
class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, inputs): return self.seq.forward(inputs)

class DilatedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=2):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, dilation=dilation, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        return torch.cat((x, out), dim=1)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels)
        self.conv2 = ConvBNAct(out_channels, out_channels)
        self.conv3 = ConvBNAct(out_channels, out_channels)
    def forward(self, x): return self.conv3(self.conv2(self.conv1(x)))

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    def forward(self, x_bottom): return self.conv(self.upsample(x_bottom))

class UNetWithVGG16BN(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.n_classes = n_classes
        # Ensure you are using an up-to-date way to get pretrained weights if VGG16_BN_Weights.IMAGENET1K_V1 is deprecated
        # For older torchvision, models.vgg16_bn(pretrained=pretrained) might be used.
        # For newer (0.13+), the weights enum is preferred.
        if hasattr(models, 'VGG16_BN_Weights'):
            weights = models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
            vgg16_bn_model = models.vgg16_bn(weights=weights)
        else: # Fallback for older torchvision
            vgg16_bn_model = models.vgg16_bn(pretrained=pretrained)
            
        features = vgg16_bn_model.features

        # VGG16_BN MaxPool indices: 6, 13, 23, 33, 43
        self.enc1 = features[:7]    # Output after features[6] (MaxPool2d): H/2, W/2
        self.enc2 = features[7:14]  # Output after features[13] (MaxPool2d): H/4, W/4
        self.enc3 = features[14:24] # Output after features[23] (MaxPool2d): H/8, W/8
        self.enc4 = features[24:34] # Output after features[33] (MaxPool2d): H/16, W/16
        self.enc5 = features[34:44] # Output after features[43] (MaxPool2d): H/32, W/32

        self.bottleneck = DilatedBottleneck(512, 512) # Input 512 channels, output 1024 channels

        # Decoder
        self.up4 = UpBlock(1024, 512) # Input from bottleneck (1024), output 512
        self.dec4 = DecoderBlock(512 + 512, 512) # Skip from enc4 (512)

        self.up3 = UpBlock(512, 256)
        self.dec3 = DecoderBlock(256 + 256, 256) # Skip from enc3 (256)

        self.up2 = UpBlock(256, 128)
        self.dec2 = DecoderBlock(128 + 128, 128) # Skip from enc2 (128)

        self.up1 = UpBlock(128, 64)
        self.dec1 = DecoderBlock(64 + 64, 64)   # Skip from enc1 (64)

        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x): # x is the input image, e.g., (B, 3, H_input, W_input)
        # Encoder
        e1 = self.enc1(x)    # Spatial: H_input/2, W_input/2
        e2 = self.enc2(e1)   # Spatial: H_input/4, W_input/4
        e3 = self.enc3(e2)   # Spatial: H_input/8, W_input/8
        e4 = self.enc4(e3)   # Spatial: H_input/16, W_input/16
        e5 = self.enc5(e4)   # Spatial: H_input/32, W_input/32

        # Bottleneck
        b = self.bottleneck(e5) # Spatial: H_input/32, W_input/32

        # Decoder
        d4_up = self.up4(b)     # Spatial: H_input/16, W_input/16
        d4_cat = torch.cat((d4_up, e4), dim=1)
        d4 = self.dec4(d4_cat)  # Spatial: H_input/16, W_input/16

        d3_up = self.up3(d4)    # Spatial: H_input/8, W_input/8
        d3_cat = torch.cat((d3_up, e3), dim=1)
        d3 = self.dec3(d3_cat)  # Spatial: H_input/8, W_input/8

        d2_up = self.up2(d3)    # Spatial: H_input/4, W_input/4
        d2_cat = torch.cat((d2_up, e2), dim=1)
        d2 = self.dec2(d2_cat)  # Spatial: H_input/4, W_input/4

        d1_up = self.up1(d2)    # Spatial: H_input/2, W_input/2
        d1_cat = torch.cat((d1_up, e1), dim=1)
        d1 = self.dec1(d1_cat)  # Spatial: H_input/2, W_input/2 (e.g., 128x128 if input was 256x256)

        # *** Add final upsampling to match input spatial dimensions ***
        # x.shape[2:] dynamically gets (H_input, W_input) from the original input tensor x
        final_features_upsampled = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=True)
        # final_features_upsampled now has spatial dimensions: H_input, W_input (e.g., 256x256)

        return self.outconv(final_features_upsampled)