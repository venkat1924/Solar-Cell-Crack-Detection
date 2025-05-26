import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F # Import F for interpolate

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
        # In this design, out_channels refers to the channels of the dilated convolutions
        # The block will output in_channels + out_channels due to concatenation
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
        self.conv3 = ConvBNAct(out_channels, out_channels) # Added for consistency if this was intended
                                                          # Original was self.conv2(self.conv1(x)), implying 2 layers.
                                                          # If 3 layers: self.conv3(self.conv2(self.conv1(x)))
                                                          # For 2 layers: remove self.conv3 and change forward
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) # if 3 layers
        return x
        # If only two conv layers were intended in DecoderBlock as per original structure:
        # return self.conv2(self.conv1(x))


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1) # 1x1 conv to adjust channels

    def forward(self, x_bottom):
        x_bottom_upsampled = self.upsample(x_bottom)
        return self.conv(x_bottom_upsampled)


class UNetWithConvNeXtBase(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super().__init__()
        self.n_classes = n_classes

        if hasattr(models, 'ConvNeXt_Base_Weights'):
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            convnext_model = models.convnext_base(weights=weights)
        else: # Fallback for older torchvision versions
            convnext_model = models.convnext_base(pretrained=pretrained)

        cn_features = convnext_model.features

        # ConvNeXt features structure:
        # features[0]: stem (Conv2d, LayerNorm) -> H/4, W/4, C=128
        # features[1]: stage 1 (blocks)       -> H/4, W/4, C=128
        # features[2]: downsampling           -> H/8, W/8, C=256 (input for stage2)
        # features[3]: stage 2 (blocks)       -> H/8, W/8, C=256
        # features[4]: downsampling           -> H/16, W/16, C=512 (input for stage3)
        # features[5]: stage 3 (blocks)       -> H/16, W/16, C=512
        # features[6]: downsampling           -> H/32, W/32, C=1024 (input for stage4)
        # features[7]: stage 4 (blocks)       -> H/32, W/32, C=1024

        self.enc1_stem_s1 = nn.Sequential(cn_features[0], cn_features[1]) # Output: 128 ch, H/4
        self.enc2_ds_s2 = nn.Sequential(cn_features[2], cn_features[3])   # Output: 256 ch, H/8
        self.enc3_ds_s3 = nn.Sequential(cn_features[4], cn_features[5])   # Output: 512 ch, H/16
        self.enc4_ds_s4 = nn.Sequential(cn_features[6], cn_features[7])   # Output: 1024 ch, H/32 (to bottleneck)

        # Bottleneck: Takes 1024 from enc4, DilatedBottleneck(in, out) outputs in+out
        # We want DilatedBottleneck's internal convs to also be 1024.
        self.bottleneck = DilatedBottleneck(1024, 1024) # Output: 1024 (input) + 1024 (processed) = 2048 ch

        # Decoder
        # Upsample from bottleneck (2048 ch) to match enc3 (512 ch) spatial dims (H/16)
        self.up3 = UpBlock(2048, 512) # Output 512 ch for concatenation
        self.dec3 = DecoderBlock(512 + 512, 512) # Skip from enc3 (512 ch)

        # Upsample from dec3 (512 ch) to match enc2 (256 ch) spatial dims (H/8)
        self.up2 = UpBlock(512, 256)
        self.dec2 = DecoderBlock(256 + 256, 256) # Skip from enc2 (256 ch)

        # Upsample from dec2 (256 ch) to match enc1 (128 ch) spatial dims (H/4)
        self.up1 = UpBlock(256, 128)
        self.dec1 = DecoderBlock(128 + 128, 128) # Skip from enc1 (128 ch)

        # Output convolution. The output of dec1 is H/4, W/4 with 128 channels.
        # We will upsample this to original image size before the final convolution.
        self.outconv = nn.Conv2d(128, n_classes, kernel_size=1)


    def forward(self, x): # x is (B, 3, H_input, W_input)
        # Encoder
        e1 = self.enc1_stem_s1(x)  # Spatial: H_input/4, W_input/4; Channels: 128
        e2 = self.enc2_ds_s2(e1)   # Spatial: H_input/8, W_input/8; Channels: 256
        e3 = self.enc3_ds_s3(e2)   # Spatial: H_input/16, W_input/16; Channels: 512
        e4 = self.enc4_ds_s4(e3)   # Spatial: H_input/32, W_input/32; Channels: 1024

        # Bottleneck
        b = self.bottleneck(e4)    # Spatial: H_input/32, W_input/32; Channels: 2048

        # Decoder
        d3_up = self.up3(b)        # Spatial: H_input/16, W_input/16
        d3_cat = torch.cat((d3_up, e3), dim=1) # Channels: 512 (up) + 512 (e3) = 1024
        d3 = self.dec3(d3_cat)     # Spatial: H_input/16, W_input/16; Channels: 512

        d2_up = self.up2(d3)       # Spatial: H_input/8, W_input/8
        d2_cat = torch.cat((d2_up, e2), dim=1) # Channels: 256 (up) + 256 (e2) = 512
        d2 = self.dec2(d2_cat)     # Spatial: H_input/8, W_input/8; Channels: 256

        d1_up = self.up1(d2)       # Spatial: H_input/4, W_input/4
        d1_cat = torch.cat((d1_up, e1), dim=1) # Channels: 128 (up) + 128 (e1) = 256
        d1 = self.dec1(d1_cat)     # Spatial: H_input/4, W_input/4; Channels: 128

        # Final upsampling to match input spatial dimensions
        # d1 is currently at H_input/4, W_input/4
        final_features_upsampled = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=True)
        # final_features_upsampled now has spatial dimensions: H_input, W_input

        return self.outconv(final_features_upsampled)

# Example Usage:
if __name__ == '__main__':
    # Create a dummy input tensor (Batch, Channels, Height, Width)
    # ConvNeXt typically expects input size >= 224 for ImageNet pretraining,
    # but U-Nets are flexible with input sizes.
    # For patch-based segmentation, ensure patch size is divisible by 32 (max downsampling factor).
    dummy_input = torch.randn(2, 3, 256, 256)
    num_classes = 4 # Example: background, class1, class2, class3

    # Instantiate the model
    model_convnext = UNetWithConvNeXtBase(n_classes=num_classes, pretrained=True)
    model_convnext.eval() # Set to evaluation mode for inference

    # Perform a forward pass
    with torch.no_grad(): # No need to compute gradients for a simple test
        output = model_convnext(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape) # Should be (B, n_classes, H_input, W_input)

    # --- Test the original VGG16 based model for comparison of structure ---
    # model_vgg = UNetWithVGG16BN(n_classes=num_classes, pretrained=True)
    # model_vgg.eval()
    # with torch.no_grad():
    # output_vgg = model_vgg(dummy_input)
    # print("VGG Input shape:", dummy_input.shape)
    # print("VGG Output shape:", output_vgg.shape)

    # Check parameters (optional)
    # total_params_convnext = sum(p.numel() for p in model_convnext.parameters() if p.requires_grad)
    # print(f"Total trainable parameters (ConvNeXt U-Net): {total_params_convnext:,}")
    # total_params_vgg = sum(p.numel() for p in model_vgg.parameters() if p.requires_grad)
    # print(f"Total trainable parameters (VGG16 U-Net): {total_params_vgg:,}")
