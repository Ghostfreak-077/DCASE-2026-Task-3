import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetSAISELD(nn.Module):
    def __init__(self, n_classes, in_channels=4, img_w=360, img_h=180):
        super().__init__()
        self.img_w = img_w
        self.img_h = img_h
        
        # Encoder Spatial Extraction
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck Latent Space
        self.bottleneck = ConvBlock(256, 512)
        
        # Decoder Spatial Reconstruction
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        
        # Multi-task Estimation Heads
        self.energy_head = nn.Conv2d(64, 1, kernel_size=1)
        self.spatial_head = nn.Conv2d(64, n_classes * 3, kernel_size=1) # Active track Cartesian coordinates

    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool1(s1)
        s2 = self.enc2(p1)
        p2 = self.pool2(s2)
        s3 = self.enc3(p2)
        p3 = self.pool3(s3)
        
        b = self.bottleneck(p3)
        
        d3 = self.up3(b)
        d3 = torch.cat((d3, s3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((d2, s2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((d1, s1), dim=1)
        d1 = self.dec1(d1)
        
        energy = torch.sigmoid(self.energy_head(d1))
        spatial = self.spatial_head(d1)
        return {"energy": energy, "spatial": spatial}