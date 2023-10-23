import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False, padding_mode='replicate')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class FrontFusion(nn.Module):
    def __init__(self, channels):
        super(FrontFusion, self).__init__()
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.relu(self.bn(self.conv(x)))
        return x

class FinalFusion(nn.Module):
    def __init__(self, channels):
        super(FinalFusion, self).__init__()
        self.conv = nn.Conv2d(channels * 3, channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.relu(self.bn(self.conv(x)))
        return x

class UP(nn.Module):
    def __init__(self, channels):
        super(UP, self).__init__()
        self.conv = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.relu(x)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SegNet(nn.Module):
    def __init__(self,input_nbr=3,label_nbr=22):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11_rgb = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.conv11_depth = nn.Conv2d(1, 64, kernel_size=3, padding=1)

        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)

        self.SA = SpatialAttention()
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.up1 = UP(512)
        self.up2 = UP(256)
        self.up3 = UP(128)
        self.up4 = UP(64)
        self.front_fusion = FrontFusion(64)
        self.final_fusion = FinalFusion(64)
        self.se_layer = SELayer(64)

    def forward(self, rgb, depth):
        x11 = F.relu(self.bn11(self.conv11_rgb(rgb)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1p, id1 = F.max_pool2d(x12,kernel_size=2, stride=2,return_indices=True)
        x1p_sa = self.SA(x1p)
        x1p_sa_pool = self.maxpool(x1p_sa)

        x11_depth = F.relu(self.bn11(self.conv11_depth(depth)))
        x12_depth = F.relu(self.bn12(self.conv12(x11_depth)))
        x1p_depth, id1_depth = F.max_pool2d(x12_depth, kernel_size=2, stride=2,return_indices=True)
        x1p_depth_sa = self.SA(x1p_depth)
        x1p_depth_sa_pool = self.maxpool(x1p_depth_sa)

        x1p = F.relu(x1p * x1p_depth_sa)
        x1p_depth = F.relu(x1p_depth * x1p_sa)
        x1p_fusion = self.front_fusion(x1p, x1p_depth)
        x1p_fusion_sa = self.SA(x1p_fusion)
        x1p_fusion_sa_pool = self.maxpool(x1p_fusion_sa)

        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2p, id2 = F.max_pool2d(x22,kernel_size=2, stride=2,return_indices=True)
        x2p_sa = self.SA(x2p)
        x2p_sa_pool = self.maxpool(x2p_sa)

        x21_depth = F.relu(self.bn21(self.conv21(x1p_depth)))
        x22_depth = F.relu(self.bn22(self.conv22(x21_depth)))
        x2p_depth, id2_depth = F.max_pool2d(x22_depth,kernel_size=2, stride=2,return_indices=True)
        x2p_depth_sa = self.SA(x2p_depth)
        x2p_depth_sa_pool = self.maxpool(x2p_depth_sa)

        x21_fusion = F.relu(self.bn21(self.conv21(x1p_fusion)))
        x22_fusion = F.relu(self.bn22(self.conv22(x21_fusion)))
        x2p_fusion, id2_fusion = F.max_pool2d(x22_fusion,kernel_size=2, stride=2,return_indices=True)
        x2p_fusion_sa = self.SA(x2p_fusion)
        x2p_fusion_sa_pool = self.maxpool(x2p_fusion_sa)

        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x33 = F.relu(x1p_sa_pool * x33)
        x3p, id3 = F.max_pool2d(x33,kernel_size=2, stride=2,return_indices=True)

        x31_depth = F.relu(self.bn31(self.conv31(x2p_depth)))
        x32_depth = F.relu(self.bn32(self.conv32(x31_depth)))
        x33_depth = F.relu(self.bn33(self.conv33(x32_depth)))
        x33_depth = F.relu(x1p_depth_sa_pool * x33_depth)
        x3p_depth, id3_depth = F.max_pool2d(x33_depth,kernel_size=2, stride=2,return_indices=True)

        x31_fusion = F.relu(self.bn31(self.conv31(x2p_fusion)))
        x32_fusion = F.relu(self.bn32(self.conv32(x31_fusion)))
        x33_fusion = F.relu(self.bn33(self.conv33(x32_fusion)))
        x33_fusion = F.relu(x1p_fusion_sa_pool * x33_fusion)
        x3p_fusion, id3_fusion = F.max_pool2d(x33_fusion,kernel_size=2, stride=2,return_indices=True)

        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x43 = F.relu(x2p_sa_pool * x43)
        x4p, id4 = F.max_pool2d(x43,kernel_size=2, stride=2,return_indices=True)

        x41_depth = F.relu(self.bn41(self.conv41(x3p_depth)))
        x42_depth = F.relu(self.bn42(self.conv42(x41_depth)))
        x43_depth = F.relu(self.bn43(self.conv43(x42_depth)))
        x43_depth = F.relu(x2p_depth_sa_pool * x43_depth)
        x4p_depth, id4_depth = F.max_pool2d(x43_depth, kernel_size=2, stride=2, return_indices=True)

        x41_fusion = F.relu(self.bn41(self.conv41(x3p_fusion)))
        x42_fusion = F.relu(self.bn42(self.conv42(x41_fusion)))
        x43_fusion = F.relu(self.bn43(self.conv43(x42_fusion)))
        x43_fusion = F.relu(x2p_fusion_sa_pool * x43_fusion)
        x4p_fusion, id4_fusion = F.max_pool2d(x43_fusion, kernel_size=2, stride=2, return_indices=True)

        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        x51_depth = F.relu(self.bn51(self.conv51(x4p_depth)))
        x52_depth = F.relu(self.bn52(self.conv52(x51_depth)))
        x53_depth = F.relu(self.bn53(self.conv53(x52_depth)))
        x5p_depth, id5_depth = F.max_pool2d(x53_depth, kernel_size=2, stride=2, return_indices=True)

        x51_fusion = F.relu(self.bn51(self.conv51(x4p_fusion)))
        x52_fusion = F.relu(self.bn52(self.conv52(x51_fusion)))
        x53_fusion = F.relu(self.bn53(self.conv53(x52_fusion)))
        x5p_fusion, id5_fusion = F.max_pool2d(x53_fusion, kernel_size=2, stride=2, return_indices=True)

        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))
        x51d = self.up1(x51d, x4p)

        x5d_depth = F.max_unpool2d(x5p_depth, id5_depth, kernel_size=2, stride=2)
        x53d_depth = F.relu(self.bn53d(self.conv53d(x5d_depth)))
        x52d_depth = F.relu(self.bn52d(self.conv52d(x53d_depth)))
        x51d_depth = F.relu(self.bn51(self.conv51d(x52d_depth)))
        x51d_depth = self.up1(x51d_depth, x4p_depth)

        x5d_fusion = F.max_unpool2d(x5p_fusion, id5_fusion, kernel_size=2, stride=2)
        x53d_fusion = F.relu(self.bn53d(self.conv53d(x5d_fusion)))
        x52d_fusion = F.relu(self.bn52d(self.conv52d(x53d_fusion)))
        x51d_fusion = F.relu(self.bn51d(self.conv51d(x52d_fusion)))
        x51d_fusion = self.up1(x51d_fusion, x4p_fusion)

        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))
        x41d = self.up2(x41d, x3p)

        x4d_depth = F.max_unpool2d(x51d_depth, id4_depth, kernel_size=2, stride=2)
        x43d_depth = F.relu(self.bn43d(self.conv43d(x4d_depth)))
        x42d_depth = F.relu(self.bn42d(self.conv42d(x43d_depth)))
        x41d_depth = F.relu(self.bn41d(self.conv41d(x42d_depth)))
        x41d_depth = self.up2(x41d_depth, x3p_depth)

        x4d_fusion = F.max_unpool2d(x51d_fusion, id4_fusion, kernel_size=2, stride=2)
        x43d_fusion = F.relu(self.bn43d(self.conv43d(x4d_fusion)))
        x42d_fusion = F.relu(self.bn42d(self.conv42d(x43d_fusion)))
        x41d_fusion = F.relu(self.bn41d(self.conv41d(x42d_fusion)))
        x41d_fusion = self.up2(x41d_fusion, x3p_fusion)

        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))
        x31d = self.up3(x31d, x2p)

        x3d_depth = F.max_unpool2d(x41d_depth, id3_depth, kernel_size=2, stride=2)
        x33d_depth = F.relu(self.bn33d(self.conv33d(x3d_depth)))
        x32d_depth = F.relu(self.bn32d(self.conv32d(x33d_depth)))
        x31d_depth = F.relu(self.bn31d(self.conv31d(x32d_depth)))
        x31d_depth = self.up3(x31d_depth, x2p_depth)

        x3d_fusion = F.max_unpool2d(x41d_fusion, id3_fusion, kernel_size=2, stride=2)
        x33d_fusion = F.relu(self.bn33d(self.conv33d(x3d_fusion)))
        x32d_fusion = F.relu(self.bn32d(self.conv32d(x33d_fusion)))
        x31d_fusion = F.relu(self.bn31d(self.conv31d(x32d_fusion)))
        x31d_fusion = self.up3(x31d_fusion, x2p_fusion)

        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))
        x21d = self.up4(x21d, x1p)

        x2d_depth = F.max_unpool2d(x31d_depth, id2_depth, kernel_size=2, stride=2)
        x22d_depth = F.relu(self.bn22d(self.conv22d(x2d_depth)))
        x21d_depth = F.relu(self.bn21d(self.conv21d(x22d_depth)))
        x21d_depth = self.up4(x21d_depth, x1p_depth)

        x2d_fusion = F.max_unpool2d(x31d_fusion, id2_fusion, kernel_size=2, stride=2)
        x22d_fusion = F.relu(self.bn22d(self.conv22d(x2d_fusion)))
        x21d_fusion = F.relu(self.bn21d(self.conv21d(x22d_fusion)))
        x21d_fusion = self.up4(x21d_fusion, x1p_fusion)

        x21d = self.final_fusion(x21d, x21d_depth, x21d_fusion)
        x21d = self.se_layer(x21d)

        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d
