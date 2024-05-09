from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    # ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4):
        super().__init__()

        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel,
                             out_features=in_channel // ratio,
                             bias=False)

        # relu激活，可自行换别的激活函数
        self.relu = nn.ReLU()

        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio,
                             out_features=in_channel,
                             bias=False)

        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):  # inputs 代表输入特征图

        b, c, h, w = inputs.shape

        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)

        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)

        x = self.relu(x)

        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)

        # 对通道权重归一化处理
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs


class DoubleConv(nn.Sequential):

    def __init__(self, in_channels, out_channels):

        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      dilation=1),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      dilation=1),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels))
        self.ReLU = nn.ReLU()

        self.C3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                                nn.BatchNorm2d(out_channels), SE(out_channels))

    def forward(self, x):
        out = self.conv(x)
        out1 = self.C3(x)
        _x = self.ReLU(out1 + out)
        return _x


class DoubleConv1(nn.Sequential):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      dilation=1),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      padding=1,
                      bias=False,
                      dilation=1),
            # nn.GELU(),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Down(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.max = nn.MaxPool2d(2, stride=2)
        self._2conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        m = self.max(x)
        x = self._2conv(m)
        return x


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv1(in_channels, out_channels,
                                    in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [
            diff_x // 2, diff_x - diff_x // 2, diff_y // 2,
            diff_y - diff_y // 2
        ])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(OutConv,
              self).__init__(nn.Conv2d(in_channels, num_classes,
                                       kernel_size=1))


class UNet(nn.Module):

    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 32):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        up1 = self.up1(x5, x4)
        up2 = self.up2(up1, x3)
        up3 = self.up3(up2, x2)
        up4 = self.up4(up3, x1)

        logits = self.out_conv(up4)

        return logits


if __name__ == '__main__':
    model = UNet(in_channels=3).cuda()
    x = torch.rand(1, 3, 256, 256).cuda()
    out = model(x)['out']
    print(out.shape)
