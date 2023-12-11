import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
                nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
            )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AttBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample=None):
        super(AttBlock, self).__init__()
        self.ca = ChannelAttention(in_size)
        self.sa = SpatialAttention()

        if in_size != out_size:
            downsample = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.downsample = downsample

    def forward(self, inputs):
        outputs = self.ca(inputs) * inputs
        outputs = self.sa(outputs) * outputs

        if self.downsample is not None:
            outputs = self.downsample(outputs)

        return outputs


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, k=3, s=1, p=1, downsample=None):
        super(ResBlock, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=k, stride=s, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=k, stride=s, padding=p),
        )

        if in_size != out_size:
            downsample = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        res = inputs
        outputs = self.convs(inputs)

        if self.downsample is not None:
            res = self.downsample(inputs)

        outputs += res
        outputs = self.relu(outputs)
        return outputs


class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, layer_num):
        super(ConvBlock, self).__init__()
        self.layerNum = layer_num
        res_convs = []
        for i in range(1, layer_num + 1):
            if i == 1:
                res_convs.append(ResBlock(in_size, out_size))
            else:
                res_convs.append(ResBlock(out_size, out_size))
        self.res_convs = nn.Sequential(*res_convs)

    def forward(self, inputs):
        outputs = self.res_convs(inputs)
        return outputs


class DownSamBlock(nn.Module):
    def __init__(self, in_size, out_size, layer_num):
        super(DownSamBlock, self).__init__()
        scale_factor = layer_num
        self.shallowDownSam = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)
        self.DownSam = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Att = AttBlock(in_size, out_size)
        self.conv = ConvBlock(out_size, out_size, layer_num)

    def forward(self, shallow_fs, inputs):
        shallow_fs = self.shallowDownSam(shallow_fs)
        outputs = self.DownSam(inputs)
        outputs = torch.cat([shallow_fs, outputs], dim=1)
        outputs = self.Att(outputs)
        outputs = self.conv(outputs)
        return outputs


class UpSamBlock(nn.Module):
    def __init__(self, in_size, out_size, layer_num):
        super(UpSamBlock, self).__init__()
        self.up = nn.ConvTranspose2d(out_size, out_size, kernel_size=2, stride=2)

        self.Att = AttBlock(in_size, out_size)
        self.conv = ConvBlock(out_size, out_size, layer_num)

    def forward(self, residual, inputs):
        outputs = self.up(inputs)
        outputs = torch.cat([residual, outputs], dim=1)
        outputs = self.Att(outputs)
        outputs = self.conv(outputs)
        return outputs


class RecovBlock(nn.Module):
    def __init__(self, in_size, out_size, scale_factor, up=None):
        super(RecovBlock, self).__init__()
        if scale_factor != 1:
            up = nn.ConvTranspose2d(in_size, in_size, kernel_size=scale_factor, stride=scale_factor)
        self.up = up

        self.recov_conv = nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        if self.up is not None:
            inputs = self.up(inputs)

        outputs = self.recov_conv(inputs)
        return outputs


class FusRecovBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample=None):
        super(FusRecovBlock, self).__init__()
        self.ca = ChannelAttention(in_size, 2)
        self.sa = SpatialAttention()

        self.convs = nn.Sequential(
            nn.Conv2d(in_size, in_size, kernel_size=3, stride=1, padding=3 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size, in_size, kernel_size=5, stride=1, padding=5 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size, out_size, kernel_size=7, stride=1, padding=7 // 2),
        )

        if in_size != out_size:
            downsample = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.downsample = downsample

    def forward(self, inputs):
        outputs = self.ca(inputs) * inputs
        outputs = self.sa(outputs) * outputs

        if self.downsample is not None:
            res = self.downsample(outputs)

        outputs = self.convs(outputs)
        outputs += res
        return outputs


class ShallowFsBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample=None):
        super(ShallowFsBlock, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=7, stride=1, padding=7 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=5, stride=1, padding=5 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=3 // 2),
        )

        if in_size != out_size:
            downsample = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.downsample = downsample

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        res = inputs
        outputs = self.convs(inputs)

        if self.downsample is not None:
            res = self.downsample(inputs)

        outputs += res
        outputs = self.relu(outputs)
        return outputs


class IntConvBlock(nn.Module):
    def __init__(self, in_size, out_size, layer_num):
        super(IntConvBlock, self).__init__()
        self.Att = AttBlock(in_size, out_size)
        self.conv = ConvBlock(out_size, out_size, layer_num)

    def forward(self, orgF, inputs):
        inputs = torch.cat([orgF, inputs], dim=1)
        outputs = self.Att(inputs)
        outputs = self.conv(outputs)
        return outputs


class RAAUNet(nn.Module):
    def __init__(self, in_channels=1, filter_num=64):
        super(RAAUNet, self).__init__()
        scale_factor = [1, 2, 4, 8, 16]

        self.shallowConv = ShallowFsBlock(in_channels, filter_num)


        self.intConv = IntConvBlock(filter_num + in_channels, filter_num, scale_factor[0])


        self.DownBlock1 = DownSamBlock(filter_num * 2, filter_num, scale_factor[1])

        self.DownBlock2 = DownSamBlock(filter_num * 2, filter_num, scale_factor[2])

        self.DownBlock3 = DownSamBlock(filter_num * 2, filter_num, scale_factor[3])

        self.DownBlock4 = DownSamBlock(filter_num * 2, filter_num, scale_factor[4])


        self.UpConcatBlock4 = UpSamBlock(filter_num * 2, filter_num, scale_factor[3])

        self.UpConcatBlock3 = UpSamBlock(filter_num * 2, filter_num, scale_factor[2])

        self.UpConcatBlock2 = UpSamBlock(filter_num * 2, filter_num, scale_factor[1])

        self.UpConcatBlock1 = UpSamBlock(filter_num * 2, filter_num, scale_factor[0])


        self.Recov5 = RecovBlock(filter_num, 1, scale_factor[4])

        self.Recov4 = RecovBlock(filter_num, 1, scale_factor[3])

        self.Recov3 = RecovBlock(filter_num, 1, scale_factor[2])

        self.Recov2 = RecovBlock(filter_num, 1, scale_factor[1])

        self.Recov1 = RecovBlock(filter_num, 1, scale_factor[0])


        self.fusionRecov = FusRecovBlock(in_channels * 5, in_channels)

    def forward(self, inputs):
        residual = inputs

        shallowFs = self.shallowConv(inputs)

        res0 = self.intConv(inputs, shallowFs)
        res1 = self.DownBlock1(shallowFs, res0)
        res2 = self.DownBlock2(shallowFs, res1)
        res3 = self.DownBlock3(shallowFs, res2)
        res4 = self.DownBlock4(shallowFs, res3)

        up4 = self.UpConcatBlock4(res3, res4)
        up3 = self.UpConcatBlock3(res2, up4)
        up2 = self.UpConcatBlock2(res1, up3)
        up1 = self.UpConcatBlock1(res0, up2)

        recov = self.Recov5(res4)
        recov = torch.cat([self.Recov4(up4), recov], dim=1)
        recov = torch.cat([self.Recov3(up3), recov], dim=1)
        recov = torch.cat([self.Recov2(up2), recov], dim=1)
        recov = torch.cat([self.Recov1(up1), recov], dim=1)

        outputs = self.fusionRecov(recov)

        outputs += residual
        return outputs
