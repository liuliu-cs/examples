import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


__all__ = ['Inception3_DS_half', 'inception_v3_ds_half']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}


def inception_v3_ds_half(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError
    return Inception3_DS_half(**kwargs)


class Inception3_DS_half(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=False, transform_input=False):
        super(Inception3_DS_half, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 16, kernel_size=3, stride=1)
        self.Conv2d_2a_3x3 = BasicConv2d(16, 16, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(16, 32, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(32, 40, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(40, 96, kernel_size=3)
        self.Mixed_5b = InceptionA(96, pool_features=16)
        self.Mixed_5c = InceptionA(128, pool_features=32)
        self.Mixed_5d = InceptionA(144, pool_features=32)
        self.Mixed_6a = InceptionB(144)
        self.Mixed_6b = InceptionC(384, channels_7x7=64)
        self.Mixed_6c = InceptionC(384, channels_7x7=80)
        self.Mixed_6d = InceptionC(384, channels_7x7=80)
        self.Mixed_6e = InceptionC(384, channels_7x7=96)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(384)
        self.Mixed_7b = InceptionE(640)
        self.Mixed_7c = InceptionE(1024)
        self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                X = stats.truncnorm(-2, 2, scale=stddev)
                values = torch.Tensor(X.rvs(m.weight.data.numel()))
                values = values.view(m.weight.data.size())
                m.weight.data.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3  16x16x3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32 14x14x32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32 12x12x32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64 12x12x64
        # x = F.max_pool2d(x, kernel_size=3, stride=1)
        # 73 x 73 x 64   12x12x64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80   12x12x80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192  10x10x192
        # x = F.max_pool2d(x, kernel_size=3, stride=1)
        # 35 x 35 x 192  10x10x192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256  10x10x256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288  10x10x288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288  10x10x288
        x = self.Mixed_6a(x)
        # 17 x 17 x 768  4x4x768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768  4x4x768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768  4x4x768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768  4x4x768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768  4x4x768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)
        # 17 x 17 x 768  4x4x768
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280   4x4x1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048   4x4x2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048   4x4x2048
        x = F.avg_pool2d(x, kernel_size=2)
        # 1 x 1 x 2048   1x1x2048
        x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
        if self.training and self.aux_logits:
            return x, aux
        return x



class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 32, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 24, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(24, 32, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(48, 48, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels):
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 192, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(32, 48, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(48, 48, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        out_channels = 96
        self.branch1x1 = BasicConv2d(in_channels, out_channels, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, out_channels, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, out_channels, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):
# modified, set stride to 1
    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 96, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(96, 160, kernel_size=3, stride=1)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 96, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(96, 96, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(96, 96, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(96, 96, kernel_size=3, stride=1)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 160, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(192, 192, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(192, 192, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 224, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(224, 192, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(192, 192, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(192, 192, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 96, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        # self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1 = BasicConv2d(128, 768, kernel_size=4)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        # x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768  4x4x768
        x = self.conv0(x)
        # 5 x 5 x 128  4x4x128
        x = self.conv1(x)
        # 1 x 1 x 768  1x1x768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
