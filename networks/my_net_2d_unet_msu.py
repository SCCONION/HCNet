
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, normalization='none'):
        super().__init__()
        ops = []

        ops.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm2d(out_channels))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=out_channels))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm2d(out_channels))
        ops.append(nn.ReLU(inplace=True))

        ops.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm2d(out_channels))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=out_channels))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm2d(out_channels))
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class MultiScaleFilterBank(nn.Module):
    """
    """
    def __init__(self, in_channels, use_bn=True, use_relu=True):
        super().__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu

        # 1x1
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=not use_bn)

        # 3x3
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=not use_bn)

        # 5x5
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, bias=not use_bn)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn3 = nn.BatchNorm2d(in_channels)
            self.bn5 = nn.BatchNorm2d(in_channels)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        f1 = self.conv1x1(x)
        if self.use_bn:
            f1 = self.bn1(f1)
        if self.use_relu:
            f1 = self.relu(f1)

        # 3x3
        f3 = self.conv3x3(x)
        if self.use_bn:
            f3 = self.bn3(f3)
        if self.use_relu:
            f3 = self.relu(f3)

        # 5x5
        f5 = self.conv5x5(x)
        if self.use_bn:
            f5 = self.bn5(f5)
        if self.use_relu:
            f5 = self.relu(f5)

        return f1, f3, f5


class MultiScaleFeedbackModule(nn.Module):
    """
    """
    def __init__(self, channels_list, use_bn=True, use_relu=False):

        super().__init__()

        self.filter_banks_1 = nn.ModuleList([
            MultiScaleFilterBank(ch, use_bn=use_bn, use_relu=use_relu)
            for ch in channels_list
        ])
        self.filter_banks_2 = nn.ModuleList([
            MultiScaleFilterBank(ch, use_bn=use_bn, use_relu=use_relu)
            for ch in channels_list
        ])

        self.fusion_weights = nn.ParameterList([
            nn.Parameter(torch.ones(3) / 3)
            for _ in channels_list
        ])

    def forward(self, stage_feat1, stage_feat2, use_learned_weights=False):

        feedback_list = []

        for idx, (feat1, feat2) in enumerate(zip(stage_feat1, stage_feat2)):
            # decoder1
            d1_f1, d1_f3, d1_f5 = self.filter_banks_1[idx](feat1)

            # decoder2
            d2_f1, d2_f3, d2_f5 = self.filter_banks_2[idx](feat2)

            diff_1 = d1_f1 - d2_f1
            diff_3 = d1_f3 - d2_f3
            diff_5 = d1_f5 - d2_f5

            if use_learned_weights:

                weights = F.softmax(self.fusion_weights[idx], dim=0)
                feedback = weights[0] * diff_1 + weights[1] * diff_3 + weights[2] * diff_5
            else:

                feedback = diff_1 + diff_3 + diff_5

            feedback_list.append(feedback)

        return feedback_list


class Encoder(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False):
        super().__init__()
        self.has_dropout = has_dropout


        self.conv1 = DoubleConv(n_channels, n_filters, normalization)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(n_filters, n_filters * 2, normalization)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(n_filters * 2, n_filters * 4, normalization)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(n_filters * 4, n_filters * 8, normalization)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5 = DoubleConv(n_filters * 8, n_filters * 16, normalization)  # bottleneck

        self.dropout = nn.Dropout2d(p=0.3, inplace=False)

    def forward(self, input, en=None):

        if en is None:
            en = []

        x1 = self.conv1(input)
        if len(en) != 0:
            x1 = x1 + en[4]
        x1_pool = self.pool1(x1)

        x2 = self.conv2(x1_pool)
        if len(en) != 0:
            x2 = x2 + en[3]
        x2_pool = self.pool2(x2)

        x3 = self.conv3(x2_pool)
        if len(en) != 0:
            x3 = x3 + en[2]
        x3_pool = self.pool3(x3)

        x4 = self.conv4(x3_pool)
        if len(en) != 0:
            x4 = x4 + en[1]
        x4_pool = self.pool4(x4)

        x5 = self.conv5(x4_pool)
        if len(en) != 0:
            x5 = x5 + en[0]

        if self.has_dropout:
            x5 = self.dropout(x5)

        return [x1, x2, x3, x4, x5]


class UpBlock(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels, normalization='none', up_type=0):
        super().__init__()


        if up_type == 0:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        elif up_type == 1:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )

        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels, normalization)

    def forward(self, x, skip):
        x = self.up(x)

        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class Decoder(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False, has_residual=False, up_type=0):
        super().__init__()
        self.has_dropout = has_dropout

        self.up1 = UpBlock(n_filters * 16, n_filters * 8, n_filters * 8, normalization, up_type)
        self.up2 = UpBlock(n_filters * 8, n_filters * 4, n_filters * 4, normalization, up_type)
        self.up3 = UpBlock(n_filters * 4, n_filters * 2, n_filters * 2, normalization, up_type)
        self.up4 = UpBlock(n_filters * 2, n_filters, n_filters, normalization, up_type)

        self.out_conv = nn.Conv2d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout2d(p=0.5, inplace=False)

    def forward(self, features, f1='none', f2='none'):
        x1, x2, x3, x4, x5 = features

        if f1 == 'none' and f2 == 'none':
            d4 = self.up1(x5, x4)
            d3 = self.up2(d4, x3)
            d2 = self.up3(d3, x2)
            d1 = self.up4(d2, x1)

            if self.has_dropout:
                d1 = self.dropout(d1)
            out_seg = self.out_conv(d1)

        else:
            if f2 != 'none':
                m5, m4, m3, m2, m1 = f1
                m5_, m4_, m3_, m2_, m1_ = f2
                w5 = torch.sigmoid(m5)
                w4 = torch.sigmoid(m4)
                w3 = torch.sigmoid(m3)
                w2 = torch.sigmoid(m2)
                w1 = torch.sigmoid(m1)
                w5_ = torch.sigmoid(m5_)
                w4_ = torch.sigmoid(m4_)
                w3_ = torch.sigmoid(m3_)
                w2_ = torch.sigmoid(m2_)
                w1_ = torch.sigmoid(m1_)

                x5_w = x5 + 0.5 * (x5 * w5 + x5 * w5_)
                x4_w = x4 + 0.5 * (x4 * w4 + x4 * w4_)
                x3_w = x3 + 0.5 * (x3 * w3 + x3 * w3_)
                x2_w = x2 + 0.5 * (x2 * w2 + x2 * w2_)
                x1_w = x1 + 0.5 * (x1 * w1 + x1 * w1_)

                d4 = self.up1(x5_w, x4_w)
                d3 = self.up2(d4, x3_w)
                d2 = self.up3(d3, x2_w)
                d1 = self.up4(d2, x1_w)

                if self.has_dropout:
                    d1 = self.dropout(d1)
                out_seg = self.out_conv(d1)

            else:
                m5, m4, m3, m2, m1 = f1
                w5 = torch.sigmoid(m5).detach()
                w4 = torch.sigmoid(m4).detach()
                w3 = torch.sigmoid(m3).detach()
                w2 = torch.sigmoid(m2).detach()
                w1 = torch.sigmoid(m1).detach()

                x5_w = x5 + x5 * w5
                x4_w = x4 + x4 * w4
                x3_w = x3 + x3 * w3
                x2_w = x2 + x2 * w2
                x1_w = x1 + x1 * w1

                d4 = self.up1(x5_w, x4_w)
                d3 = self.up2(d4, x3_w)
                d2 = self.up3(d3, x2_w)
                d1 = self.up4(d2, x1_w)

                if self.has_dropout:
                    d1 = self.dropout(d1)
                out_seg = self.out_conv(d1)

        return out_seg, [x5, d4, d3, d2, d1]


class SideConv(nn.Module):

    def __init__(self, n_classes=2, n_filters=16):
        super().__init__()
        self.side5 = nn.Conv2d(n_filters * 16, n_classes, 1, padding=0)
        self.side4 = nn.Conv2d(n_filters * 8, n_classes, 1, padding=0)
        self.side3 = nn.Conv2d(n_filters * 4, n_classes, 1, padding=0)
        self.side2 = nn.Conv2d(n_filters * 2, n_classes, 1, padding=0)
        self.side1 = nn.Conv2d(n_filters, n_classes, 1, padding=0)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, stage_feat):
        x5, d4, d3, d2, d1 = stage_feat

        out5 = self.side5(x5)
        out5 = self.upsamplex2(self.upsamplex2(self.upsamplex2(self.upsamplex2(out5))))

        out4 = self.side4(d4)
        out4 = self.upsamplex2(self.upsamplex2(self.upsamplex2(out4)))

        out3 = self.side3(d3)
        out3 = self.upsamplex2(self.upsamplex2(out3))

        out2 = self.side2(d2)
        out2 = self.upsamplex2(out2)

        out1 = self.side1(d1)

        return [out5, out4, out3, out2, out1]


class HCNet_Net_2D_MultiScale(nn.Module):

    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none',
                 has_dropout=False, has_residual=False,
                 use_multiscale_feedback=True, feedback_use_bn=True, feedback_use_relu=False,
                 use_learned_weights=False):

        super().__init__()
        self.use_multiscale_feedback = use_multiscale_feedback
        self.use_learned_weights = use_learned_weights

        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, up_type=1)
        self.sideconv1 = SideConv(n_classes=n_classes, n_filters=n_filters)


        if use_multiscale_feedback:
            channels_list = [n_filters * 16, n_filters * 8, n_filters * 4, n_filters * 2, n_filters]
            self.multiscale_feedback = MultiScaleFeedbackModule(
                channels_list=channels_list,
                use_bn=feedback_use_bn,
                use_relu=feedback_use_relu
            )

    def forward(self, input, en=None):

        features = self.encoder(input, en=en)
        out_seg1, stage_feat1 = self.decoder1(features)
        out_seg2, stage_feat2 = self.decoder2(features, stage_feat1)
        deep_out1 = self.sideconv1(stage_feat1)
        return out_seg1, out_seg2, [stage_feat2, stage_feat1], deep_out1, []

    def compute_feedback(self, stage_feat1, stage_feat2, lam=1.0):

        if self.use_multiscale_feedback:

            feedback_list = self.multiscale_feedback(
                stage_feat1, stage_feat2,
                use_learned_weights=self.use_learned_weights
            )

            feedback_list = [lam * fb.detach() for fb in feedback_list]
        else:

            feedback_list = []
            for feat1, feat2 in zip(stage_feat1, stage_feat2):
                feedback_list.append(lam * (feat1 - feat2).detach())

        return feedback_list



HCNet_Net_2D = HCNet_Net_2D_MultiScale
