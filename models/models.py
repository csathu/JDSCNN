import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from lib.nn import SynchronizedBatchNorm2d
import math
from .attention_blocks import DualAttBlock
from .resnet import BasicBlock as ResBlock
from . import GSConv as gsc
import cv2
from .norm import Norm2d
from .operators import PSPModule
from mynn import Norm2d, Upsample
from .operators import conv_bn_relu, conv_sigmoid


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def intersectionAndUnion(self, imPred, imLab, numClass):
        imPred = np.asarray(imPred.cpu()).copy()
        imLab = np.asarray(imLab.cpu()).copy()

        imPred += 1
        imLab += 1
        imPred = imPred * (imLab > 0)

        intersection = imPred * (imPred == imLab)
        (area_intersection, _) = np.histogram(
            intersection, bins=numClass, range=(1, numClass))

        (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
        (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
        area_union = area_pred + area_lab - area_intersection

        jaccard = area_intersection / area_union
        jaccard = (jaccard[1] + jaccard[2]) / 2
        return jaccard if jaccard <= 1 else 0

    def pixel_acc(self, pred, label, num_class):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 1).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)

        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)

        jaccard = []

        for i in range(1, num_class):
            v = (label == i).long()
            pred = (preds == i).long()
            anb = torch.sum(v * pred)
            try:
                j = anb.float() / (torch.sum(v).float() + torch.sum(pred).float() - anb.float() + 1e-10)
            except:
                j = 0

            j = j if j <= 1 else 0
            jaccard.append(j)

        return acc, jaccard

    def jaccard(self, pred, label):
        AnB = torch.sum(pred.long() & label)
        return AnB / (pred.view(-1).sum().float() + label.view(-1).sum().float() - AnB)


class SegmentationModule(SegmentationModuleBase):
    def __init__(self, crit, unet, num_class):
        super(SegmentationModule, self).__init__()
        self.crit = crit
        self.unet = unet
        self.num_class = num_class

    def forward(self, feed_dict, epoch, *, segSize=None):
        if segSize is None:
            p = self.unet(feed_dict['image'])
            loss = self.crit(p, feed_dict['mask'], epoch=epoch)
            acc = self.pixel_acc(torch.round(nn.functional.softmax(p[0], dim=1)).long(),
                                 feed_dict['mask'][0].long().cuda(), self.num_class)
            return loss, acc

        if segSize == True:
            p = self.unet(feed_dict['image'])
            pred = nn.functional.softmax(p[0], dim=1)
            return pred

        else:
            p = self.unet(feed_dict['image'])
            loss = self.crit((p[0], p[1]),
                             (feed_dict['mask'][0].long().unsqueeze(0), feed_dict['mask'][1].unsqueeze(0)))
            pred = nn.functional.softmax(p[0], dim=1)
            return pred, loss


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def Conv3x3_Bn_Relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


def Conv1x1_Bn_Relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super(ConvRelu, self).__init__()
        self.en = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.en(x)
        x = self.activation(x)
        return x


class ModelBuilder():
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def build_unet(self, num_class=1, arch='albunet', weights=''):
        arch = arch.upper()
        if arch == 'JDSCNN':
            unet = JDSCNN(num_classes=num_class)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:
            unet.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
            print("Loaded pretrained UNet weights.")
        print('Loaded weights for unet')
        return unet


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, is_deconv=True):
        super(DecoderBlock, self).__init__()
        self.in_channels = in_channels

        if is_deconv:
            self.block = nn.Sequential(
                Conv3x3_Bn_Relu(in_channels, middle_channels),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv3x3_Bn_Relu(in_channels, middle_channels),
                Conv3x3_Bn_Relu(middle_channels, out_channels),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)


class SkipConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        return self.block(x)


class JDSCNN(nn.Module):  # JDSCNN
    def __init__(self, num_classes=4, num_filters=32, pretrained=True, is_deconv=True):
        super(JDSCNN, self).__init__()

        self.num_classes = num_classes
        self.MaxPool = nn.MaxPool2d(2, 2)
        self.encoder = torchvision.models.densenet121(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)
        self.Sigmoid = nn.Sigmoid()

        self.ge1 = conv_bn_relu(num_filters*4, num_filters*4, 1, norm_layer=nn.BatchNorm2d)
        self.ge2 = conv_bn_relu(num_filters*8, num_filters*4, 1, norm_layer=nn.BatchNorm2d)
        self.ge3 = conv_bn_relu(num_filters*16, num_filters*4, 1, norm_layer=nn.BatchNorm2d)
        self.gd1 = conv_bn_relu(num_filters*4, num_filters*4, 1, norm_layer=nn.BatchNorm2d)
        self.gd2 = conv_bn_relu(num_filters*4, num_filters*8, 1, norm_layer=nn.BatchNorm2d)
        self.gd3 = conv_bn_relu(num_filters*4, num_filters*16, 1, norm_layer=nn.BatchNorm2d)
        self.gg1 = conv_sigmoid(num_filters*4, num_filters*4)
        self.gg2 = conv_sigmoid(num_filters*8, num_filters*4)
        self.gg3 = conv_sigmoid(num_filters*16, num_filters*4)
        self.b3 = nn.Conv2d(num_filters*8, 1, kernel_size=1)
        self.b4 = nn.Conv2d(num_filters*16, 1, kernel_size=1)
        self.b5 = nn.Conv2d(num_filters*32, 1, kernel_size=1)
        self.sb5 = nn.Conv2d(8, num_filters*16, kernel_size=1)
        self.cb5 = nn.Conv2d(num_filters*32, num_filters*16, kernel_size=1)
        self.d0 = nn.Conv2d(num_filters*4, num_filters*2, kernel_size=1)
        self.res1 = ResBlock(num_filters*2, num_filters*2)
        self.d1 = nn.Conv2d(num_filters*2, num_filters, kernel_size=1)
        self.res2 = ResBlock(num_filters, num_filters)
        self.d2 = nn.Conv2d(num_filters, 16, kernel_size=1)
        self.res3 = ResBlock(16, 16)
        self.d3 = nn.Conv2d(16, 8, kernel_size=1)
        self.fuse = nn.Conv2d(8, 1, kernel_size=1, padding=0, bias=False)
        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.gate1 = gsc.GatedSpatialConv2d(num_filters, num_filters)
        self.gate2 = gsc.GatedSpatialConv2d(16, 16)
        self.gate3 = gsc.GatedSpatialConv2d(8, 8)
        self.expand = nn.Sequential(nn.Conv2d(1, num_filters, kernel_size=1),
                                    Norm2d(num_filters),
                                    nn.ReLU(inplace=True))
        self.en1 = nn.Sequential(self.encoder.features.conv0,
                                   self.encoder.features.norm0)
        self.en2 = self.encoder.features.denseblock1
        self.en2t = self.encoder.features.transition1
        self.en3 = self.encoder.features.denseblock2
        self.en3t = self.encoder.features.transition2
        self.en4 = self.encoder.features.denseblock3
        self.en4t = self.encoder.features.transition3
        self.en5 = nn.Sequential(self.encoder.features.denseblock4,
                                   self.encoder.features.norm5)
        self.center = PSPModule(features=num_filters*32, norm_layer=nn.BatchNorm2d, out_features=num_filters*16)
        norm_layer = nn.BatchNorm2d
        norm_layer = Norm2d
        self.de5 = DualAttBlock(inchannels=[num_filters*16, num_filters*32], outchannels=num_filters*16)
        self.de4 = DualAttBlock(inchannels=[num_filters*16, num_filters*16], outchannels=num_filters*8)
        self.de3 = DualAttBlock(inchannels=[num_filters*16 + num_filters*8, num_filters*8], outchannels=num_filters*4)
        self.de2 = DualAttBlock(inchannels=[num_filters*16 + num_filters*8 + num_filters*4, num_filters*4], outchannels=num_filters*2)
        self.de1 = DecoderBlock(num_filters*16 + num_filters*8 + num_filters*4 + num_filters*2, 48, num_filters, is_deconv)
        self.de0 = Conv3x3_Bn_Relu(num_filters * 2, num_filters)
        self.final = nn.Conv2d(num_filters, self.num_classes, kernel_size=1)

    def forward(self, x, return_att=False):
        x_size = x.size()

        en1 = self.en1(x)
        en2 = self.en2t(self.en2(en1))
        en3 = self.en3t(self.en3(en2))
        en4 = self.en4t(self.en4(en3))
        conv5 = self.en5(en4)

        ss = F.interpolate(self.d0(en2), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        b3 = F.interpolate(self.b3(en3), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss, g1 = self.gate1(ss, b3)
        ss = self.res2(ss)
        ss = self.d2(ss)
        b4 = F.interpolate(self.b4(en4), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss, g2 = self.gate2(ss, b4)
        ss = self.res3(ss)
        ss = self.d3(ss)
        b5 = F.interpolate(self.b5(conv5), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss, g3 = self.gate3(ss, b5)
        sb5 = F.interpolate(ss, size=[16, 16], mode='bilinear', align_corners=True)
        sb5 = self.sb5(sb5)
        ss = self.fuse(ss)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.Sigmoid(ss)

        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv2.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()

        cat = torch.cat([edge_out, canny], dim=1)
        acts = self.cw(cat)
        acts = self.Sigmoid(acts)
        edge = self.expand(acts)
        cb5 = self.cb5(conv5)
        conv5 = torch.cat([cb5, sb5], dim=1)

        en2 = F.interpolate(en2, scale_factor=2, mode='bilinear', align_corners=True)
        en3 = F.interpolate(en3, scale_factor=2, mode='bilinear', align_corners=True)
        en4 = F.interpolate(en4, scale_factor=2, mode='bilinear', align_corners=True)

        m2, m3, m4 = en2, en3, en4
        m2_size = m2.size()[2:]
        m3_size = m3.size()[2:]
        m4_size = m4.size()[2:]

        g_m2 = self.gg1(m2)
        g_m3 = self.gg2(m3)
        g_m4 = self.gg3(m4)

        m2 = self.ge1(m2)
        m3 = self.ge2(m3)
        m4 = self.ge3(m4)

        m2 = m2 + g_m2 * m2 + (1 - g_m2) * (Upsample(g_m3 * m3, size=m2_size) + Upsample(g_m4 * m4, size=m2_size))
        m3 = m3 + g_m3 * m3 + (1 - g_m3) * (
                Upsample(g_m2 * m2, size=m3_size) + Upsample(g_m4 * m4, size=m3_size))
        m4 = m4 + m4 * g_m4 + (1 - g_m4) * (
                Upsample(g_m3 * m3, size=m4_size) + Upsample(g_m2 * m2, size=m4_size))

        m2 = self.gd1(m2)
        m3 = self.gd2(m3)
        m4 = self.gd3(m4)
        center = self.center(self.MaxPool(conv5))
        de5, _ = self.de5([center, conv5])
        de4, _ = self.de4([de5, m4])

        de5_out = Upsample(de5, size=(int(de5.size()[2] * 2), int(de5.size()[3] * 2)))
        de3, _ = self.de3([torch.cat([de5_out, de4], dim=1), m3])
        de5_out = Upsample(de5_out, size=(int(de5_out.size()[2] * 2), int(de5_out.size()[3] * 2)))
        de4_out = Upsample(de4, size=(int(de4.size()[2] * 2), int(de4.size()[3] * 2)))
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        de2, _ = self.de2([torch.cat([de5_out, de4_out, de3], dim=1), m2])
        de5_out = Upsample(de5, size=(int(de5.size()[2] * 8), int(de5.size()[3] * 8)))
        de4_out = Upsample(de4, size=(int(de4.size()[2] * 4), int(de4.size()[3] * 4)))
        de3_out = Upsample(de3, size=(int(de3.size()[2] * 2), int(de3.size()[3] * 2)))
        de1 = self.de1(torch.cat([de5_out, de4_out, de3_out, de2], dim=1))  
        de0 = self.de0(torch.cat([de1, edge], dim=1))

        x_out = self.final(de0)

        return x_out, edge_out


