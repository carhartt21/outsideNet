import torch
import torch.nn as nn
import torchvision
import os
import logging
from . import resnet

BatchNorm2d = torch.nn.BatchNorm2d

logger = logging.getLogger(__name__)

class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        if type(pred) is list:
            pred = pred[-1]
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc


class SegmentationModule(SegmentationModuleBase): 
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def __init__(self, crit, num_class=24, weights_resNet='', weights='', spatial_mask=False,
                 use_softmax=True):
        super(SegmentationModule, self).__init__()
        pretrained = True if len(weights) == 0 else False
        orig_resnet = resnet.__dict__['resnet50'](pretrained=False)
        self.backbone = Resnet(orig_resnet)
        self.mlp = multi_level_pooling()
        self.ff = feature_fusion(num_class=num_class, spatial_mask=spatial_mask, use_softmax=use_softmax)
        
        net_encoder.apply(ModelBuilder.weights_init)
        # encoders are usually pretrained
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            logger.info('Loading weights for backbone')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)

        self.crit = crit

    def forward(self, feed_dict, *, label_size=None):
        # training
        if type(feed_dict) is list:
            feed_dict = feed_dict[0]
            # convert to torch.cuda.FloatTensor
            if torch.cuda.is_available():
                feed_dict['img_data'] = feed_dict['img_data'].cuda()
                feed_dict['label_data'] = feed_dict['label_data'].cuda()
            else:
                raise RuntimeError('Cannot convert torch.Floattensor into torch.cuda.FloatTensor')
        if self.training:
            pred = self.decoder(self.encoder(feed_dict['img_data'], return_feature_maps=True), label_size=label_size)
            loss = self.crit(pred, feed_dict['label_data'])
            acc = self.pixel_acc(pred, feed_dict['label_data'])
            return loss, acc
        # inference
        else:
            pred = self.ff(
                self.mlp(
                    self.encoder(feed_dict['img_data'], return_feature_maps=True), label_size=label_size))
            return pred


class ModelBuilder:
   

    @staticmethod
    def build_encoder(arch='resnet50dilated', fc_dim=512, weights='', spatial_mask=False, align_corners=False):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'resnet50':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=False)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet50dilated':
            orig_resnet = resnet.__dict__['resnet50'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        elif arch == 'resnet101':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = Resnet(orig_resnet)
        elif arch == 'resnet101dilated':
            orig_resnet = resnet.__dict__['resnet101'](pretrained=pretrained)
            net_encoder = ResnetDilated(orig_resnet, dilate_scale=8)
        else:
            raise Exception('Architecture undefined!')

        net_encoder.apply(ModelBuilder.weights_init)
        # encoders are usually pretrained
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='outsideNet',
                      fc_dim=512, num_class=150,
                      weights='', use_softmax=False,
                      align_corners=False):
        arch = arch.lower()
        if arch == 'c1':
            net_decoder = C1(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax)
        elif arch == 'outsideNet':
            net_decoder = outsideNet(
                num_class=num_class,
                fc_dim=fc_dim,
                use_softmax=use_softmax,
                fpn_dim=512)
        else:
            raise Exception('Architecture undefined!')
        if arch != 'ocr':
            net_decoder.apply(ModelBuilder.weights_init)
            if len(weights) > 0:
                logger.info('Loading weights for net_decoder')
                net_decoder.load_state_dict(
                    torch.load(weights, map_location=lambda storage, loc: storage), strict=False)

        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1, bias=False):
    # 3x3 convolution + BN + ReLu
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3,
                  stride=stride, padding=1, bias=bias),
        BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class Resnet(nn.Module):
    def __init__(self, orig_resnet):
        super(Resnet, self).__init__()

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


class ResnetDilated(nn.Module):
    def __init__(self, orig_resnet, dilate_scale=8):
        super(ResnetDilated, self).__init__()
        from functools import partial

        if dilate_scale == 8:
            orig_resnet.layer3.apply(
                partial(self._nostride_dilate, dilate=2))
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            orig_resnet.layer4.apply(
                partial(self._nostride_dilate, dilate=2))

        # take pretrained resnet, except AvgPool and FC
        self.conv1 = orig_resnet.conv1
        self.bn1 = orig_resnet.bn1
        self.relu1 = orig_resnet.relu1
        self.conv2 = orig_resnet.conv2
        self.bn2 = orig_resnet.bn2
        self.relu2 = orig_resnet.relu2
        self.conv3 = orig_resnet.conv3
        self.bn3 = orig_resnet.bn3
        self.relu3 = orig_resnet.relu3
        self.maxpool = orig_resnet.maxpool
        self.layer1 = orig_resnet.layer1
        self.layer2 = orig_resnet.layer2
        self.layer3 = orig_resnet.layer3
        self.layer4 = orig_resnet.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, return_feature_maps=False):
        conv_out = []

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        conv_out.append(x)
        x = self.layer2(x)
        conv_out.append(x)
        x = self.layer3(x)
        conv_out.append(x)
        x = self.layer4(x)
        conv_out.append(x)

        if return_feature_maps:
            return conv_out
        return [x]


# last conv
class C1(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048, use_softmax=False):
        super(C1, self).__init__()
        self.use_softmax = use_softmax

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out, label_size=None):
        conv5 = conv_out[-1]
        x = self.cbr(conv5)
        x = self.conv_last(x)
        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=label_size, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)
        return x


# outsideNet
class multi_level_pooling(nn.Module):
    def __init__(self, fc_dim=2048,
                 use_softmax=False, scales=(1, 2, 4, 8),
        super(multi_level_pooling, self).__init__()

        # Multi-Level Pooling
        self.mlp_pooling_layers = []
        self.mlp_conv_layers = []

        for scale in scales:
            self.mlp_pooling_layers.append(nn.AdaptiveAvgPool2d(scale))
            self.mlp_conv_layers.append(nn.Sequential(
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                BatchNorm2d(512),
                nn.ReLU(inplace=True)
            ))
        self.mlp_pooling_layers = nn.ModuleList(self.mlp_pooling_layers)
        self.mlp_conv_layers = nn.ModuleList(self.mlp_conv_layers)
        self.mlp_last_conv = conv3x3_bn_relu(fc_dim + len(scales) * 512, fpn_dim, 1)

    def forward(self, conv_out, label_size=None):
        conv5 = conv_out[-1]

        input_size = conv5.size()
        mlp_out = [conv5]
        for pool_scale, pool_conv in zip(self.mlp_pooling_layers, self.mlp_conv_layers):
            mlp_out.append(pool_conv(nn.functional.interpolate(
                pool_scale(conv5),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False)))
        mlp_out = torch.cat(mlp_out, 1)
        f = self.mlp_last_conv(mlp_out)

        return f

class feature_fusion(nn.Module):
    def __init__(self, num_class=24, fc_dim=2048,
                 use_softmax=False, fpn_inplanes=(256, 512, 1024, 2048), fpn_dim=512):
        super(feature_fusion, self).__init__()
        self.use_softmax = use_softmax

        # FPN Module
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:   # skip the top layer
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(fpn_inplane, fpn_dim, kernel_size=1, bias=False),
                BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True)
            ))
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        for _ in range(len(fpn_inplanes) - 1):  # skip the top layer
            self.fpn_out.append(nn.Sequential(conv3x3_bn_relu(fpn_dim, fpn_dim, 1)))
        self.fpn_out = nn.ModuleList(self.fpn_out)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out, label_size=None):
        conv5 = conv_out[-1]

        fpn_feature_list = [f]
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch

            f = nn.functional.interpolate(
                f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)  # top-down branch
            f = conv_x + f

            fpn_feature_list.append(self.fpn_out[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        if self.use_softmax:  # is True during inference
            x = nn.functional.interpolate(
                x, size=label_size, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
            return x

        x = nn.functional.log_softmax(x, dim=1)

        return x

