#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = getLogger('DepthEstimation')


class DepthEstimation(nn.Module):

    def __init__(self, exec_type, config):

        super(DepthEstimation, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def init_weights(self):
        pass

    def load_weights(self, trained_weights):

        state_dict = torch.load(trained_weights, map_location=lambda storage, loc: storage)
        try:
            # Load weights trained by single GPU into single GPU
            self.load_state_dict(state_dict) 
        except:
            # Load weights trained by multi GPU into single GPU
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.', '')
                new_state_dict[k] = v

            self.load_state_dict(new_state_dict) 


class Encoder(nn.Module):

    def __init__(self):

        super(Encoder, self).__init__()       

        self.backbone = models.densenet161(pretrained=True)

    def forward(self, x):

        features = [x]

        for _, v in self.backbone.features._modules.items():
            features.append(v(features[-1]))

        return features


class Decoder(nn.Module):

    def __init__(self, num_features=2208, decoder_width=0.5):

        super(Decoder, self).__init__()

        features = int(num_features * decoder_width)

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)

        self.up_sample1 = UpSample(skip_input=features // 1 + 384, output_features=features //  2)
        self.up_sample2 = UpSample(skip_input=features // 2 + 192, output_features=features //  4)
        self.up_sample3 = UpSample(skip_input=features // 4 +  96, output_features=features //  8)
        self.up_sample4 = UpSample(skip_input=features // 8 +  96, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):

        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)
        x_d1 = self.up_sample1(x_d0, x_block3)
        x_d2 = self.up_sample2(x_d1, x_block2)
        x_d3 = self.up_sample3(x_d2, x_block1)
        x_d4 = self.up_sample4(x_d3, x_block0)

        output = self.conv3(x_d4)

        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=True)

        return output


class UpSample(nn.Sequential):
    
    def __init__(self, skip_input, output_features):

        super(UpSample, self).__init__()        

        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):

        x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)

        x = self.convA(torch.cat([x, concat_with], dim=1))
        x = self.leakyreluA(x)

        x = self.convB(x)
        x = self.leakyreluB(x)

        return x


if __name__ == '__main__':
    pass
