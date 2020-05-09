#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch

class HrizontalFlip(object):

    def __call__(self, sample):
        """
        sample: List of follows
            - image : C x H x W
            - depth : C x H x W
        """
        image, depth = sample
        if torch.rand(1) > 0.5:
            image = image[:,:,torch.arange(image.size(2)-1, -1, -1)]
            depth = depth[:,:,torch.arange(depth.size(2)-1, -1, -1)]
        
        return image, depth
