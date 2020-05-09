#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(object):

    def __init__(self, config, device):

        self.depth_loss = nn.L1Loss()
        #self.depth_loss = nn.MSELoss()
        #self.depth_loss = DepthLogLoss(config.train.loss.balance_factor)
        self.gradient_loss = GradientLoss(device)

    def calc_loss(self, out_depths, gt_depths):

        loss_depth = self.depth_loss(out_depths, gt_depths)
        loss_grad = self.gradient_loss(out_depths, gt_depths)

        return loss_depth, loss_grad

class DepthLogLoss(nn.Module):

    def __init__(self, balance_factor):

        super(DepthLogLoss, self).__init__()
        
        self.balance_factor = balance_factor

    def forward(self, inputs, targets):

        n, _, h, w = inputs.shape
        n_pixel = n * h * w

        inputs = torch.log(inputs + 1e-8) # log(0) is '-inf'
        targets = torch.log(targets)
        d = inputs - targets

        loss = torch.sum(d**2) / n_pixel - self.balance_factor * (torch.sum(d)**2) / (n_pixel**2)

        return loss


class GradientLoss(nn.Module):

    def __init__(self, device):

        super(GradientLoss, self).__init__()

        self.sobel_filter = SobelFilter(device)

    def forward(self, inputs, targets):

        n, _, h, w = inputs.shape
        n_pixel = n * h * w

        inputs_grad_h = F.conv2d(inputs, self.sobel_filter.sobel_filter_h)
        inputs_grad_v = F.conv2d(inputs, self.sobel_filter.sobel_filter_v)
        targets_grad_h = F.conv2d(targets, self.sobel_filter.sobel_filter_h)
        targets_grad_v = F.conv2d(targets, self.sobel_filter.sobel_filter_v)

        loss = torch.sum(abs(inputs_grad_h - targets_grad_h) + abs(inputs_grad_v - targets_grad_v)) / n_pixel

        return loss


class SobelFilter(object):

    def __init__(self, device):

        self.kernel_h = torch.FloatTensor([[1, 2, 1],
                                           [0, 0, 0],
                                           [-1, -2, -1]])
        self.sobel_filter_h = self.kernel_h.expand(1, 1, 3, 3).to(device)

        self.kernel_v = torch.FloatTensor([[1, 0, -1],
                                           [2, 0, -2],
                                           [1, 0, -1]])
        self.sobel_filter_v = self.kernel_v.expand(1, 1, 3, 3).to(device)


if __name__ == '__main__':
    pass
