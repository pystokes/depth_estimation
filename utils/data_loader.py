#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os.path as osp
from pathlib import Path
import pickle
import re
import cv2
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms

from utils.augmentations import HrizontalFlip

class CreateDataLoader(object):

    def build_for_train(self, config):

        train_imgs, train_deps, valid_imgs, valid_deps  = self.make_filepath_list(config.train.input_dir)

        # Dataset
        train_dataset = BatchDataset(phase='train',
                                     imgs=train_imgs,
                                     deps=train_deps,
                                     config=config,
                                     transform=transforms.Compose(
                                         [
                                             HrizontalFlip()
                                         ]
                                     ))
        valid_dataset = BatchDataset(phase='train',
                                     imgs=valid_imgs,
                                     deps=valid_deps,
                                     config=config,
                                     transform=None)

        # Data loader
        train_loader = data.DataLoader(train_dataset,
                                       batch_size=config.train.batch_size,
                                       shuffle=config.train.shuffle)
        valid_loader = data.DataLoader(valid_dataset,
                                       batch_size=config.train.batch_size,
                                       shuffle=False)

        return train_loader, valid_loader


    def build_for_detect(self, config, x_dir):

        inputs = [img_path for img_path in Path(x_dir).glob('*') if re.fullmatch('.jpg|.jpeg|.png', img_path.suffix.lower())]

        dataset = BatchDataset(
            phase='predict',
            imgs=inputs,
            deps=None,
            config=config
        )

        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        return data_loader


    def make_filepath_list(self, input_dir):

        img_path_format = osp.join(input_dir, 'JPEGImages', '%s.jpg')
        dep_path_format = osp.join(input_dir, 'Annotations/Depths', '%s.pkl')

        train_files = osp.join(input_dir, 'ImageSets/Main/train.txt')
        val_files = osp.join(input_dir, 'ImageSets/Main/valid.txt')

        # Train files
        train_imgs = []
        train_deps = []

        for line in open(train_files):

            filename = line.strip()
            img_path = (img_path_format % filename)
            dep_path = (dep_path_format % filename)
            train_imgs.append(img_path)
            train_deps.append(dep_path)
        
        # validation files
        val_imgs = []
        val_deps = []

        for line in open(val_files):

            filename = line.strip()
            img_path = (img_path_format % filename)
            dep_path = (dep_path_format % filename)
            val_imgs.append(img_path)
            val_deps.append(dep_path)
        
        return train_imgs, train_deps, val_imgs, val_deps


class BatchDataset(torch.utils.data.Dataset):

    def __init__(self, phase, imgs, deps, config, transform=None):

        self.phase = phase
        self.imgs = imgs
        self.deps = deps
        self.resize_h = config.model.input_size_hw[0]
        self.resize_w = config.model.input_size_hw[1]
        self.depth_min = config.model.depth_min
        self.depth_max = config.model.depth_max
        self.rgb_means = config.model.rgb_means
        self.rgb_stds = config.model.rgb_stds
        self.transform = transform

    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, idx):

        # Load input image
        img = cv2.imread(str(self.imgs[idx])).astype(np.float32)
        img = cv2.resize(img, (self.resize_w, self.resize_h))
        img = img[:, :, ::-1].copy() # Reorder from BGR to RGB

        # Normalize
        img /= 255.
        img -= self.rgb_means
        img /= self.rgb_stds

        # [H, W, C] -> [C, H, W]
        img = torch.from_numpy(img).permute(2, 0, 1)

        # Load and augment targets
        if self.phase == 'train':

            # Load depth
            with open(str(self.deps[idx]), 'rb') as f:
                dep = pickle.load(f)
            dep = cv2.resize(dep, (self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC)

            # To Tensor
            dep = torch.from_numpy(dep)

            # Unsqueeze to make shape [C, H, W]
            dep = torch.unsqueeze(dep, dim=0)

            # Augmentation
            if self.transform:
                img, dep = self.transform([img, dep])

            return img, dep
            
        else:
            return str(self.imgs[idx]), img


if __name__ == '__main__':
    pass
