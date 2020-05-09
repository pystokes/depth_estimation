#!/usr/bin/python3
# -*- coding: utf-8 -*-

from logging import getLogger
from pathlib import Path
from attrdict import AttrDict
import torch

logger = getLogger('DepthEstimation')

class Config(object):

    def __init__(self):

        # Requirements : model
        _org_size_hw = (450, 600)
        _input_size_hw = (224, 224)
        _depth_min = 0.7132995128631592
        _depth_max = 9.99547004699707
        # Requirements : preprocess
        _preprocess_input_file = None
        _preprocess_save_dir = None
        # Requirements : train
        _train_input_dir = '../_data_storage/NYUD_v2/preprocessed'
        _train_save_dir = None

        self.model = {
            'org_size_hw': _org_size_hw,
            'input_size_hw': _input_size_hw,
            'depth_min': _depth_min,
            'depth_max': _depth_max,
            'freeze_backbone': True,
            'n_classes': 41,
            # Image must be divided by 255 before normalizing by follows
            'rgb_means': (0.406, 0.456, 0.485),
            'rgb_stds': (0.225, 0.224, 0.229)
        }

        self.preproces = {
            'input_file': _preprocess_input_file,
            'save_dir': _preprocess_save_dir,
        }

        self.train = {
            'input_dir': _train_input_dir,
            'save_dir': _train_save_dir,
            'resume_weight_path': '',
            'num_workers': 0,
            'batch_size': 128, # Max is 64 per one TITAN RTX
            'epoch': 500,
            'shuffle': True,
            'split_random_seed': 0,
            'weight_save_period': 5,
            'loss': {
                'balance_factor': 0.5,
            },
            'optimizer': {
                'type': 'adam',
                'adam': {
                    'lr': 5e-5,
                    'betas': (0.9, 0.999),
                    'eps': 1e-08,
                    'weight_decay': 0,
                    'amsgrad': False
                },
                'sgd': {
                    'lr': 1e-5,
                    'momentum': 0.9,
                    'weight_decay': 5e-4,
                    'wait_decay_epoch': 50,
                    'T_max': 10
                }
            }
        }

        self.detect = {
            'trained_weight_path': '',
            'visualize': True,
            'save_results': False,
            'colors': {
                "0": [0, 0, 0],
                "1": [255,0,0],
                "2": [212,155,187],
                "3": [148,89,142],
                "4": [56,13,127],
                "5": [196,214,129],
                "6": [256,162,205],
                "7": [70,31,71],
                "8": [100,77,107],
                "9": [169,63,35],
                "10": [158,209,41],
                "11": [242,72,209],
                "12": [172,9,221],
                "13": [190,26,181],
                "14": [25,244,191],
                "15": [2,207,122],
                "16": [59,109,127],
                "17": [184,29,136],
                "18": [32,141,98],
                "19": [66,128,157],
                "20": [178,163,87],
                "21": [170,48,139],
                "22": [171,4,25],
                "23": [178,9,239],
                "24": [139,242,27],
                "25": [101,158,132],
                "26": [38,248,10],
                "27": [70,206,248],
                "28": [200,93,100],
                "29": [128,77,56],
                "30": [108,25,156],
                "31": [161,161,25],
                "32": [82,214,47],
                "33": [159,128,206],
                "34": [245,38,43],
                "35": [123,115,135],
                "36": [83,151,25],
                "37": [249,201,249],
                "38": [113,208,227],
                "39": [39,235,111],
                "40": [117,247,162]
            }
        }

    def build_config(self):

        config = {
            'model': self.model,
            'preprocess': self.preproces,
            'train': self.train,
            'detect': self.detect,
        }

        logger.info(config)

        return AttrDict(config)


if __name__ == '__main__':

    from pprint import pprint

    config = Config().build_config()
    pprint(config)
