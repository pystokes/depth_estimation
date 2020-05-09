#!/usr/bin/python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from logging import getLogger
import cv2
import numpy as np
import torch

logger = getLogger('DepthEstimation')

class Detector(object):

    def __init__(self, model, device, config, save_dir):
        
        self.model = model
        self.device = device
        self.config = config
        self.save_dir = save_dir.joinpath('detected')
        self.save_dir_depth = self.save_dir.joinpath('depths')

        self.save_dir_depth.mkdir(exist_ok=True, parents=True)

        # Configs
        self.depth_min = self.config.model.depth_min
        self.depth_max = self.config.model.depth_max
        self.colors = self.config.detect.colors


    def run(self, data_loader):

        logger.info('Begin detection.')
        self.model.eval()
        with torch.no_grad():

            detected_depth_list = [] if self.config.detect.save_results else None

            n_detected = 0
            for img_path, img in data_loader:

                # Convert tuple of length 1 to string
                img_path = img_path[0]

                if self.device.type == 'cuda':
                    img = img.to(self.device)

                depth = self.model(img)
                depth = depth.to('cpu')

                print(depth.min(), depth.max())

                if self.config.detect.save_results:
                    detected_depth_list.append(depth.tolist())

                if self.config.detect.visualize:
                    self._visualize(Path(img_path), depth)
                
                n_detected += 1
                if not (n_detected % 100):
                    logger.info(f'Progress: [{n_detected:08}/{len(data_loader.dataset):08}]')

        if self.config.detect.save_results:
            with open(str(self.save_dir.parent.joinpath('detected_depth.json')), 'w') as f:
                json.dump(detected_depth_list, f, ensure_ascii=False, indent=4)

        logger.info('Detection has finished.')


    def _visualize(self, img_path, depth):

        # Fill zero at incorrect depth predictions (< 0)
        visualized_dep = np.where(depth < self.depth_min, self.depth_min, depth)
        visualized_dep = np.where(depth > self.depth_max, self.depth_max, visualized_dep)

        # Depth
        visualized_dep = (visualized_dep - self.depth_min) / (self.depth_max - self.depth_min) * 255
        visualized_dep = np.squeeze(visualized_dep)
        visualized_dep = cv2.resize(visualized_dep, (self.config.model.org_size_hw[1], self.config.model.org_size_hw[0]), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(self.save_dir_depth.joinpath(img_path.name)), visualized_dep)


if __name__ == '__main__':
    pass
