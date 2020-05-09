#!/usr/bin/env python3

import json
from pathlib import Path
import pickle
import random
import cv2
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io


# Config
random_seed = 17 # If change this value, you have to change 'colors' inconfig.py, too.
n_classes = 41
save_dir = Path('../../_data_storage/NYUD_v2/preprocessed')
img_h = 480
img_w = 640
crop_h = 450
crop_w = 600
n_train = 381
n_valid = 414
n_test = 654

# Pre-checked parameters
pre_depth_min = 0.7132995128631592
pre_depth_max = 9.99547004699707

# Prepare
save_dir.mkdir(exist_ok=False, parents=True)
save_dir.joinpath('ImageSets', 'Main').mkdir(exist_ok=False, parents=True)
save_dir.joinpath('Annotations', 'Depths').mkdir(exist_ok=False, parents=True)
save_dir.joinpath('Annotations', 'Labels').mkdir(exist_ok=False, parents=True)
save_dir.joinpath('JPEGImages').mkdir(exist_ok=False, parents=True)
save_dir.joinpath('_Visualized', 'Depths').mkdir(exist_ok=False, parents=True)
save_dir.joinpath('_Visualized', 'Labels').mkdir(exist_ok=False, parents=True)
margin_h = int((img_h - crop_h) / 2)
margin_w = int((img_w - crop_w) / 2)

# Generate color list for each class
random.seed(random_seed)
color_list = {}
for cls_idx in range(n_classes):
    if int(cls_idx) == 0:
        color_list[str(cls_idx)] = (0, 0, 0)
    elif int(cls_idx) == 1:
        color_list[str(cls_idx)] = (255, 0, 0)
    else:
        color_list[str(cls_idx)] = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
with open(save_dir.joinpath('color_list.json'), 'w') as f:
    json.dump(color_list, f, ensure_ascii=False, indent=4)

# Load data
data = h5py.File('../../_data_storage/NYUD_v2/nyu_depth_v2_labeled.mat', 'r')
labels = scipy.io.loadmat('../../_data_storage/NYUD_v2/labels40.mat')
labels = np.transpose(labels['labels40'], axes=[2,0,1])
images = data['images']
depths = data['depths']
assert len(images) == len(depths) == len(labels) == 1449, 'Dataset has fatal eorror.'

# Crop and save
dep_min = 99999
dep_max = 0
seg_min = 99999
seg_max = 0
img_counter = 0
for idx in range(n_train + n_valid + n_test):

    print(f'Processing: {img_counter:08}')

    img = np.transpose(images[idx], axes=[2,1,0])
    dep = np.transpose(depths[idx], axes=[1,0])
    seg = np.transpose(labels[idx], axes=[1,0])

    # Horizontal and vertical flip for adjust direction
    seg = cv2.rotate(seg, cv2.ROTATE_90_CLOCKWISE)
    seg = cv2.flip(seg, flipCode=1)

    # Crop image to deal with missing values on the edge
    crop_img = img[margin_h:-margin_h, margin_w:-margin_w, :]
    crop_dep = dep[margin_h:-margin_h, margin_w:-margin_w]
    crop_seg = seg[margin_h:-margin_h, margin_w:-margin_w]

    if crop_dep.min() < dep_min:
        dep_min = crop_dep.min()
    if crop_dep.max() > dep_max:
        dep_max = crop_dep.max()
    if crop_seg.min() < seg_min:
        seg_min = crop_seg.min()
    if crop_seg.max() > seg_max:
        seg_max = crop_seg.max()

    # Save image and depth
    save_id = f'{img_counter:08}'
    img_save_name = save_id + '.jpg'
    dep_save_name = save_id + '.pkl'
    seg_save_name = save_id + '.pkl'

    cv2.imwrite(str(save_dir.joinpath('JPEGImages', img_save_name)), crop_img)
    with open(save_dir.joinpath('Annotations', 'Depths', dep_save_name), 'wb') as f:
        pickle.dump(crop_dep, f)
    with open(save_dir.joinpath('Annotations', 'Labels', seg_save_name), 'wb') as f:
        pickle.dump(crop_seg, f)

    # For visualization
    visualized_dep = (crop_dep - pre_depth_min) / (pre_depth_max - pre_depth_min) * 255
    cv2.imwrite(str(save_dir.joinpath('_Visualized', 'Depths', img_save_name)), visualized_dep)

    visualized_seg = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
    for cls_idx, color_bgr in color_list.items():
        for i_channel, color_value in enumerate(color_bgr):
            paint = np.where(crop_seg == int(cls_idx), True, False)
            visualized_seg[:,:,i_channel][paint] = color_value
    cv2.imwrite(str(save_dir.joinpath('_Visualized', 'Labels', seg_save_name.replace('.pkl', '.jpg'))), visualized_seg)
    """
    # For just resize check
    crop_dep = cv2.resize(crop_dep, (224, 224), interpolation=cv2.INTER_CUBIC)
    visualized_dep = (crop_dep - pre_depth_min) / (pre_depth_max - pre_depth_min) * 255
    cv2.imwrite(str(save_dir.joinpath('_Visualized', 'Depths', img_save_name)), visualized_dep)

    visualized_seg = np.zeros((224, 224, 3), dtype=np.uint8)
    crop_seg = cv2.resize(crop_seg, (224, 224), interpolation=cv2.INTER_NEAREST)
    for cls_idx, color_bgr in color_list.items():
        for i_channel, color_value in enumerate(color_bgr):
            paint = np.where(crop_seg == int(cls_idx), True, False)
            visualized_seg[:,:,i_channel][paint] = color_value
    cv2.imwrite(str(save_dir.joinpath('_Visualized', 'Labels', seg_save_name.replace('.pkl', '.jpg'))), visualized_seg)
    """

    # Add list
    if img_counter < n_train:
        phase = 'train'
    elif img_counter < n_train + n_valid:
        phase = 'valid'
    else:
        phase = 'test'
    
    with open(save_dir.joinpath('ImageSets', 'Main', phase + '.txt'), 'a') as f:
        f.write(save_id + '\n')

    img_counter += 1

config = {'depth_min': float(dep_min),
          'depth_max': float(dep_max),
          'label_min': int(seg_min),
          'label_max': int((seg_max))}

with open(save_dir.joinpath('config.json'), 'w') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
