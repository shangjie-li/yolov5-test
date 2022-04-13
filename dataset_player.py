import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import argparse
import yaml
from numpy import random

import numpy as np
import torch

from utils.datasets import LoadImagesAndLabels, load_image
from utils.general import check_dataset, check_file, scale_coords, xywh2xyxy
from utils.plots import plot_one_box


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--training', action='store_true', help='use training split')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    opt = parser.parse_args()
    
    opt.data, opt.hyp = check_file(opt.data), check_file(opt.hyp)  # check files
    
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    
    # Dataset
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    
    check_dataset(data_dict)  # check
    if opt.training:
        data_path = data_dict['train']
        augment = True
    else:
        data_path = data_dict['val']
        augment = False
        
    names = data_dict['names']
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    path = data_dict['train'] if opt.training else data_dict['val']
    dataset = LoadImagesAndLabels(data_path, batch_size=1, augment=augment, hyp=hyp,
        rect=opt.rect, cache_images=opt.cache_images, single_cls=opt.single_cls,
        image_weights=opt.image_weights
    )
    
    for i in range(len(dataset)):
        print('\n--------[%d/%d]--------' % (i + 1, len(dataset)))
        img_raw, _, _ = load_image(dataset, i)
        
        img_tensor, labels, index, shapes = dataset[i]
        img_array = img_tensor.numpy().transpose(1, 2, 0) # to 416x416x3
        img_array = img_array[:, :, ::-1] # to BGR
        img_array = img_array.copy()
        
        if len(labels):
            h, w = img_array.shape[0], img_array.shape[1]
            labels[:, -4:] = xywh2xyxy(labels[:, -4:])
            labels[:, -4::2] *= w
            labels[:, -3::2] *= h
            labels[:, -4:] = labels[:, -4:].round()
            for batch_id, class_id, *xyxy in labels:
                s = f'{names[int(class_id)]}'
                plot_one_box(xyxy, img_array, label=s, color=colors[int(class_id)], line_thickness=3)
        
        cv2.imshow('Raw Data', img_raw)
        cv2.imshow('Processed Data', img_array)
        
        print('index:', index)
        print('shape:', img_raw.shape)
        print('labels:\n', labels)
        print()
        
        key = cv2.waitKey(0)
        if key == 27:
            break
        else:
            continue
        
        
    
    
