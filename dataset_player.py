import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

import argparse
import yaml

import numpy as np
import torch

from utils.datasets import LoadImagesAndLabels, load_image
from utils.general import check_dataset, check_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
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
        
    path = data_dict['train'] if opt.training else data_dict['val']
    dataset = LoadImagesAndLabels(data_path, batch_size=1, augment=augment, hyp=hyp,
        rect=opt.rect, cache_images=opt.cache_images, single_cls=opt.single_cls,
        image_weights=opt.image_weights
    )
    
    for i in range(len(dataset)):
        print('\n--------[%d/%d]--------' % (i + 1, len(dataset)))
        img_raw, _, _ = load_image(dataset, i)
        cv2.imshow('Raw Data', img_raw)
        
        img_tensor, labels, index, shapes = dataset[i]
        img_array = img_tensor.numpy().transpose(1, 2, 0) # to 416x416x3
        img_array = img_array[:, :, ::-1] # to BGR
        cv2.imshow('Processed Data', img_array)
        
        print('index:', index)
        print('shape:', img_array.shape)
        print('labels:\n', labels)
        print()
        
        # labels: batch_id, class_id, ????
        
        key = cv2.waitKey(0)
        if key == 27:
            break
        else:
            continue
        
        
    
    
