try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from utils.general import *

if __name__ == '__main__':
    if len(sys.argv) == 2:
        # Usage: python dataset_player.py weights/yolov5s_coco_100.pt
        weights = sys.argv[1]
        print('Stripping optimizer in weights: %s...' % weights)
        print()
    else:
        print('Error: Only one parameter (weights) is supported.')
        exit()
    
    strip_optimizer(weights)

