try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from utils.general import strip_optimizer


if __name__ == '__main__':
    # Usage: python strip_optimizer.py path_to_your_ckpt
    if len(sys.argv) == 2:
        weights = sys.argv[1]
        print('Stripping optimizer in weights: %s...' % weights)
    else:
        print('Error: Only the parameter path_to_your_ckpt is needed.')
        exit()
    strip_optimizer(weights)

