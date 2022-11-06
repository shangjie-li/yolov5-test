import os
import time
import threading
import argparse
import numpy as np
import torch

import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device


parser = argparse.ArgumentParser(
    description='Demo script for YOLOv5')
parser.add_argument('--weights', default='weights/coco/yolov5s_100ep.pt', type=str,
    help='Weights of the model.')
parser.add_argument('--score_thresh', default=0.3, type=float,
    help='The score threshold for detection.')
parser.add_argument('--class_filter', default=None, type=str,
    help='A filter to keep desired classes, e.g., 0/2/5/7 (split by a slash).')
parser.add_argument('--sub_image', default='/kitti/camera_color_left/image_raw', type=str,
    help='The image topic to subscribe.')
parser.add_argument('--pub_image', default='/result', type=str,
    help='The image topic to publish.')
parser.add_argument('--frame_rate', default=10, type=int,
    help='Working frequency.')
parser.add_argument('--display', action='store_true',
    help='Whether to display and save all videos.')
parser.add_argument('--print', action='store_true',
    help='Whether to print and record infos.')
args = parser.parse_args()


image_lock = threading.Lock()


def get_stamp(header):
    return header.stamp.secs + 0.000000001 * header.stamp.nsecs


def publish_image(pub, data, frame_id='base_link'):
    assert len(data.shape) == 3, 'len(data.shape) must be equal to 3.'
    header = Header(stamp=rospy.Time.now())
    header.frame_id = frame_id

    msg = Image()
    msg.height = data.shape[0]
    msg.width = data.shape[1]
    msg.encoding = 'rgb8'
    msg.data = np.array(data).tobytes()
    msg.header = header
    msg.step = msg.width * 1 * 3

    pub.publish(msg)


def display(img, v_writer, win_name='result'):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    v_writer.write(img)
    key = cv2.waitKey(1)
    if key == 27:
        v_writer.release()
        return False
    else:
        return True


def print_info(frame, stamp, delay, labels, scores, boxes, file_name='result.txt'):
    time_str = 'frame:%d  stamp:%.3f  delay:%.3f' % (frame, stamp, delay)
    print(time_str)
    with open(file_name, 'a') as fob:
        fob.write(time_str + '\n')
    for i in range(len(labels)):
        info_str = 'box:%d %d %d %d  score:%.2f  label:%s' % (
            boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], scores[i], labels[i]
        )
        print(info_str)
        with open(file_name, 'a') as fob:
            fob.write(info_str + '\n')
    print()
    with open(file_name, 'a') as fob:
        fob.write('\n')


def draw_predictions(img, label, score, box, color=(156, 39, 176)):
    f_face = cv2.FONT_HERSHEY_SIMPLEX
    f_scale = 0.5
    f_thickness, l_thickness = 1, 2

    h, w, _ = img.shape
    u1, v1, u2, v2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
    cv2.rectangle(img, (u1, v1), (u2, v2), color, l_thickness)

    text = '%s: %.2f' % (label, score)
    text_w, text_h = cv2.getTextSize(text, f_face, f_scale, f_thickness)[0]
    text_h += 6
    if v1 - text_h < 0:
        cv2.rectangle(img, (u1, text_h), (u1 + text_w, 0), color, -1)
        cv2.putText(img, text, (u1, text_h - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (u1, v1), (u1 + text_w, v1 - text_h), color, -1)
        cv2.putText(img, text, (u1, v1 - 4), f_face, f_scale, (255, 255, 255), f_thickness, cv2.LINE_AA)

    return img


class Yolov5Detector():
    def __init__(self, weights):
        imgsz = 640
        self.device = device = select_device('')
        self.half = half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = model = attempt_load(weights, map_location=device)  # load FP32 model
        self.stride = stride = int(model.stride.max())  # model stride
        self.imgsz = imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Get names
        self.names = model.module.names if hasattr(model, 'module') else model.names

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    def run(self, img0, conf_thres=0.25, iou_thres=0.45, class_filter=None):
        """
        Args:
            img0: (h, w, 3), BGR format
            conf_thres: float, object confidence threshold
            iou_thres: float, IOU threshold for NMS
            class_filter: list(int), filter by class, for instance [0, 2, 3]
        Returns:
            labels: list(str), names of objects
            scores: list(float)
            boxes: (n, 4), xyxy format
        """
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=class_filter, agnostic=False)

        # Process detections
        det = pred[0]
        if len(det):
            # Rescale boxes from imgsz to img0 size
            boxes = scale_coords(img.shape[2:], det[:, :4], img0.shape).round().cpu().numpy()  # xyxy
            labels = [self.names[int(cls)] for cls in det[:, -1]]
            scores = [float('%.2f' % conf) for conf in det[:, -2]]
            return labels, scores, boxes
        else:
            return [], [], np.array([]).reshape(-1, 4)


def image_callback(image):
    global image_stamp, image_frame
    image_lock.acquire()
    image_stamp = get_stamp(image.header)
    image_frame = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1) # BGR image
    image_lock.release()


def timer_callback(event):
    global image_stamp, image_frame
    image_lock.acquire()
    cur_stamp = image_stamp
    cur_frame = image_frame[:, :, ::-1].copy() # to RGB
    image_lock.release()
    
    global frame
    frame += 1
    start = time.time()
    labels, scores, boxes = detector.run(
        cur_frame, conf_thres=score_thresh, class_filter=class_filter)

    cur_frame = cur_frame[:, :, ::-1].copy() # to BGR
    for i in np.argsort(scores):
        cur_frame = draw_predictions(
            cur_frame, str(labels[i]), float(scores[i]), boxes[i])
    
    if args.display:
        if not display(cur_frame, v_writer, win_name='result'):
            print("\nReceived the shutdown signal.\n")
            rospy.signal_shutdown("Everything is over now.")

    cur_frame = cur_frame[:, :, ::-1].copy() # to RGB
    publish_image(pub, cur_frame)
    delay = round(time.time() - start, 3)
    
    if args.print:
        print_info(frame, cur_stamp, delay, labels, scores, boxes, file_name)


if __name__ == '__main__':
    rospy.init_node("yolov5", anonymous=True, disable_signals=True)
    frame = 0

    if args.print:
        file_name = 'result.txt'
        with open(file_name, 'w') as fob:
            fob.seek(0)
            fob.truncate()

    assert os.path.exists(args.weights), '%s Not Found' % args.weights
    detector = Yolov5Detector(weights=args.weights)
    score_thresh = args.score_thresh
    class_filter = list(map(int, args.class_filter.split('/'))) if args.class_filter is not None else None

    image_stamp = None
    image_frame = None
    rospy.Subscriber(args.sub_image, Image, image_callback, queue_size=1, buff_size=52428800)
    while image_frame is None:
        time.sleep(0.1)
        print('Waiting for topic %s...' % args.sub_image)
    print('  Done.\n')

    if args.display:
        win_h, win_w = image_frame.shape[0], image_frame.shape[1]
        v_path = 'result.mp4'
        v_format = cv2.VideoWriter_fourcc(*"mp4v")
        v_writer = cv2.VideoWriter(v_path, v_format, args.frame_rate, (win_w, win_h), True)

    pub = rospy.Publisher(args.pub_image, Image, queue_size=1)
    rospy.Timer(rospy.Duration(1 / args.frame_rate), timer_callback)
    rospy.spin()
