# yolov5-test

A test version of Yolov5 in PyTorch for object detection

## Acknowledgement
 - This repository references [ultralytics](https://github.com/ultralytics/yolov5)'s work.

## Dataset
 - Check COCO 2017 dataset
   ```
   python dataset_player.py
   python dataset_player.py --training
   ```
 - Check KITTI dataset
   ```
   python dataset_player.py --data=data/kitti.yaml
   python dataset_player.py --data=data/kitti.yaml --training
   ```

## Training
 - Train on COCO 2017 dataset
   ```
   python train.py
   ```
 - Train on KITTI dataset
   ```
   python train.py --data=data/kitti.yaml
   ```

## Evaluation
 - Evaluate on COCO 2017 dataset
   ```
   python test.py --weights=weights/coco/yolov5s_100ep.pt
   ```
 - Evaluate on KITTI dataset
   ```
   python test.py --weights=weights/kitti/yolov5s_100ep.pt
   ```

| Model                | Dataset  | Epoch | val mAP@.5        | val mAP@.5:.95    |
|:--------------------:|:--------:|:-----:|:-----------------:|:-----------------:|
| yolov5s(no pretrain) | COCO     | 100   | 0.51              | 0.314             |
| yolov5s(no pretrain) | COCO128  | 300   |                   |                   |
| yolov5s(no pretrain) | KITTI    | 100   |                   |                   |
| yolov5s              | KITTI    | 100   |                   |                   |
| yolov5s(no pretrain) | SEUMM HQ | 100   |                   |                   |
| yolov5s              | SEUMM HQ | 100   |                   |                   |

## Demo
 - Run on the image with COCO 2017 model
   ```
   python detect.py --weights=weights/coco/yolov5s_100ep.pt --source=my_image.jpg --view-img
   ```
 - Run on the image with KITTI model
   ```
   python detect.py --weights=weights/kitti/yolov5s_100ep.pt --source=my_image.jpg --view-img
   ```
