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
 - Check SEUMM HQ dataset
   ```
   python dataset_player.py --data=data/seumm_hq.yaml
   python dataset_player.py --data=data/seumm_hq.yaml --training
   ```
 - Check SEUMM dataset
   ```
   python dataset_player.py --data=data/seumm.yaml
   python dataset_player.py --data=data/seumm.yaml --training
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
 - Train on SEUMM HQ dataset
   ```
   python train.py --data=data/seumm_hq.yaml
   ```
 - Train on SEUMM dataset
   ```
   python train.py --data=data/seumm.yaml
   ```

## Evaluation
 - Evaluate on COCO 2017 dataset
   ```
   python test.py --weights=weights/coco/yolov5s_100ep.pt
   ```
 - Evaluate on KITTI dataset
   ```
   python test.py --data=data/kitti.yaml --weights=weights/kitti/yolov5s_100ep.pt
   ```
 - Evaluate on SEUMM HQ dataset
   ```
   python test.py --data=data/seumm_hq.yaml --weights=weights/seumm_hq/yolov5s_100ep.pt
   ```
 - Evaluate on SEUMM dataset
   ```
   python test.py --data=data/seumm.yaml --weights=weights/seumm/yolov5s_100ep.pt
   ```
 - The result should be

| Model                | Dataset  | Epoch | val mAP@.5     | val mAP@.5:.95 |
|:--------------------:|:--------:|:-----:|:--------------:|:--------------:|
| yolov5s              | COCO     | 100   | 0.510          | 0.314          |
| yolov5s              | KITTI    | 100   | 0.647          | 0.388          |
| yolov5s (pretrained) | KITTI    | 100   | 0.679          | 0.430          |
| yolov5s              | SEUMM HQ | 100   | 0.952          | 0.613          |
| yolov5s (pretrained) | SEUMM HQ | 100   | 0.966          | 0.650          |
| yolov5s              | SEUMM    | 100   | 0.886          | 0.564          |
| yolov5s (pretrained) | SEUMM    | 100   | 0.906          | 0.596          |

## Demo
 - Run a demo with COCO 2017 model
   ```
   python detect.py --weights=weights/coco/yolov5s_100ep.pt --source=my_image.jpg --view-img
   python detect.py --weights=weights/coco/yolov5s_100ep.pt --source=test_images
   ```
 - Run a demo with KITTI model
   ```
   python detect.py --weights=weights/kitti/yolov5s_100ep.pt --source=my_image.jpg --view-img
   python detect.py --weights=weights/kitti/yolov5s_100ep.pt --source=test_images
   ```
 - Run a demo with SEUMM HQ model
   ```
   python detect.py --weights=weights/seumm_hq/yolov5s_100ep.pt --source=my_image.jpg --view-img
   python detect.py --weights=weights/seumm_hq/yolov5s_100ep.pt --source=test_images
   ```
 - Run a demo with SEUMM model
   ```
   python detect.py --weights=weights/seumm/yolov5s_100ep.pt --source=my_image.jpg --view-img
   python detect.py --weights=weights/seumm/yolov5s_100ep.pt --source=test_images
   ```
