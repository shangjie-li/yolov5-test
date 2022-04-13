# yolov5-test

A test version of Yolov5 in PyTorch for object detection

## Acknowledgement
 - This repository references [ultralytics](https://github.com/ultralytics/yolov5)'s work.

## Dataset
 - Check the COCO 2017 dataset
   ```
   python dataset_player.py
   python dataset_player.py --training
   ```

## Training
 - Train on COCO 2017 dataset
   ```
   python train.py
   ```

## Evaluation
 - Evaluate on COCO 2017 dataset
   ```
   python test.py --weights=weights/yolov5s_coco_100.pt
   ```
| Model            | val mAP@.5     | val mAP@.5:.95 |
|:----------------:|:--------------:|:--------------:|
| yolov5s_coco_100 | 0.51           | 0.314          |

## Demo
 - Run on the image with COCO 2017 model
   ```
   python detect.py --weights=weights/yolov5s_coco_100.pt --source=my_image.jpg --view-img
   ```
