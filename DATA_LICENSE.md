# Data and evidence terms

KITTI: Andreas Geiger, Philip Lenz, Raquel Urtasun, CVPR 2012, *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*.

Source: https://www.cvlibs.net/datasets/kitti/eval_tracking.php
License source: https://www.cvlibs.net/datasets/kitti/

KITTI images, annotations and derived data/evidence are subject to Creative Commons Attribution-NonCommercial-ShareAlike 3.0: https://creativecommons.org/licenses/by-nc-sa/3.0/

This is non-commercial educational research. Raw images, source label archives and model weights are not redistributed in Git. Reports retain KITTI attribution and sequence/frame lineage. The code license does not relicense upstream data. No official benchmark submission or commercial-use rights are claimed.

Detector: torchvision Faster R-CNN MobileNetV3 320 COCO_V1. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html
Model weights are downloaded separately from the official PyTorch host. The source weight hash is frozen in the protocol. No model training is performed.
