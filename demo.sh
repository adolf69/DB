#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 python demo.py \
experiments/seg_detector/rctw17_resnet50_deform_thre.yaml \
--image_path images/miao2.png \
--resume outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/final \
--visualize \
--box_thresh 0.1 \
--thresh 0.3 \
--image_short_side 2400
#--polygon
#--image_short_side 2400
