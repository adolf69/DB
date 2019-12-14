#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python demo.py \
experiments/seg_detector/rctw17_resnet50_deform_thre.yaml \
--image_path images/miao_111.png \
--resume outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/model_epoch_5_minibatch_1500 \
--box_thresh 0.1 \
--thresh 0.1 \
--polygon \
--visualize