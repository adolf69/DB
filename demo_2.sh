#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python demo_2.py \
experiments/seg_detector/td500_resnet50_deform_thre.yaml \
--image_path images/miao_111.png \
--resume models/td500_resnet50 \
--box_thresh 0.1 \
--visualize