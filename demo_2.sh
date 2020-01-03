#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 python demo.py \
experiments/seg_detector/rctw17_resnet50_deform_thre.yaml \
--image_path images/miao_111.png \
--resume models/pre-trained-model-synthtext-resnet50 \
--box_thresh 0.2 \
--thresh 0.6 \
--visualize
#--image_short_side 1024 \
