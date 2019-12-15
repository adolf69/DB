#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python demo.py \
experiments/seg_detector/rctw17_resnet50_deform_thre.yaml \
--image_path images/miao_111.png \
--resume models/pre-trained-model-synthtext-resnet50 \
--image_short_sid 1024 \
--box_thresh 0.1 \
--thresh 0.3 \
--visualize
