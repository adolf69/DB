#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python demo.py \
experiments/seg_detector/rctw17_resnet50_deform_thre.yaml \
--image_path images/test/test_2.png \
--resume output_aws/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss_rctw_x500_50/model/final \
--visualize \
--box_thresh 0.3 \
--thresh 0.3 \
--IsResize \
--image_short_side 1600
#--visualize


