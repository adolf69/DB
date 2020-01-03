#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4; python train.py \
experiments/seg_detector/rctw17_resnet50_deform_thre.yaml \
--num_gpus 1 --batch_size 8 \
--resume outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss_rctw_test_15/model/final \
--epochs 5
