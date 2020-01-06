#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1; python train.py \
experiments/seg_detector/rctw17_resnet50_deform_thre.yaml \
--num_gpus 1 --batch_size 8 \
--resume outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss_1215/model/final \
--epochs 150 \
--lr 0.0007
#--resume outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss_1215/model/final \

