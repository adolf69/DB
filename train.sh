#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=4,5,6,7; python train.py \
experiments/seg_detector/rctw17_resnet50_deform_thre.yaml \
--num_gpus 4 --batch_size 16 \
--num_workers 32 \
--resume output_aws/outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss_rctw200/model/final \
# --epochs 200 \
#--lr 0.000
#--resume outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss_1215/model/final \

