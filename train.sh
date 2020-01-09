#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1,2,4; python train.py \
experiments/seg_detector/rctw17_resnet50_deform_thre.yaml \
--num_gpus 4 --batch_size 16 \
--resume outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss/model/final_320_640 \
--epochs 30 \
--lr 0.00001
#--resume outputs/workspace/DB/SegDetectorModel-seg_detector/deformable_resnet50/L1BalanceCELoss_1215/model/final \

