#!/usr/bin/env bash

# Inference on Scene Flow test set
# python inference.py \
#--mode test \
#--data_dir /netscratch/kraza/SceneFlow_Mixed \
#--pretrained_aanet checkpoints/aanetms_sceneflow/aanet_best.pth \
#--batch_size 1 \
#--img_height 576 \
#--img_width 960 \
#--feature_type ganet \
#--feature_pyramid \
#--refinement_type stereodrnet \
#--no_intermediate_supervision

# Inference on KITTI 2015 test set for submission
python inference.py \
--mode test \
--data_dir /ds-av/public_datasets/kitti2015/raw \
--dataset_name KITTI2015 \
--max_disp 24 \
--pretrained_aanet checkpoints/aanetms_kitti15_md_24/aanet_latest.pth \
--batch_size 1 \
--img_height 384 \
--img_width 1248 \
--feature_type ganet \
--feature_pyramid \
--refinement_type stereodrnet \
--no_intermediate_supervision \
--output_dir output/kitti15_test

## Inference on KITTI 2012 test set for submission
#CUDA_VISIBLE_DEVICES=0 python inference.py \
#--mode test \
#--data_dir data/KITTI/kitti_2012/data_stereo_flow \
#--dataset_name KITTI2012 \
#--pretrained_aanet pretrained/aanet+_kitti12-56157808.pth \
#--batch_size 1 \
#--img_height 384 \
#--img_width 1248 \
#--feature_type ganet \
#--feature_pyramid \
#--refinement_type hourglass \
#--no_intermediate_supervision \
#--output_dir output/kitti12_test
