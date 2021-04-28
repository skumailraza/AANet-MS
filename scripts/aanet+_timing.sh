#!/usr/bin/env bash

# Timing on Scene Flow test set
 python -u inference.py \
--mode test \
--data_dir /netscratch/kraza/SceneFlow_Mixed \
--pretrained_aanet checkpoints/aanetms_sceneflow/aanet_latest.pth \
--max_disp 24 \
--batch_size 1 \
--img_height 576 \
--img_width 960 \
--feature_type ganet \
--feature_pyramid \
--refinement_type stereodrnet \
--no_intermediate_supervision \
--count_time 2>&1 |tee logs/profiler/aanetms_sf_md_24_SDRNet.txt

# Timing on KITTI 2015 test set
 python -u inference.py \
--mode test \
--max_disp 24 \
--data_dir /ds-av/public_datasets/kitti2015/raw \
--dataset_name KITTI2015 \
--pretrained_aanet checkpoints/aanetms_sceneflow/aanet_latest.pth \
--batch_size 1 \
--img_height 384 \
--img_width 1248 \
--feature_type ganet \
--feature_pyramid \
--refinement_type stereodrnet \
--no_intermediate_supervision \
--output_dir output/kitti15_test \
--count_time 2>&1 |tee logs/profiler/aanetms_k15_md_24_SDRNet.txt
