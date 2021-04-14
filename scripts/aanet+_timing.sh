#!/usr/bin/env bash

# Timing on Scene Flow test set
CUDA_VISIBLE_DEVICES=0 python inference.py \
--mode test \
--data_dir /home/kraza/SceneFlow_Mixed \
--pretrained_aanet pretrained/aanet_sceneflow-5aa5a24e.pth \
--batch_size 1 \
--img_height 576 \
--img_width 960 \
--feature_type ganet \
--feature_pyramid \
--refinement_type stereodrnet \
--no_intermediate_supervision \
--count_time

# Timing on KITTI 2015 test set
CUDA_VISIBLE_DEVICES=0 python inference.py \
--mode test \
--data_dir /mnt/serv-2101/public_datasets/kitti2015/raw \
--dataset_name KITTI2015 \
--pretrained_aanet pretrained/aanet_kitti15-fb2a0d23.pth \
--batch_size 1 \
--img_height 384 \
--img_width 1248 \
--feature_type ganet \
--feature_pyramid \
--refinement_type stereodrnet \
--no_intermediate_supervision \
--output_dir output/kitti15_test \
--count_time
