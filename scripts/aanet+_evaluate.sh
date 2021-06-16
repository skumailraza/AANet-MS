#!/usr/bin/env bash
sh compile.sh

# Evaluate the best validation model on Scene Flow test set
 python train.py \
--mode test \
--checkpoint_dir checkpoints/aanetms_sceneflow_md_24 \
--data_dir /netscratch/kraza/SceneFlow_Mixed \
--batch_size 64 \
--max_disp 24 \
--val_batch_size 1 \
--img_height 288 \
--img_width 576 \
--val_img_height 576 \
--val_img_width 960 \
--feature_type ganet \
--feature_pyramid \
--wandb 1 \
--wbRunName "aanetms_md_24_sf_eval" \
--refinement_type stereodrnet \
--milestones 20,30,40,50,60 \
--max_epoch 64 \
--evaluate_only

# Evaluate a specific model on Scene Flow test set
# python train.py \
#--mode test \
#--checkpoint_dir checkpoints/aanet+_sceneflow \
#--pretrained_aanet pretrained/aanet+_sceneflow-d3e13ef0.pth \
#--batch_size 64 \
#--val_batch_size 1 \
#--img_height 288 \
#--img_width 576 \
#--val_img_height 576 \
#--val_img_width 960 \
#--feature_type ganet \
#--feature_pyramid \
#--refinement_type hourglass \
#--milestones 20,30,40,50,60 \
#--max_epoch 64 \
#--evaluate_only