#!/usr/bin/env bash
#cd nets/deform_conv/
#sh build.sh
# Train on Scene Flow training set
CUDA_VISIBLE_DEVICES=0,1 python train.py \
--mode val \
--checkpoint_dir checkpoints/aanet+_sceneflow \
--data_dir /home/kraza/SceneFlow_Mixed \
--batch_size 2 \
--max_disp 24 \
--val_batch_size 2 \
--img_height 288 \
--img_width 576 \
--val_img_height 576 \
--val_img_width 960 \
--feature_type ganet \
--feature_pyramid \
--refinement_type stereonet \
--milestones 20,30,40,50,60 \
--max_epoch 64 \
--wandb 0 \
--wbRunName "aanetms+_sceneflow"
exit
# Train on mixed KITTI 2012 and KITTI 2015 training set
# python train.py \
#--data_dir data/KITTI \
#--dataset_name KITTI_mix \
#--checkpoint_dir checkpoints/aanet+_kittimix \
#--pretrained_aanet checkpoints/aanet+_sceneflow/aanet_best.pth \
#--batch_size 32 \
#--val_batch_size 32 \
#--img_height 288 \
#--img_width 1152 \
#--val_img_height 384 \
#--val_img_width 1248 \
#--feature_type ganet \
#--feature_pyramid \
#--refinement_type hourglass \
#--load_pseudo_gt \
#--milestones 400,600,800,900 \
#--max_epoch 1000 \
#--save_ckpt_freq 100 \
#--no_validate \
#--wandb 1 \
#--wbRunName "aanet+_kittimix"

# Train on KITTI 2015 training set
 python train.py \
--data_dir /ds-av/public_datasets/kitti2015/raw \
--dataset_name KITTI2015 \
--mode train_all \
--checkpoint_dir checkpoints/aanet+_kitti15 \
--pretrained_aanet checkpoints/aanet+_sceneflow/aanet_latest.pth \
--batch_size 24 \
--val_batch_size 16 \
--img_height 384 \
--img_width 1248 \
--val_img_height 384 \
--val_img_width 1248 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--highest_loss_only \
--freeze_bn \
--learning_rate 1e-4 \
--milestones 400,600,800,900 \
--max_epoch 1000 \
--save_ckpt_freq 100 \
--no_validate \
--wandb 1 \
--wbRunName "aanet+_kitti15"


# Train on KITTI 2012 training set
 python train.py \
--data_dir /ds-av/public_datasets/kitti2012/raw \
--dataset_name KITTI2012 \
--mode train_all \
--checkpoint_dir checkpoints/aanet+_kitti12 \
--pretrained_aanet checkpoints/aanet+_sceneflow/aanet_latest.pth \
--batch_size 24 \
--val_batch_size 16 \
--img_height 384 \
--img_width 1248 \
--val_img_height 384 \
--val_img_width 1248 \
--feature_type ganet \
--feature_pyramid \
--refinement_type hourglass \
--highest_loss_only \
--freeze_bn \
--learning_rate 1e-4 \
--milestones 400,600,800,900 \
--max_epoch 1000 \
--save_ckpt_freq 100 \
--no_validate \
--wandb 1 \
--wbRunName "aanet+_kitti12"

