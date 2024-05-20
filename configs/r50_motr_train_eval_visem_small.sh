# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------


# for visem
# sh configs/r50_motr_train_eval_visem_small.sh

PRETRAIN=r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth
EXP_DIR=exps_visem/exp_0521_timeAttn_test


python3 -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env main.py --meta_arch motr \
    --use_checkpoint \
    --dataset_file e2e_visem \
    --with_box_refine --lr_drop 10 --lr 2e-4 --lr_backbone 2e-5 \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' --sample_interval 10 \
    --sampler_steps 5 9 15 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos --merger_dropout 0 --dropout 0 --random_drop 0.1 --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' --extra_track_attn \
    --data_txt_path_train ./datasets/data_path/joint.train \
    --data_txt_path_val ./datasets/data_path/mot17.train \
    --epoch 50 \
    --device 'cuda' \
    --nheads 8  \
    --pretrained ${PRETRAIN} \
    --loss_schedule \
    --giou_loss_coef 4 --final_weight 0.1 \
    #sampler_steps 5 9 15
   
    #| tee ${EXP_DIR}/output.log

python3  util/learning_curve.py ${EXP_DIR}


#evaluation phase
python3 eval_visem.py \
     --meta_arch motr \
     --dataset_file e2e_visem \
     --with_box_refine \
     --lr_drop 100 \
     --lr 2e-4 \
     --lr_backbone 2e-5 \
     --pretrained ${EXP_DIR}/best_weight.pth \
     --output_dir ${EXP_DIR}  \
     --batch_size 1 \
     --sample_mode 'random_interval' \
     --sample_interval 10 \
     --sampler_steps 50 90 120 \
     --sampler_lengths 2 3 4 5 \
     --update_query_pos \
     --merger_dropout 0 \
     --dropout 0 \
     --random_drop 0.1 \
     --fp_ratio 0.3 \
     --query_interaction_layer 'QIM' \
     --extra_track_attn \
     --data_txt_path_train ./datasets/data_path/joint.train \
     --data_txt_path_val ./datasets/data_path/mot17.train \
     --resume ${EXP_DIR}/best_weight.pth  \
     --nheads 8 \

