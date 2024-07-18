#!/usr/bin/env bash
#!sh track_exps/mot_train_val_mota.sh

GROUNDTRUTH=mot/train

OUTPUT_DIR=output/exp20

RESULTS=${OUTPUT_DIR}/val/tracks
GT_TYPE=_val_half
THRESHOLD=-1


python3 -m torch.distributed.launch \
--nproc_per_node=1 \
--use_env main_track.py  \
--output_dir ${OUTPUT_DIR} \
--dataset_file mot \
--coco_path mot \
--batch_size 4  \
--with_box_refine  \
--num_queries 500 \
--set_cost_class 2 \
--set_cost_bbox 3 \
--set_cost_giou 4 \
--epochs 10 \
--lr_drop 100 \
--device cuda:1
#--frozen_weights mot/619mot17_mot17.pth



python3 main_track.py  \
--output_dir ${OUTPUT_DIR} \
--dataset_file mot \
--coco_path mot \
--batch_size 1 \
--resume ${OUTPUT_DIR}/checkpoint.pth \
--eval \
--with_box_refine \
--num_queries 500


python3 track_tools/eval_motchallenge.py \
--groundtruths ${GROUNDTRUTH} \
--tests ${RESULTS} \
--gt_type ${GT_TYPE} \
--eval_official \
--score_threshold ${THRESHOLD}

#visualize curve phase
python3 learning_curve.py  ${OUTPUT_DIR}