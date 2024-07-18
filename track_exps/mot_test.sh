#!/usr/bin/env bash

python3 main_track.py \
--output_dir output \
--dataset_file mot \
--coco_path mot \
--batch_size 1 \
--resume mot/619mot17_mot17.pth \
--track_eval_split test \
--eval --with_box_refine  \
--num_queries 500