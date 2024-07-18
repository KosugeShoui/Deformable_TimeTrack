#!/usr/bin/env bash

python3 main_track.py  \
--output_dir output \
--dataset_file mot \
--coco_path mot \
--batch_size 1 \
--resume mot/619mot17_mot17.pth \
--eval \
--with_box_refine \
--num_queries 500 \
--det_val \
--track_eval_split val