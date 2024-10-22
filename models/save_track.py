"""
Copyright (c) https://github.com/xingyizhou/CenterTrack
Modified by Peize Sun, Rufeng Zhang
"""
# coding: utf-8
import os
import json
import logging
from collections import defaultdict


def save_track(results, out_root, video_to_images, video_names, data_split='val'):
    assert out_root is not None
    out_dir = os.path.join(out_root, data_split)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # save it in standard mot format.
    track_dir = os.path.join(out_dir, "tracks")
    if not os.path.exists(track_dir):
        os.mkdir(track_dir)
    
    for video_id in video_to_images.keys():
        video_to_image_infos = video_to_images[video_id]
        video_name = video_names[video_id]
        file_path = os.path.join(track_dir, "{}.txt".format(video_name))
        f = open(file_path, "w")
        tracks = defaultdict(list)
        for image_info in video_to_image_infos:
            image_id, frame_id = image_info["image_id"], image_info["frame_id"]
            if image_id in results:
                result = results[image_id]
                for item in result:
                    #print(item.keys())
                    if not ("tracking_id" in item):
                        raise NotImplementedError
                    tracking_id = item["tracking_id"]
                    bbox = item["bbox"]
                    bbox = [bbox[0], bbox[1], bbox[2], bbox[3], item['score'], item['active'],item['class']]
                    tracks[tracking_id].append([frame_id] + bbox)

        rename_track_id = 0
        for track_id in sorted(tracks):
            rename_track_id += 1
            for t in tracks[track_id]:
                if t[6] > 0:
                    f.write("{},{},{:.2f},{:.2f},{:.2f},{:.2f},{},{:.3f},-1,-1\n".format(
                        t[0], rename_track_id, t[1], t[2], t[3] - t[1], t[4] - t[2], t[7],t[5]))
        f.close()
