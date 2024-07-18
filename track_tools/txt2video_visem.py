import os
import sys
import json
import cv2
import glob as gb
from colormap import colormap
import argparse


def txt2img(visual_path="visual_val_gt",exp_name='',video_num=''):
    print("Starting txt2img")

    valid_labels = {1}
    ignore_labels = {2, 7, 8, 12}

    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    color_list = colormap()

    gt_json_path = 'visem/annotations/val_half.json'
    img_path = 'visem/train/'
    show_video_names = [video_num]

    test_json_path = 'visem/annotations/test.json'
    test_img_path = 'visem/test/'
    
    if visual_path == "visual_test_predict_visem":
        img_path = test_img_path
        gt_json_path = test_json_path
    for show_video_name in show_video_names:
        img_dict = dict()
        
        if visual_path == "visual_val_gt":
            txt_path = 'visem/train/' + show_video_name + '/gt/gt_val_half.txt'
        elif visual_path == "visual_val_predict_visem":
            txt_path = exp_name + '/val/tracks/'+ show_video_name + '.txt'
        elif visual_path == "visual_test_predict_visem":
            txt_path = exp_name + '/test/tracks/'+ show_video_name + '.txt'
        else:
            raise NotImplementedError
        
        with open(gt_json_path, 'r') as f:
            gt_json = json.load(f)

        for ann in gt_json["images"]:
            file_name = ann['file_name']
            video_name = file_name.split('/')[0]
            #print(video_name)
            if video_name == show_video_name:
                img_dict[ann['frame_id']] = img_path + file_name


        txt_dict = dict()    
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                linelist = line.split(',')

                mark = int(float(linelist[6]))
                label = int(float(linelist[7]))
                vis_ratio = float(linelist[8])
                
                if visual_path == "visual_val_gt":
                    if mark == 0 or label not in valid_labels or label in ignore_labels or vis_ratio <= 0:
                        continue

                img_id = linelist[0]
                obj_id = linelist[1]
                bbox = [float(linelist[2]), float(linelist[3]), 
                        float(linelist[2]) + float(linelist[4]), 
                        float(linelist[3]) + float(linelist[5]), int(obj_id)]
                if int(img_id) in txt_dict:
                    txt_dict[int(img_id)].append(bbox)
                else:
                    txt_dict[int(img_id)] = list()
                    txt_dict[int(img_id)].append(bbox)

        for img_id in sorted(txt_dict.keys()):
            img = cv2.imread(img_dict[img_id])
            for bbox in txt_dict[img_id]:
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color_list[bbox[4]%79].tolist(), thickness=2)
                cv2.putText(img, "{}".format(int(bbox[4])), (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_list[bbox[4]%79].tolist(), 2)
            os.makedirs(visual_path + "/" + show_video_name + '/tracked_images',exist_ok = True)
            cv2.imwrite(visual_path + "/" + show_video_name + '/tracked_images/' + "{:0>6d}.png".format(img_id), img)
        print(show_video_name, "Done")
    print("txt2img Done")

        
def img2video(visual_path="visual_val_gt",video_num=''):
    print("Starting img2video")

    img_paths = gb.glob(visual_path + '/'+ video_num + '/tracked_images' + "/*.png") 
    fps = 45
    size = (640,480) 
    videowriter = cv2.VideoWriter(visual_path + '/' + video_num + '/output_video' + '_' + video_num + '.avi' ,cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)

    for img_path in sorted(img_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, size)
        videowriter.write(img)

    videowriter.release()
    print("img2video Done")

"""
if __name__ == '__main__':
    #visual_path="visual_val_predict_visem"
    visual_path="visual_test_predict_visem"
    if len(sys.argv) > 1:
        visual_path =sys.argv[1]
    
    #select exp name & video num 
    exp_name = 'output_visem/exp_0604_ep50_re'
    video_num = 38
    txt2img(visual_path,exp_name,str(video_num))
    img2video(visual_path,str(video_num))
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some paths and parameters.")
    parser.add_argument('--visual_path', nargs='?', default="visual_test_predict_visem", help="Path for visual files")
    parser.add_argument('--exp_name', required=True, help="Experiment name")
    parser.add_argument('--video_num', required=True, type=int, help="Video number")

    args = parser.parse_args()

    txt2img(args.visual_path, args.exp_name, str(args.video_num))
    img2video(args.visual_path, str(args.video_num))