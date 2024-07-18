import sys
import os
import argparse
from multiprocessing import freeze_support
import TrackEval.trackeval as trackeval # noqa: E402

def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    return [line.strip() for line in data]

if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file', type=str, required=True, help='Path to ground truth text file')
    parser.add_argument('--pred_file', type=str, required=True, help='Path to prediction text file')
    args = parser.parse_args()

    gt_data = read_txt_file(args.gt_file)
    pred_data = read_txt_file(args.pred_file)

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = False
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    default_metrics_config = {'METRICS': ['HOTA'], 'THRESHOLD': 0.5}
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    dataset_config = {
        'GT_DATA': gt_data,
        'PRED_DATA': pred_data,
        'CLASSES_TO_EVAL': ['pedestrian'],
        'BENCHMARK': 'MOT17',
        'SPLIT_TO_EVAL': 'train',
        'DO_PREPROC': True
    }

    # Run code
    evaluator = trackeval.Evaluator(config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(config)]

    evaluator.evaluate([dataset], metrics)
