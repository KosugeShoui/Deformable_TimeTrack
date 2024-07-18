import argparse
import matplotlib.pyplot as plt
import json
import os

def main(exp_name):
    exp_name_log = os.path.join(exp_name, 'log.txt')
    
    with open(exp_name_log, 'r') as file:
        lines = file.readlines()

    json_txt = lines[0]
    data = json.loads(json_txt)

    train_loss_ce = data["train_loss_ce"]
    train_loss_bbox = data["train_loss_bbox"]
    train_loss_giou = data["train_loss_giou"]

    w1, w2, w3 = 2, 5, 10
    train_loss_ce_list = []
    train_loss_bbox_list = []
    train_loss_giou_list = []

    for line in lines:
        train_loss_dict = json.loads(line)
        train_loss_ce_list.append(train_loss_dict['train_loss_ce'])
        train_loss_bbox_list.append(train_loss_dict['train_loss_bbox'])
        train_loss_giou_list.append(train_loss_dict['train_loss_giou'])

    plt.plot(train_loss_ce_list, label='Focal Class Loss (λ = {})'.format(w1))
    plt.plot(train_loss_bbox_list, label='Bounding Box Loss (λ = {})'.format(w2))
    #plt.plot(train_loss_giou_list, label='Giou Loss (λ = {})'.format(w3))
    plt.plot(train_loss_giou_list, label='Giou Loss Schedule')
    plt.xlabel('epoch num')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(os.path.join(exp_name, 'learning_curve.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process experiment name')
    parser.add_argument('exp_name', type=str, help='experiment name')
    args = parser.parse_args()
    main(args.exp_name)
