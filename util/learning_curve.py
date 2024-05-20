import argparse
import matplotlib.pyplot as plt
import json
import os
import numpy as np

def plot_learning_curve(exp_name):
    
    exp_name_log = os.path.join(exp_name, 'log.txt')
    with open(exp_name_log, 'r') as file:
        lines = file.readlines()

    data = [json.loads(line) for line in lines]

    epochs = np.arange(1, len(data)+1)
    train_loss_ce_list = [entry['train_loss'] for entry in data]
    #train_loss_bbox_list = [entry['train_loss_bbox'] for entry in data]
    #train_loss_giou_list = [entry['train_loss_giou'] for entry in data]
    
    plt.clf()

    plt.plot(epochs,train_loss_ce_list, label='train loss')
    #plt.plot(train_loss_bbox_list, label='Bounding Box Loss (λ = {})'.format(w2))
    #plt.plot(train_loss_giou_list, label='Giou Loss (λ = {})'.format(w3))
    #plt.plot(train_loss_giou_list, label='Giou Loss Schedule')
    plt.xlabel('epoch num')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(exp_name, 'learning_curve.png'))
    plt.show()
    
    #return train_loss_ce_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process experiment name')
    parser.add_argument('exp_name', type=str, help='experiment name')
    args = parser.parse_args()
    plot_learning_curve(args.exp_name)
