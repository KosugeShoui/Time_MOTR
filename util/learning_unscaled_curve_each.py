import matplotlib.pyplot as plt 
import numpy as np
import json
import os
import argparse

def plot_combined_unscaled_loss(exp_name):
    exp_name_log = os.path.join(exp_name, 'log.txt')
    with open(exp_name_log, 'r') as file:
        lines = file.readlines()

    data = [json.loads(line) for line in lines]

    epochs = np.arange(1, len(data)+1)
    train_loss_ce_list = [entry['train_loss_ce_unscaled'] for entry in data]
    train_loss_bbox_list = [entry['train_loss_bbox_unscaled'] for entry in data]
    train_loss_giou_list = [entry['train_loss_giou_unscaled'] for entry in data]

    # 横に並べるための subplot を定義
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Focal Class Loss グラフ
    axs[0].plot(epochs, train_loss_ce_list, label='Focal Class Loss', color='blue')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Focal Class Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Bounding Box Loss グラフ
    axs[1].plot(epochs, train_loss_bbox_list, label='Bounding Box Loss', color='orange')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Bounding Box Loss')
    axs[1].legend()
    axs[1].grid(True)

    # Giou Loss グラフ
    axs[2].plot(epochs, train_loss_giou_list, label='Giou Loss', color='green')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].set_title('Giou Loss')
    axs[2].legend()
    axs[2].grid(True)

    # グラフ間のスペースを調整
    plt.tight_layout()

    # 保存して表示
    plt.savefig(os.path.join(exp_name, 'combined_loss_unscaled.png'))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot combined loss curves.')
    parser.add_argument('exp_name', type=str, help='Path to the log file containing loss data.')

    args = parser.parse_args()

    plot_combined_unscaled_loss(args.exp_name)
