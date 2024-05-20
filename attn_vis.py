import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def visualize(num , attn_data, frame_num):
    # 2次元目のデータを取り出す
    attn_data = attn_data.cpu().detach().numpy()
    second_dimension_data = attn_data
    
    min_attn = np.min(attn_data)
    max_attn = np.max(attn_data)
    
    normalized_attn_data = (attn_data - min_attn) / (max_attn - min_attn)
    
    original_width, original_height = 1920,1080
    
    if num >= 301:
        #print(list(normalized_attn_data))
        # Attentionマップを元の画像のサイズにリサイズ
        np.savetxt('save_{}.txt'.format(num),normalized_attn_data)
        

    # 16x16の行列に変形
    reshaped_tensor = normalized_attn_data.reshape(16, 16)
    plt.clf()

    # Matplotlibを使用して可視化する
    plt.imshow(reshaped_tensor, cmap='viridis',label='Attention Map')  # カラーマップはお好みで変更可能
    plt.colorbar()  # カラーバーを表示

    # 画像を保存
    savedir = "attn_vis"
    os.makedirs("attn_vis",exist_ok=True)
    plt.savefig(savedir + "/tensor_visualization_{}_frame_{}.png".format(num,frame_num))


