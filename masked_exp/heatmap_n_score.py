#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/11/22 11:50
# @Author  : CHEN Weiling
# @File    : temp.py
# @Software: PyCharm
# @Comments:

import numpy as np
import matplotlib.pyplot as plt
import json, argparse, os

if __name__ == '__main__':
    root_dir = os.path.abspath(os.path.join(__file__, '../..'))
    parser = argparse.ArgumentParser(description='show heatmap and captions')
    parser.add_argument('--dataset', type=str, help='dataset to generate caption embeddings')
    parser.add_argument('--vid_name', type=str, default='', help='video name without suffix')
    args = parser.parse_args()

    # get captions
    if "shanghai" in args.dataset:
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/Shanghai/RTFM_train_caption/test_captions.txt"
        ds_name = "Shanghai"
    elif "ucf" in args.dataset:
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/Crime/RTFM_train_caption/all_captions.txt"
        ds_name = "Crime"
    elif "ped2" in args.dataset:
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/UCSDped2/RTFM_train_caption/test_captions.txt"
        ds_name = "UCSDped2"
    elif "violence" in args.dataset:
        caption_path = "/home/acsguser/Codes/SwinBERT/datasets/Violence/RTFM_train_caption/all_captions.txt"
        ds_name = "Violence"
    else:
        raise ValueError("dataset not supported")

    captions = []
    with open(caption_path) as f:
        for line in f:
            cap = json.loads(line)
            key = []
            for k in cap:
                key.append(k)
            assert len(key) == 1
            key = key[0]
            if args.vid_name in key:
                captions = cap[key]
                break

    if len(captions) == 0:
        raise ValueError("no captions found")

    # get anomaly scores
    p = os.path.join(root_dir, "output", ds_name + "_" + args.vid_name + '_heatmap.npy')
    print(p)
    feat = np.load(p)  # N,T,1 = (15,28,1)
    feat = np.squeeze(feat, 2)  # N,T = (15,28)
    feat = np.transpose(feat, (1, 0))  # transpose to (T,N)

    assert len(captions) == feat.shape[0]

    for i in range(feat.shape[0]):
        tmp = ['{:.2f}'.format(x) for x in feat[i]]
        print(str(i) + "\t" + "\t".join(tmp))
        print(str(i) + "\t" + "\t".join(captions[i].split()))
