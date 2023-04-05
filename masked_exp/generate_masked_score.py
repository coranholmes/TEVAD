#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/11/22 15:14
# @Author  : CHEN Weiling
# @File    : generate_masked_score.py
# @Software: PyCharm
# @Comments: for a video, generate the masked score


from torch.utils.data import DataLoader
import torch.optim as optim
from dataset_masked import Dataset_masked
from model import Model
import option
from config import *
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from utils import *


def test_masked(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        res = []

        for i, (input, text) in enumerate(dataloader):  # test set has 199 videos
            input = input.to(device)
            input = input.permute(0, 2, 1, 3)
            text = text.to(device)
            text = text.permute(0, 2, 1, 3)
            # input.shape = (1,10,T,2048); T clips, each clip has 16frames, each frame has 10 crops
            # https://github.com/tianyu0207/RTFM/issues/51
            # 使用时可以把10那一维拉平变成(1, 10*T, 2048), 中间那一维就是visual features再和caption进行concat操作
            score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits, \
            scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(input,
                                                                           text)  # 注意这里的score_abnormal和score_normal是一维的，是每一个video的一个分数，而logits则是一个T维的vector给每一个snippet都打了分
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits

            res.append(list(sig.cpu().detach().numpy()))
        return res


if __name__ == '__main__':
    root_dir = os.path.abspath(os.path.join(__file__, '../..'))
    args = option.parser.parse_args()
    config = Config(args)
    seed_everything(args.seed)
    test_loader = DataLoader(Dataset_masked(args, test_mode=True),
                             batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=False)

    model = Model(args)
    if args.pretrained_ckpt is not None:
        print("Loading pretrained model " + args.pretrained_ckpt)
        model.load_state_dict(torch.load(args.pretrained_ckpt))
    else:
        if "shanghai" in args.dataset:
            model.load_state_dict(
                torch.load(os.path.join(root_dir, 'ckpt/my_best/shanghai_v2-both-text_agg-add-1-1-extra_loss-595-i3d-best.pkl')))
        elif "ped2" in args.dataset:
            model.load_state_dict(torch.load(os.path.join(root_dir, 'ckpt/my_best/ped2-both-text_agg-add-1-1-extra_loss-755-4869-i3d.pkl')))
        elif "violence" in args.dataset:
            model.load_state_dict(
                torch.load(os.path.join(root_dir, 'ckpt/my_best/violence-both-text_agg-add-1-1-extra_loss-445-4869-BEST.pkl')))
        elif "ucf" in args.dataset:
            model.load_state_dict(
                torch.load(os.path.join(root_dir, 'ckpt/my_best/ucf-both-text_agg-concat-1-1-extra_loss-680-2333-BEST.pkl')))
        else:
            raise NotImplementedError

    for name, value in model.named_parameters():
        print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists(os.path.join(root_dir, 'ckpt')):
        os.makedirs(os.path.join(root_dir, 'ckpt'))

    optimizer = optim.Adam(model.parameters(),
                           lr=config.lr[0], weight_decay=0.005)

    best_AUC = -1
    output_path = os.path.join(root_dir, 'output')  # put your own path here
    res = test_masked(test_loader, model, args, device)

    vid_name = args.test_rgb_list.split('/')[-1][:-4]
    np.save(os.path.join(output_path, vid_name + '_heatmap.npy'), np.array(res))
    feat = np.array(res)
    feat = np.squeeze(feat, 2)
    # transpose to (T,N)
    feat = np.transpose(feat, (1, 0))
    fig = plt.figure(figsize=(15, 15))
    ax = sns.heatmap(feat, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=.5)
    plt.savefig(os.path.join(root_dir, 'figures/heatmap/' + vid_name + '_heatmap.png'))