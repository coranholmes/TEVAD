import torch
import torch.nn as nn
import torch.nn.init as torch_init

torch.set_default_tensor_type('torch.cuda.FloatTensor')


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # train: g_x.shape=[640,32,256], which is F_c in paper

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # train: theta_x.shape=[640,32,256], F_c1 in paper
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # train: phi_x.shape=[640,256,32], F_c2 in paper

        f = torch.matmul(theta_x,
                         phi_x)  # M=(F_c1)(F_c2)^T, train: M.shape=[640,32,32], 32 is the no. of clips in each video
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)  # train: y.shape=[640,32,256]
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)  # train: W_y.shape=[640,512,32], F_c4=Conv1x1(MF_c3)
        z = W_y + x  # train: z.shape=[640,512,32], A skip connection is added, z is F_TSA

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class Aggregate(nn.Module):  # MTN
    def __init__(self, len_feature):
        super(Aggregate, self).__init__()
        bn = nn.BatchNorm1d
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=1, padding=1),
            nn.ReLU(),
            bn(int(len_feature/4))
            # nn.dropout(0.7)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=2, padding=2),
            nn.ReLU(),
            bn(int(len_feature/4))
            # nn.dropout(0.7)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=3,
                      stride=1, dilation=4, padding=4),
            nn.ReLU(),
            bn(int(len_feature/4))
            # nn.dropout(0.7),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=int(len_feature/4), kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.ReLU(),
            # nn.dropout(0.7),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv1d(in_channels=len_feature, out_channels=len_feature, kernel_size=3,
                      stride=1, padding=1, bias=False),  # should we keep the bias?
            nn.ReLU(),
            nn.BatchNorm1d(len_feature),
            # nn.dropout(0.7)
        )

        self.non_local = NONLocalBlock1D(int(len_feature/4), sub_sample=False, bn_layer=True)

    def forward(self, x):
        # x: (B, T, F)
        out = x.permute(0, 2, 1)  # train: out.shape=[640,2048,32]
        residual = out

        # The module on the left uses the pyramid dilated convolutions to capture the local consecutive snippets
        # dependency over different temporal scales
        out1 = self.conv_1(out)  # PDC1, train: out1.shape=[640,512,32]
        out2 = self.conv_2(out)  # PDC2, train: out2.shape=[640,512,32]
        out3 = self.conv_3(out)  # PDC3, train: out3.shape=[640,512,32]
        out_d = torch.cat((out1, out2, out3), dim=1)  # train: out3.shape=[640,1536,32]

        # The module on the right relies on a self-attention network to compute the global temporal correlations
        out = self.conv_4(out)  # train: out.shape=[640,512,32]
        out = self.non_local(out)  # TSA, train: out.shape=[640,512,32]

        out = torch.cat((out_d, out), dim=1)  # train: out.shape=[640,2048,32]
        out = self.conv_5(out)  # fuse all the features together, train: out.shape=[640,2048,32]
        out = out + residual
        out = out.permute(0, 2, 1)  # train: out.shape=[640,32,2048]
        # out: (B, T, 1)
        return out


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.fusion = args.fusion
        self.batch_size = args.batch_size
        self.feature_group = args.feature_group
        self.aggregate_text = args.aggregate_text
        self.num_segments = 32
        self.k_abn = self.num_segments // 10  # top k for abnormal snippets
        self.k_nor = self.num_segments // 10  # top k for normal snippets

        self.Aggregate = Aggregate(len_feature=args.feature_size)
        self.Aggregate_text = Aggregate(len_feature=args.emb_dim)
        if self.feature_group == 'both':
            if args.fusion == 'concat':
                self.fc1 = nn.Linear(args.feature_size + args.emb_dim, 512)
            elif args.fusion == 'add' or args.fusion == 'product':
                self.fc0 = nn.Linear(args.feature_size, args.emb_dim)
                self.fc1 = nn.Linear(args.emb_dim, 512)
            elif 'up' in args.fusion:
                self.fc_vis = nn.Linear(args.feature_size, args.feature_size + args.emb_dim)
                self.fc_text = nn.Linear(args.emb_dim, args.feature_size + args.emb_dim)
                self.fc1 = nn.Linear(args.feature_size + args.emb_dim, 512)
            else:
                raise ValueError('Unknown fusion method: {}'.format(args.fusion))
        elif self.feature_group == 'text':
            self.fc1 = nn.Linear(args.emb_dim, 512)
        else:
            self.fc1 = nn.Linear(args.feature_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.drop_out = nn.Dropout(0.7)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.apply(weight_init)

    def forward(self, inputs, text):

        k_abn = self.k_abn
        k_nor = self.k_nor

        out = inputs  # shape=[64,10,32,2048]
        bs, ncrops, t, f = out.size()  # t is no. of clips
        bs2, ncrops2, t2, f2 = text.size()

        out = out.view(-1, t, f)
        out2 = text.view(-1, t2, f2)

        out = self.Aggregate(out)  # train: out.shape=[640,32,2048]
        out = self.drop_out(out)  # train: out.shape=[640,32,2048]

        if self.aggregate_text:
            out2 = self.Aggregate_text(out2)  # train: out2.shape=[640,32,args.emb_dim]
            out2 = self.drop_out(out2)  # train: out2.shape=[640,32,args.emb_dim]

        # 在这里进行对齐维度的操作！！！
        if out.shape[1] < out2.shape[1]:  # out(vis)比out2(text)少帧
            # remove the last frame of out2
            out2 = out2[:, :(out.shape[1] - out2.shape[1]), :]
        elif out.shape[1] > out2.shape[1]:  # out(vis)总比out2(text)多1帧
            # padding out2 by repeating the last frame
            out2 = torch.cat((out2, out2[:, (out2.shape[1] - out.shape[1]):, :]), dim=1)
        t = out.shape[1]

        # concat visual features with text features here，
        if self.fusion == 'concat':
            if self.feature_group == 'both':
                out = torch.cat([out, out2], dim=2)  # train: out.shape=[64, 10, 32, 2048+args.emb_dim]
            elif self.feature_group == 'text':
                out, ncrops, f = out2, ncrops2, f2
        elif self.fusion == 'product':
            out = self.relu(self.fc0(out))  # vis feature reduces to dim=args.emb_dim
            out = self.drop_out(out)
            out = out * out2
        elif self.fusion == 'add':
            out = self.relu(self.fc0(out))  # vis feature reduces to dim=args.emb_dim
            out = self.drop_out(out)
            out = out + out2
        elif self.fusion == 'add_up':
            out = self.relu(self.fc_vis(out))
            out = self.drop_out(out)
            out2 = self.relu(self.fc_text(out2))
            out2 = self.drop_out(out2)
            out = out + out2
        else:
            raise ValueError('Unknown fusion method: {}'.format(self.fusion))

        features = out  # [640,32,f+f2]
        scores = self.relu(self.fc1(features))  # train: scores.shape=[640,32,512]
        scores = self.drop_out(scores)
        scores = self.relu(self.fc2(scores))  # train: scores.shape=[640,32,128]
        scores = self.drop_out(scores)
        scores = self.sigmoid(self.fc3(scores))  # train: scores.shape=[640,32,1]
        scores = scores.view(bs, ncrops, -1).mean(1)  # train: scores.shape=[64,10,32]对dim=1求平均后->[64,32]
        scores = scores.unsqueeze(dim=2)  # train: scores.shape=[64,32,1]

        normal_features = features[0:self.batch_size * ncrops]  # train: normal_features.shape=[320,32,2048]
        normal_scores = scores[0:self.batch_size]  # train: normal_scores.shape=[32,32,1]

        abnormal_features = features[self.batch_size * ncrops:]  # train: abnormal_features.shape=[320,32,2048]
        abnormal_scores = scores[self.batch_size:]  # train: abnormal_scores.shape=[32,32,1]

        feat_magnitudes = torch.norm(features, p=2,
                                     dim=2)  # train: feat_magnitudes.shape=[640,32], use l2 norm to compute the feature magnitude
        feat_magnitudes = feat_magnitudes.view(bs, ncrops, -1).mean(1)  # train: feat_magnitudes.shape=[64,32]
        nfea_magnitudes = feat_magnitudes[0:self.batch_size]  # train: shape=[32,32], normal feature magnitudes
        afea_magnitudes = feat_magnitudes[self.batch_size:]  # train: shape=[32,32], abnormal feature magnitudes
        n_size = nfea_magnitudes.shape[0]

        if nfea_magnitudes.shape[0] == 1:  # this is for inference, the batch size is 1
            afea_magnitudes = nfea_magnitudes
            abnormal_scores = normal_scores
            abnormal_features = normal_features

        #######  process abnormal videos -> select top3 feature magnitude  #######

        select_idx = torch.ones_like(nfea_magnitudes).cuda()
        select_idx = self.drop_out(select_idx)
        afea_magnitudes_drop = afea_magnitudes * select_idx
        idx_abn = torch.topk(afea_magnitudes_drop, k_abn, dim=1)[1]  # [0]为值, [1]为idx, train: shape=[32,3]
        idx_abn_feat = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_features.shape[2]])  # train: shape=[32,3,2048]

        abnormal_features = abnormal_features.view(n_size, ncrops, t, -1)  # train: shape=[32,10,32,2048]
        abnormal_features = abnormal_features.permute(1, 0, 2, 3)  # train: shape=[10,32,32,2048]

        total_select_abn_feature = torch.zeros(0)
        for abnormal_feature in abnormal_features:  # range(10)
            feat_select_abn = torch.gather(abnormal_feature, 1,
                                           idx_abn_feat)  # train: shape=[32,3,2048], top 3 features magnitude in abnormal bag
            total_select_abn_feature = torch.cat((total_select_abn_feature, feat_select_abn))

        idx_abn_score = idx_abn.unsqueeze(2).expand([-1, -1, abnormal_scores.shape[2]])  # train: shape=[32,3,1]
        score_abnormal = torch.mean(torch.gather(abnormal_scores, 1, idx_abn_score),
                                    dim=1)  # train: shape=[32,3,1]求mean后变为[32,1], top 3 scores in abnormal bag based on the top-3 magnitude

        ####### process normal videos -> select top3 feature magnitude #######

        select_idx_normal = torch.ones_like(nfea_magnitudes).cuda()
        select_idx_normal = self.drop_out(select_idx_normal)
        nfea_magnitudes_drop = nfea_magnitudes * select_idx_normal
        idx_normal = torch.topk(nfea_magnitudes_drop, k_nor, dim=1)[1]
        idx_normal_feat = idx_normal.unsqueeze(2).expand([-1, -1, normal_features.shape[2]])

        normal_features = normal_features.view(n_size, ncrops, t, -1)
        normal_features = normal_features.permute(1, 0, 2, 3)

        total_select_nor_feature = torch.zeros(0)
        for nor_fea in normal_features:
            feat_select_normal = torch.gather(nor_fea, 1,
                                              idx_normal_feat)  # top 3 features magnitude in normal bag (hard negative)
            total_select_nor_feature = torch.cat((total_select_nor_feature, feat_select_normal))

        idx_normal_score = idx_normal.unsqueeze(2).expand([-1, -1, normal_scores.shape[2]])
        score_normal = torch.mean(torch.gather(normal_scores, 1, idx_normal_score), dim=1)  # top 3 scores in normal bag

        feat_select_abn = total_select_abn_feature  # train: shape=[320,3,2048]
        feat_select_normal = total_select_nor_feature  # train: shape=[320,3,2048]

        # score_abnormal, score_normal (shape=[32,1]) are the score of a video, while scores (shape=[64,32,1]) are the score vector of snippets in a video
        # therefore, we use score_abnormal and score_normal during training and scores during inference
        return score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_select_abn, feat_select_abn, scores, feat_select_abn, feat_select_abn, feat_magnitudes
