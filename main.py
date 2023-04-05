from torch.utils.data import DataLoader
import torch.optim as optim
from model import Model
from dataset import Dataset
from train import train
from test_10crop import test
import option
from tqdm import tqdm
from utils import *
from config import *

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    seed_everything(args.seed)
    text_opt = "text_agg" if args.aggregate_text else "no_text_agg"
    extra_loss_opt = "extra_loss" if args.extra_loss else "no_loss"
    if args.emb_folder == "":
        sb_pt_name = "vatex"
    else:
        sb_pt_name = args.emb_folder[11:]  # sent_emb_n_XXX
    print("Using SwinBERT pre-trained model: ", sb_pt_name)
    viz_name = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(args.dataset, args.feature_group, text_opt, args.fusion, args.normal_weight, args.abnormal_weight, extra_loss_opt, args.alpha, sb_pt_name)
    viz = Visualizer(env=viz_name, use_incoming_socket=False)
    # Dataloader for normal videos
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator(device='cuda'))
    # Dataloader for abnormal videos
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True, generator=torch.Generator(device='cuda'))
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False, generator=torch.Generator(device='cuda'))

    model = Model(args)
    if args.pretrained_ckpt is not None:
        print("Loading pretrained model " + args.pretrained_model)
        model.load_state_dict(torch.load(args.pretrained_model))
    # for name, value in model.named_parameters():
    #     print(name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.Adam(model.parameters(),
                            lr=config.lr[0], weight_decay=0.005)  # default lr=0.001


    test_info = {"epoch": [], "test_AUC": [], "test_AP": []}
    best_AUC, best_ap = -1, -1
    best_epoch = -1
    output_path = 'output'   # put your own path here
    # auc = test(test_loader, model, args, viz, device)

    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,  # 总的项目数
            dynamic_ncols=True  # 自动调整进度条宽度
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        if (step - 1) % len(train_nloader) == 0:  # when step=1, create the iter
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:  # when step=1, create the iter
            loadera_iter = iter(train_aloader)

        train(loadern_iter, loadera_iter, model, args, optimizer, viz, device)

        if step % 5 == 0 and step > 50:

            auc, ap = test(test_loader, model, args, viz, device)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)
            test_info["test_AP"].append(ap)

            if "violence" in args.dataset:  # use AUPR as the metric for violence dataset
                if test_info["test_AP"][-1] > best_ap:
                    best_ap = test_info["test_AP"][-1]
                    best_epoch = step
                    torch.save(model.state_dict(), './ckpt/' + '{}-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format(args.dataset, args.feature_group, text_opt, args.fusion, args.alpha, extra_loss_opt, step, args.seed, sb_pt_name))
                    save_best_record(test_info, os.path.join(output_path, '{}-{}-{}-{}-{}-{}-{}-{}-AP.txt'.format(args.dataset, args.feature_group, text_opt, args.fusion, args.alpha, extra_loss_opt, step, sb_pt_name)), "test_AP")
                APs = test_info["test_AP"]
                APs_mean, APs_median, APs_std, APs_max, APs_min = np.mean(APs), np.median(APs), np.std(APs), np.max(APs), np.min(APs)
                print("std\tmean\tmedian\tmin\tmax\tAP")
                print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(APs_std * 100, APs_mean * 100, APs_median * 100,
                                                                        APs_min * 100, APs_max * 100))
            else:  # user AUROC as the metric for other datasets
                if test_info["test_AUC"][-1] > best_AUC:
                    best_AUC = test_info["test_AUC"][-1]
                    best_epoch = step
                    torch.save(model.state_dict(), './ckpt/' + '{}-{}-{}-{}-{}-{}-{}-{}-{}.pkl'.format(args.dataset, args.feature_group, text_opt, args.fusion, args.alpha, extra_loss_opt, step, args.seed, sb_pt_name))
                    save_best_record(test_info, os.path.join(output_path, '{}-{}-{}-{}-{}-{}-{}-{}-AUC.txt'.format(args.dataset, args.feature_group, text_opt, args.fusion, args.alpha, extra_loss_opt, step, sb_pt_name)), "test_AUC")

                AUCs = test_info["test_AUC"]
                AUCs_mean, AUCs_median, AUCs_std, AUCs_max, AUCs_min = np.mean(AUCs), np.median(AUCs), np.std(AUCs), np.max(AUCs), np.min(AUCs)
                print("std\tmean\tmedian\tmin\tmax\tAUC")
                print("{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}".format(AUCs_std * 100, AUCs_mean * 100, AUCs_median * 100,
                                                                      AUCs_min * 100, AUCs_max * 100))
    print("Best result:" + viz_name + "-" + str(best_epoch))
    torch.save(model.state_dict(), './ckpt/' + args.dataset + 'final.pkl')
