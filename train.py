import argparse
import time
import numpy as np
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torchvision import transforms
from loss import CenterLoss_mmd
from datasets import get_datasets
from datasets.dataset import FaceDataset, DEVICE_INFOS
from networks.base_model import fas_base_model
from utils import *
from loss import UCircle

torch.backends.cudnn.benchmark = True


# Step learning-rate policy
def adj_lr(epoch,args,optimizer):
    base_lr =args.base_lr
    warm_up_epoch = args.wp_epoch
    decay_epoch = args.decay_epoch

    if epoch+1 <=warm_up_epoch:
        cur_lr = base_lr/((warm_up_epoch - epoch)+1)
    elif epoch+1 <= decay_epoch:
        cur_lr = base_lr
    elif epoch+1 <= 40:
        cur_lr = base_lr*0.1
    elif epoch + 1 <= 60:
        cur_lr = base_lr*0.01
    else:
        cur_lr = base_lr*0.001

    optimizer.param_groups[0]['lr'] = cur_lr  # Classifier
    optimizer.param_groups[1]['lr'] = 0.65*cur_lr  #Encoder

    return cur_lr,optimizer

def log_f(f, console=True):
    def log(msg):
        with open(f, 'a') as file:
            file.write(msg)
            file.write('\n')
        if console:
            print(msg)
    return log


def main(args):

    # make dirs
    model_root_path = os.path.join(args.result_path, args.result_name, "model")
    check_folder(model_root_path)
    score_root_path = os.path.join(args.result_path, args.result_name, "score")
    check_folder(score_root_path)
    csv_root_path = os.path.join(args.result_path, args.result_name, "csv")
    check_folder(csv_root_path)
    log_path = os.path.join(args.result_path, args.result_name, "log.txt")
    print = log_f(log_path)

    # Pre-define the normalizer
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform_list_weak = [
        # transforms.ColorJitter(
        #     brightness=(0.8, 1.2),
        #     contrast=(0.8, 1.2),
        #     saturation=(0.8, 1.2),
        #     hue=(-0.03, 0.03)
        # ),
        transforms.RandomRotation(degrees=(-60, 60)),
        transforms.RandomResizedCrop((args.img_size,args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer,
    ]

    train_transform_weak = transforms.Compose(train_transform_list_weak)

    test_transform = transforms.Compose([
        transforms.Resize((args.val_size+16, args.val_size+16)),
        transforms.CenterCrop((args.val_size, args.val_size)),
        transforms.ToTensor(),
        normalizer
    ])

    # protocol_decoder: return the standard name of datasets used for training and testing, respectively.
    data_name_list_train, data_name_list_test = protocol_decoder(args.protocol)

    train_set_live = get_datasets(args.data_dir, FaceDataset, train=True, protocol=args.protocol,transform=train_transform_weak, debug_subset_size=args.debug_subset_size,is_live=1)
    train_loader_live = DataLoader(train_set_live, batch_size=args.batch_size, shuffle=True, num_workers=8,drop_last=True)

    train_set_spoofing = get_datasets(args.data_dir, FaceDataset, train=True, protocol=args.protocol,transform=train_transform_weak, debug_subset_size=args.debug_subset_size,is_live=0)
    train_loader_spoofing = DataLoader(train_set_spoofing, batch_size=args.batch_size, shuffle=True, num_workers=8,drop_last=True)

    test_set = get_datasets(args.data_dir, FaceDataset, train=False, protocol=args.protocol,transform=test_transform, debug_subset_size=args.debug_subset_size)
    test_loader = DataLoader(test_set[data_name_list_test[0]], batch_size=args.batch_size, shuffle=False, num_workers=8,drop_last=False)

    assert len(train_set_spoofing) > len(train_set_live)

    print('live samples in traning set:{}'.format(len(train_set_live)))
    print('spoofing samples in traning set:{}'.format(len(train_set_spoofing)))
    print('Total sample of training set:{}'.format(len(train_set_live)+len(train_set_spoofing)))
    print('Total sample of testing set:{}'.format(len(test_set[data_name_list_test[0]])))

    live_cls_list = []
    spoof_cls_list = []
    for dataset in data_name_list_train:
        live_cls_list += DEVICE_INFOS[dataset]['live']
        spoof_cls_list += DEVICE_INFOS[dataset]['spoof']

    model = fas_base_model(pretrained=True)
    model = model.cuda()
    # def loss
    ce_loss = nn.BCELoss().cuda()
    circle_loss = UCircle().cuda()

    # def optimizer
    ignored_params = list(map(id, model.encoder.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())

    optimizer = torch.optim.AdamW([
        {'params': base_params},
        {'params': model.encoder.parameters()}],
        lr=args.base_lr,
        weight_decay=args.weight_decay,)

    # metrics
    eva = {
        "best_epoch": -1,
        "best_HTER": 100,
        "best_auc": -100
    }

    for epoch in range(args.start_epoch, args.num_epochs):

        ce_loss_record = AvgrageMeter()
        circle_loss_record = AvgrageMeter()
        dcls_loss_record = AvgrageMeter()
        kl_loss_record = AvgrageMeter()

        ########################### train ###########################
        model.train()

        lr, optimizer = adj_lr(epoch,args,optimizer)

        iterator1 = iter(train_loader_live)
        iterator2 = iter(train_loader_spoofing)

        i = 0

        while True: # Balance loader for live/spoof samples
            i+=1
            correct_0 = 0
            total_0 = 1

            try:
                data_live = next(iterator1)
            except StopIteration:
                iterator1 = iter(train_loader_live)
                data_live = next(iterator1)
            try:
                data_spoofing = next(iterator2)
            except StopIteration:
                break

            image_x_live = torch.cat((data_live["image_x_1"].cuda(),data_live["image_x_2"].cuda()),dim=0)
            image_x_spoof = torch.cat((data_spoofing["image_x_1"].cuda(),data_spoofing["image_x_2"].cuda()),dim=0)

            image_x = torch.cat((image_x_live,image_x_spoof),dim=0)

            label = torch.cat((data_live["label"].cuda(),data_live["label"].cuda(),data_spoofing["label"].cuda(),data_spoofing["label"].cuda()),dim=0)

            feat = model(image_x)

            label = label.float()

            logit_0 = model.fc0(feat[0]).squeeze(-1)

            loss_ce = ce_loss(logit_0, label)


            loss_circle = circle_loss(feat[1],label)

            md_loss = 1/feat[2]

            d_cls_loss = feat[3] + feat[4]

            predicted_0 = (logit_0 > 0.5).float()
            total_0 += len(logit_0)
            correct_0 += predicted_0.cpu().eq(label.cpu()).sum().item()

            ce_loss_record.update(loss_ce.data.item(), len(logit_0))

            circle_loss_record.update(loss_circle.data.item(), len(logit_0))

            dcls_loss_record.update(d_cls_loss.data.item(), len(logit_0))
            kl_loss_record.update(md_loss.data.item(), len(logit_0))


            loss_all = loss_ce + 0.2*loss_circle + 0.5*d_cls_loss + 0.01*md_loss

            model.zero_grad()
            loss_all.backward()
            optimizer.step()

            log_info = "epoch:{:d}, mini-batch:{:d}, lr={:.4f}, ce_loss_0={:.4f}, CIR_loss_0 = {:.4f},dcls_loss_0 = {:.4f},kl_loss_0 = {:.4f}, ACC_0={:.4f}".format(
                epoch + 1, i + 1, lr, ce_loss_record.avg,circle_loss_record.avg,dcls_loss_record.avg,kl_loss_record.avg, 100. * correct_0 / total_0)

            if i % args.print_freq == args.print_freq - 1:
                print(log_info)

        # whole epoch average
        print("epoch:{:d}, Train: lr={:f}, CE Loss={:.4f}, CIR Loss = {:.4f}, ".format(
            epoch + 1, lr, ce_loss_record.avg,circle_loss_record.avg))


        if epoch % args.eval_freq == 0:

            score_path = os.path.join(score_root_path, "epoch_{}".format(epoch + 1))
            check_folder(score_path)

            model.eval()
            with torch.no_grad():
                start_time = time.time()
                scores_list = []
                for i, sample_batched in enumerate(test_loader):
                    image_x = sample_batched["image_x_1"].cuda()
                    live_label = sample_batched["label"].cuda()

                    logit = model(image_x)


                    for i in range(len(logit)):
                        scores_list.append("{} {}\n".format(logit.squeeze()[i].item(), live_label[i].item()))

            map_score_val_filename = os.path.join(score_path, "{}_score.txt".format(data_name_list_test[0]))
            print("score: write test scores to {}".format(map_score_val_filename))
            with open(map_score_val_filename, 'w') as file:
                file.writelines(scores_list)

            test_ACC, fpr, FRR, HTER, auc_test, test_err, tpr = performances_val(map_score_val_filename)
            print("## {} score:".format(data_name_list_test[0]))
            print(
                "epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}, TPR={:.4f}".format(
                    epoch + 1, test_ACC, HTER, auc_test, test_err, test_ACC, tpr))
            print("test phase cost {:.4f}s".format(time.time() - start_time))

            if auc_test - HTER >= eva["best_auc"] - eva["best_HTER"]:
                eva["best_auc"] = auc_test
                eva["best_HTER"] = HTER
                eva["tpr95"] = tpr
                eva["best_epoch"] = epoch + 1
                model_path = os.path.join(model_root_path, "{}_p{}_best.pth".format(args.model_name, args.protocol))
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'eva': (HTER, auc_test)
                }, model_path)
                print("Model saved to {}".format(model_path))

            print("[Best result] epoch:{}, HTER={:.4f}, AUC={:.4f}".format(eva["best_epoch"], eva["best_HTER"],
                                                                           eva["best_auc"]))

            model_path = os.path.join(model_root_path, "{}_p{}_recent.pth".format(args.model_name, args.protocol))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'eva': (HTER, auc_test)
            }, model_path)
            print("Model saved to {}".format(model_path))

    epochs_saved = np.array([int(dir.replace("epoch_", "")) for dir in os.listdir(score_root_path)])
    epochs_saved = np.sort(epochs_saved)
    last_n_epochs = epochs_saved[::-1][:10]

    HTERs, AUROCs, TPRs = [], [], []
    for epoch in last_n_epochs:
        map_score_val_filename = os.path.join(score_root_path, "epoch_{}".format(epoch),
                                              "{}_score.txt".format(data_name_list_test[0]))
        test_ACC, fpr, FRR, HTER, auc_test, test_err, tpr = performances_val(map_score_val_filename)
        HTERs.append(HTER)
        AUROCs.append(auc_test)
        TPRs.append(tpr)
        print("## {} score:".format(data_name_list_test[0]))
        print(
            "epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}, TPR={:.4f}".format(
                epoch + 1, test_ACC, HTER, auc_test, test_err, test_ACC, tpr))

    os.makedirs('summary', exist_ok=True)
    file = open(f"summary/{args.result_name_no_protocol}.txt", "a")
    L = [f"{args.summary}\t\t{eva['best_epoch']}\t{eva['best_HTER'] * 100:.2f}\t{eva['best_auc'] * 100:.2f}" +
         f"\t{np.array(HTERs).mean() * 100:.2f}\t{np.array(HTERs).std() * 100:.2f}\t{np.array(AUROCs).mean() * 100:.2f}\t{np.array(AUROCs).std() * 100:.2f}\t" +
         f"{np.array(TPRs).mean() * 100:.2f}\t{np.array(TPRs).std() * 100:.2f}\n"]

    print('upon convergence mean HTER:')
    print(str(np.array(HTERs).mean() * 100))
    print('upon convergence mean AUC:')
    print(str(np.array(AUROCs).mean() * 100))
    print('upon convergence mean TPRs:')
    print(str(np.array(TPRs).mean() * 100))

    file.writelines(L)
    file.close()

def parse_args():
    parser = argparse.ArgumentParser()
    # build dirs
    parser.add_argument('--data_dir', type=str, default="/datasets/FAS", help='YOUR_Data_Dir')
    parser.add_argument('--result_path', type=str, default='results/', help='root result directory')
    parser.add_argument('--protocol', type=str, default="I_C_M_to_O",help='O_C_I_to_M, O_M_I_to_C, O_C_M_to_I, I_C_M_to_O, O_to_O')
    # training settings
    parser.add_argument('--model_name', type=str, default="r18_bib_fas", help='model_name')
    parser.add_argument('--eval_freq', type=int, default=1, help='evaluation frequency (per x epoch)')
    parser.add_argument('--img_size', type=int, default=256, help='img train size')
    parser.add_argument('--val_size', type=int, default=256, help='img val size')

    parser.add_argument('--pretrain', type=str, default='imagenet', help='imagenet')

    parser.add_argument('--batch_size', type=int, default=48, help='batch size')
    parser.add_argument('--train_rotation', type=str2bool, default=True, help='train_rotation')

    parser.add_argument('--base_lr', type=float, default=0.0003, help='base learning rate')
    parser.add_argument('--wp_epoch', type=int, default=5, help='learning_rate_peak_epoch')
    parser.add_argument('--decay_epoch', type=int, default=20, help='learning_rate_decay_epoch')

    # Important!! Change the seed may case up to 50% performance variation!!
    parser.add_argument('--seed', type=int, default=1223, help='batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--num_epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0008)
    # debug
    parser.add_argument('--debug_subset_size', type=int, default=None)
    return parser.parse_args()

def str2bool(x):
    return x.lower() in ('true')

if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    pretrain_alias = {
        "imagenet": "img",
    }
    args.result_name_no_protocol = f"pre({pretrain_alias[args.pretrain]})_bsz({args.batch_size})_rot({args.train_rotation})"

    args.result_name = f"{args.protocol}_" + args.result_name_no_protocol

    info_list = [args.protocol, args.batch_size, args.model_name]

    args.summary = "\t".join([str(info) for info in info_list])
    print(args.result_name)
    print(args.summary)

    main(args=args)
