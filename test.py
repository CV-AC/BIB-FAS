import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse, os
import numpy as np
from datasets import get_datasets
from datasets.dataset import FaceDataset
from networks.base_model import fas_base_model
from utils import performances_val,protocol_decoder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--protocol', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--val_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--debug_subset_size', type=int, default=None)
    return parser.parse_args()

def test_model(args):
    # transforms
    normalizer = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize((args.val_size + 16, args.val_size + 16)),
        transforms.CenterCrop((args.val_size, args.val_size)),
        transforms.ToTensor(), normalizer
    ])

    # dataset and loader
    _, data_name_list_test = protocol_decoder(args.protocol)
    test_set = get_datasets(args.data_dir, FaceDataset, train=False, protocol=args.protocol,
                            transform=test_transform, debug_subset_size=args.debug_subset_size)
    test_loader = DataLoader(test_set[data_name_list_test[0]], batch_size=args.batch_size,
                             shuffle=False, num_workers=4)

    # model
    model = fas_base_model(pretrained=False).cuda()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # test
    scores_list = []
    with torch.no_grad():
        for sample_batched in test_loader:
            image_x = sample_batched["image_x_1"].cuda()
            label = sample_batched["label"].cuda()
            logit = model(image_x)

            for i in range(len(logit)):
                scores_list.append(f"{logit.squeeze()[i].item()} {label[i].item()}\n")

    # metric
    score_file = "temp_score.txt"
    with open(score_file, "w") as f:
        f.writelines(scores_list)

    acc, fpr, frr, hter, auc, err, tpr = performances_val(score_file)
    print(f"Test Results:\nACC={acc:.4f}, HTER={hter:.4f}, AUC={auc:.4f}, TPR={tpr:.4f}")

if __name__ == "__main__":

    args = parse_args()
    test_model(args)
