import os

import PIL.Image
import torch
import pandas as pd
import cv2
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import math
from glob import glob
import re
from utils.rotate_crop import crop_rotated_rectangle, inside_rect, vis_rotcrop
import torchvision.transforms.functional as tf
import matplotlib.pyplot as plt

from ylib.scipy_misc import imread, imsave
from .meta import DEVICE_INFOS

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)


def crop_face_from_scene(image, bbox, scale):
    x1, y1, x2, y2 = [float(ele) for ele in bbox]
    h = y2 - y1
    w = x2 - x1
    # y2=y1+w
    # x2=x1+h
    y_mid = (y1 + y2) / 2.0
    x_mid = (x1 + x2) / 2.0
    h_img, w_img = image.shape[0], image.shape[1]
    w_scale = scale * w
    h_scale = scale * h
    y1 = y_mid - h_scale / 2.0
    x1 = x_mid - w_scale / 2.0
    y2 = y_mid + h_scale / 2.0
    x2 = x_mid + w_scale / 2.0
    y1 = max(math.floor(y1), 0)
    x1 = max(math.floor(x1), 0)
    y2 = min(math.floor(y2), h_img)
    x2 = min(math.floor(x2), w_img)
    region = image[y1:y2, x1:x2]
    return region


class FaceDataset(Dataset):

    def __init__(self, dataset_name, root_dir, split='train', transform=None, UUID=-1, is_live=1):
        # self.landmarks_frame = pd.read_csv(info_list, delimiter=",", header=None)
        self.split = split
        self.video_list = os.listdir(root_dir)
        self.is_live = is_live

        if self.split == 'train':
            self.video_list = self.filter_video_list(self.video_list, self.is_live)

        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transform = transform
        self.UUID = UUID

    def __len__(self):
        return len(self.video_list)

    def filter_video_list(self, video_list, is_live):
        new_video_list = []
        for video_name in video_list:
            spoofing_label = int('live' in video_name)
            if spoofing_label == is_live:
                new_video_list.append(video_name)
        return new_video_list

    def get_client_from_video_name(self, video_name):
        # used to find the clients (identity) for images in different datasets, by searching the string name.

        if 'msu' in self.dataset_name.lower() or 'replay' in self.dataset_name.lower():
            match = re.findall('client(\d\d\d)', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'oulu' in self.dataset_name.lower():
            match = re.findall('(\d+)_\d$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'casia' in self.dataset_name.lower():
            match = re.findall('_(\d+)_[H|N][R|M]_\d$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        elif 'celeba' in self.dataset_name.lower():
            match = re.findall('_(\d+)$', video_name)
            if len(match) > 0:
                client_id = match[0]
            else:
                raise RuntimeError('no client')
        else:
            raise RuntimeError("no dataset found")
        return client_id

    def __getitem__(self, idx):

        video_name = self.video_list[idx]

        spoofing_label = int('live' in video_name)

        client_id = self.get_client_from_video_name(video_name)

        image_dir = os.path.join(self.root_dir, video_name)

        image_x = self.sample_image(image_dir)

        if self.split == 'train':
            image_x_1 = self.transform(PIL.Image.fromarray(image_x))
            image_x_2 = self.transform(PIL.Image.fromarray(image_x))

        else:
            image_x_1 = self.transform(PIL.Image.fromarray(image_x))
            image_x_2 = image_x_1

        sample = {"image_x_1": np.array(image_x_1),
                  "image_x_2": np.array(image_x_2),
                  "label": spoofing_label,
                  "UUID": self.UUID,
                  'video': video_name,
                  'client_id': client_id}
        return sample

    def sample_image(self, image_dir):

        frames = glob(os.path.join(image_dir, "crop_*.jpg"))

        frames_total = len(frames)
        if frames_total == 0:
            raise RuntimeError(f"{image_dir}")

        for temp in range(500):
            if temp > 200:
                image_id = int(re.findall('_(\d+).jpg', frames[0])[0]) // 5
                # print(f"No {image_path} or {info_path} found, use backup id")
            else:
                image_id = np.random.randint(0, frames_total)

            image_name = f"crop_{image_id * 5:04d}.jpg"
            # image_name = f"square400_{image_id*5:04d}.jpg"

            image_path = os.path.join(image_dir, image_name)

            if os.path.exists(image_path):
                break

        image = imread(image_path)

        return image



