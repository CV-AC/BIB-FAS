from .performance import performances_val
import os
import torch.nn.functional as F
import numpy as np
from PIL import ImageFilter
import random

def mk_ct_label(label,UUID):
    label_ct = label.clone()

    for iii in range(0,len(label_ct)):
        if label[iii] == 0 and UUID[iii] ==0:
            label_ct[iii] = 0
        elif label[iii] ==0 and UUID[iii] ==1:
            label_ct[iii] = 1
        elif label[iii] == 0 and UUID[iii] == 2:
            label_ct[iii] = 2
        elif label[iii] == 1 and UUID[iii] ==0:
            label_ct[iii] = 3
        elif label[iii] ==1 and UUID[iii] ==1:
            label_ct[iii] = 4
        elif label[iii] == 1 and UUID[iii] == 2:
            label_ct[iii] = 5
        else:
            raise Exception

    return label_ct

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        if n > 0:
            self.sum += val * n
            self.cnt += n
            self.avg = self.sum / self.cnt


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def protocol_decoder(protocol):
    if protocol == "O_C_I_to_M":
        data_name_list_train = ["OULU", "CASIA_MFSD", "Replay_attack"]
        data_name_list_test = ["MSU_MFSD"]
    if protocol == "O_C_to_M":
        data_name_list_train = ["OULU", "CASIA_MFSD"]
        data_name_list_test = ["MSU_MFSD"]
    if protocol == "O_to_O":
        data_name_list_train = ["OULU"]
        data_name_list_test = ["OULU"]
    elif protocol == "O_M_I_to_C":
        data_name_list_train = ["OULU", "MSU_MFSD", "Replay_attack"]
        data_name_list_test = ["CASIA_MFSD"]
    elif protocol == "O_C_M_to_I":
        data_name_list_train = ["OULU", "CASIA_MFSD", "MSU_MFSD"]
        data_name_list_test = ["Replay_attack"]
    elif protocol == "I_C_M_to_O":
        data_name_list_train = ["MSU_MFSD", "CASIA_MFSD", "Replay_attack"]
        data_name_list_test = ["OULU"]
    elif protocol == "M_I_to_C":
        data_name_list_train = ["MSU_MFSD", "Replay_attack"]
        data_name_list_test = ["CASIA_MFSD"]
    elif protocol == "M_I_to_O":
        data_name_list_train = ["MSU_MFSD", "Replay_attack"]
        data_name_list_test = ["OULU"]
    return data_name_list_train, data_name_list_test

import torch
from torchvision.transforms.transforms import _setup_size


class NineCrop(torch.nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            tuple of 5 images. Image can be PIL Image or Tensor
        """
        return self.nine_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


    def nine_crop(self, img, size) :
        import numbers
        from torchvision.transforms.functional import _get_image_size, crop, center_crop

        if isinstance(size, numbers.Number):
            size = (int(size), int(size))
        elif isinstance(size, (tuple, list)) and len(size) == 1:
            size = (size[0], size[0])

        if len(size) != 2:
            raise ValueError("Please provide only two dimensions (h, w) for size.")

        image_width, image_height = _get_image_size(img)
        crop_height, crop_width = size
        if crop_width > image_width or crop_height > image_height:
            msg = "Requested crop size {} is bigger than input size {}"
            raise ValueError(msg.format(size, (image_height, image_width)))

        tl = crop(img, 0, 0, crop_height, crop_width)
        tm = crop(img, 0, (image_width - crop_width) // 2, crop_height, crop_width)
        tr = crop(img, 0, image_width - crop_width, crop_height, crop_width)

        ml = crop(img, (image_height - crop_height) // 2, 0, crop_height, crop_width)
        # mm = crop(img, (image_height - crop_height) // 2, (image_width - crop_width) // 2, crop_height, crop_width)
        mr = crop(img, (image_height - crop_height) // 2, image_width - crop_width, crop_height, crop_width)

        bl = crop(img, image_height - crop_height, 0, crop_height, crop_width)
        bm = crop(img, image_height - crop_height, (image_width - crop_width) // 2, crop_height, crop_width)
        br = crop(img, image_height - crop_height, image_width - crop_width, crop_height, crop_width)

        center = center_crop(img, [crop_height, crop_width])

        return tl, tm, tr, ml, center, mr, bl, bm, br


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RandomCutout(object):
    def __init__(self, n_holes, p=0.5):
        """
        Args:
            n_holes (int): Number of patches to cut out of each image.
            p (int): probability to apply cutout
        """
        self.n_holes = n_holes
        self.p = p

    def rand_bbox(self, W, H, lam):
        """
        Return a random box
        """
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if np.random.rand(1) > self.p:
            return img

        h = img.size(1)
        w = img.size(2)
        lam = np.random.beta(1.0, 1.0)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(w, h, lam)
        for n in range(self.n_holes):
            img[:, bby1:bby2, bbx1:bbx2] = img[:, bby1:bby2, bbx1:bbx2].mean(dim=[-2, -1], keepdim=True)
        return img


class RandomJPEGCompression(object):
    def __init__(self, quality_min=30, quality_max=90, p=0.5):
        assert 0 <= quality_min <= 100 and 0 <= quality_max <= 100
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img):
        if np.random.rand(1) > self.p:
            return img
        # Choose a random quality for JPEG compression
        quality = np.random.randint(self.quality_min, self.quality_max)

        # Save the image to a bytes buffer using JPEG format
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)

        # Reload the image from the buffer
        img = Image.open(buffer)
        return img
