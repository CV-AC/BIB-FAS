import os, torch
from utils import protocol_decoder
import math


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def get_single_dataset(data_dir, FaceDataset,is_live, data_name="", train=True,
                       transform=None, debug_subset_size=None, UUID=-1):
    if train:
        if data_name in ["OULU"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'OULU/preposess'), split='train',
                                   transform=transform, UUID=UUID,is_live = is_live)
        elif data_name in ["CASIA_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'CASIA_FASD/preposess'), split='train',

                                   transform=transform, UUID=UUID,is_live = is_live)
        elif data_name in ["Replay_attack"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'Replay/preposess'), split='train',
                                   transform=transform, UUID=UUID,is_live = is_live)
        elif data_name in ["MSU_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'MSU_MFSD/preposess'), split='train',
                                   transform=transform, UUID=UUID,is_live = is_live)
        if debug_subset_size is not None:
            data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
    else:
        if data_name in ["OULU"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'OULU/preposess'), split='test',
                                   transform=transform, UUID=UUID)
        elif data_name in ["CASIA_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'CASIA_FASD/preposess'), split='test',

                                   transform=transform, UUID=UUID)
        elif data_name in ["Replay_attack"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'Replay/preposess'), split='test',
                                   transform=transform, UUID=UUID)
        elif data_name in ["MSU_MFSD"]:
            data_set = FaceDataset(data_name, os.path.join(data_dir, 'MSU_MFSD/preposess'), split='test',
                                   transform=transform, UUID=UUID)
        if debug_subset_size is not None:
            data_set = torch.utils.data.Subset(data_set, range(0, debug_subset_size))
    # print("Loading {}, number: {}".format(data_name, len(data_set)))
    return data_set

def get_datasets(data_dir, FaceDataset,is_live = 1, train=True, protocol="1", transform=None,
                 debug_subset_size=None, ):
    data_name_list_train, data_name_list_test = protocol_decoder(protocol)

    sum_n = 0
    if train:
        data_set_sum = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[0], train=True,
                                          transform=transform,
                                          debug_subset_size=debug_subset_size, UUID=0, is_live = is_live)
        sum_n = len(data_set_sum)
        for i in range(1, len(data_name_list_train)):
            data_tmp = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_train[i], train=True,
                                          transform=transform,
                                          debug_subset_size=debug_subset_size, UUID=i, is_live = is_live)
            data_set_sum += data_tmp
            sum_n += len(data_tmp)
    else:
        data_set_sum = {}
        for i in range(len(data_name_list_test)):
            data_tmp = get_single_dataset(data_dir, FaceDataset, data_name=data_name_list_test[i], train=False,
                                          transform=transform,
                                          debug_subset_size=debug_subset_size, UUID=i, is_live= is_live)
            data_set_sum[data_name_list_test[i]] = data_tmp
            sum_n += len(data_tmp)

    return data_set_sum
