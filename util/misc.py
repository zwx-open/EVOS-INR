import random
import os
import numpy as np
import torch
import datetime
from itertools import cycle, islice
import json 
import string


def gen_cur_time():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

def gen_random_str(nums=4):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(nums))

def load_json(filepath):
    with open(filepath, 'r') as f:
        loaded_json = json.load(f)
    return loaded_json

def calc_intersec(pred, gt:torch.tensor, ratio=1):
    
    pred = pred.flatten()
    gt = gt.flatten()
    assert pred.shape[0] == gt.shape[0]
    
    len_ = pred.shape[0]
    ratio_len_ = int(ratio*len_)
    pred = pred[:ratio_len_]
    gt = gt[:ratio_len_]

    intersec = torch.isin(pred, gt).sum().item() / ratio_len_
    return intersec

def fix_seed(seed=3407):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # cuda > 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)

def random_split_into_two_list_inds(seed, dataset_length, train_weight=7, val_weight=1, occu_weight=0):
    full_idxes = [x for x in range(dataset_length)]
    all_weight = train_weight+val_weight+occu_weight
    val_length = int(val_weight / all_weight * dataset_length)
    train_length = int(train_weight / all_weight * dataset_length)
    random.seed(seed)
    val_idxes = random.sample(full_idxes, val_length)
    remain_idxes = list(set(full_idxes).difference(set(val_idxes)))
    train_idxes = random.sample(remain_idxes, train_length)
    remain_idxes = list(set(remain_idxes).difference(set(train_idxes)))
    return train_idxes, val_idxes, remain_idxes

def get_file_size(file, tag="MB"):
    size = os.path.getsize(file)
    if tag == "B":
        pass
    elif tag == "KB":
        size = size / 1024
    elif tag == "MB":
        size = size / 1024 / 1024
    elif tag == "GB":
        size = size / 1024 / 1024 / 1024
    else:
        raise NotImplementedError
    return f'{size} {tag}'


def format_time(seconds):
    '''25 min 26.94s'''
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:.2f}h {minutes:.2f}min {seconds:.2f}s"

def format_time2(seconds):
    '''25.17 min'''
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    seconds_float = seconds / 60
    minutes += seconds_float
    return f"{hours:.2f}h {minutes:.2f}min"


class AverageMeter(object):
    def __init__(self, name=""):
        self.name = name
        self.dic = {}
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self) -> str:
        return f"{self.name}: average {self.count}: {self.avg}\n"