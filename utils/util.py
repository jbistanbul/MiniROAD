import pickle
import os
import os.path as osp
import random
import numpy as np
import torch

def dump_pickle(lst, file_path, file_name):
    with open(osp.join(file_path, file_name + '.pkl'), 'wb') as f:
        pickle.dump(lst, f)

def create_dir(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)

def create_outdir(result_path):
    i = 1
    new_result_path = result_path
    while osp.exists(new_result_path):
        new_result_path = f'{result_path}_{i}'
        i += 1
    create_dir(osp.join(new_result_path, 'ckpts'))
    create_dir(osp.join(new_result_path, 'runs'))
    return new_result_path

def set_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True