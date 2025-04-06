# nlpia/cuda_utils.py
import torch
import pandas as pd


def find_unused_device():
    usages = []
    for check in range(10):
        usages.append([])
        for i, architecture in enumerate(torch.cuda.get_arch_list()):
            usages[-1].append(torch.cuda.get_memory_usage(i))
    usages = pd.DataFrame(usages).sum()
    return torch.device(f'cuda:{usages.arg_min()}')
