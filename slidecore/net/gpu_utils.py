import torch

def get_num_gpus():
    dev_cnt = torch.cuda.device_count()
    return dev_cnt

def get_devstr(gpu:int=0):
    devstr = 'cpu'
    cnt = get_num_gpus()
    if gpu>=0 and gpu<cnt:
        devstr = f'cuda:{gpu}'
    return devstr