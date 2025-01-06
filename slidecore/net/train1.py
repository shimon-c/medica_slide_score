import os.path
import matplotlib.pyplot as plt
import torch
import slidecore
import slidecore.net
#import slidecore.net.gpu_utils

from slidecore.net.resnet import ResNet
from slidecore.net.datatset import DataSet
import tqdm
import sklearn
from sklearn.metrics import confusion_matrix
import slidecore.net.yaml_obj
import shutil
import logging

#from slidecore.net.gpu_utils import get_devstr

def get_devstr(gpu:int=0):
    devstr = 'cpu'
    cnt = dev_cnt = torch.cuda.device_count()
    if gpu>=0 and gpu<cnt:
        devstr = f'cuda:{gpu}'
    return devstr

# arch=[(64,3), (128,4), (256,6),(512,3)]
# head_arch=[18,32]
# device = 'cuda'
# xsize,ysize=128,128
def train_epoch(net=None, loader=None, optim=None, loss_obj=None, device=None):
    train_loss = 0
    correct = 0
    total = 0
    net.train()
    for bid,tars in tqdm.tqdm(enumerate(loader), desc="train_epoch"):
        optim.zero_grad()
        inputs, labs = tars
        inputs = inputs.to(device)
        labs = labs.to(device)
        outputs = net(inputs,target=labs)
        N = labs.shape[0]
        labs = labs.reshape((N,))
        loss = loss_obj(outputs, labs)
        loss.backward()
        optim.step()
        train_loss += loss.item()
        _,preds = outputs.max(1)
        total += labs.shape[0]
        correct += preds.eq(labs).sum().item()
    train_loss /= bid
    acc = correct/total
    return train_loss, acc

def compute_acc(net=None, loader=None, calc_conf_mat=False, device='cuda'):
    conf_mat = None
    y_true,y_pred=[],[]
    nok = 0
    N = 0
    net.eval()
    for bid,tars in tqdm.tqdm(enumerate(loader), desc='compute-acc'):
        input, target = tars
        input = input.to(device)
        target = target.to(device)
        output = net(input)
        _,pr = output.max(1)
        tt = target.reshape((target.shape[0],))
        rel_ids = pr != slidecore.net.datatset.DataSet.NOT_RELV_IMG
        pr = pr[rel_ids]
        if pr.shape[0]<=0:
            continue
        tt = tt[rel_ids]
        nok += pr.eq(tt).sum().item()
        N += target.shape[0]
        y_pred.extend(pr.tolist())
        y_true.extend(tt.tolist())
    N = len(y_true)
    acc = nok/N
    if calc_conf_mat:
        conf_mat = confusion_matrix(y_true, y_pred)
    return acc,conf_mat

def test_net(model_path, loader=None, device=None):
    resnet, args, optim_params,sched_params,epoch = slidecore.net.resnet.ResNet.load(model_path)
    restnet = resnet.to(device)
    acc, confm = compute_acc(net=resnet,loader=loader)
    print(f'Accuracy:{acc}')

def train(args, log_obj=None):
    nepochs = args['nepochs']
    xsize, ysize = args['xsize'], args['ysize']
    device = get_devstr(args['gpu'])
    good_path ,bad_path,not_rel=args['train_good'],args['train_bad'],args['train_relv']
    test_good, test_bad=args['test_good'], args['test_bad']
    checkpoint_dir = args['checkpoints_dir']
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir,ignore_errors=True)
    os.makedirs(args['checkpoints_dir'], exist_ok=True)
    batch_size=args['batch_size']
    train_ds = slidecore.net.datatset.DataSet(root_dir=args['train_set_dir'],
                                              good_path=good_path, bad_path=bad_path,
                                              not_rel=not_rel, xsize=xsize, ysize=ysize,
                                              augmentations=args['augmentations'],
                                              train_csv_file=args['train_csv_file'])
    dataset_str = f'----Train stat -----\n{train_ds.dataset_stat_str}\nds-size:{len(train_ds)}\n--------\n'
    print(dataset_str)
    log_obj.info(dataset_str)
    log_obj.info(train_ds.dataset_stat_str)
    max_std = train_ds.max_std
    num_cls = train_ds.get_num_of_classes()
    args['out_cls'] = num_cls
    resnet = slidecore.net.resnet.ResNet(args=args)
    resnet.set_max_hcf([max_std])
    test_ds = slidecore.net.datatset.DataSet(root_dir=args['test_set_dir'],
                                         good_path=test_good, bad_path=test_bad,
                                         xsize=xsize, ysize=ysize, test_flag=True,
                                             train_csv_file=args['test_csv_file'])
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ld = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    weight_decay = args['weight_decay']
    lr = args.get('lr')
    if lr is None:
        lr = 0.01
    if args.get('optim') == 'Adam':
        optim = torch.optim.Adam(lr=lr, params=resnet.parameters(),weight_decay=weight_decay)
    else:
        optim = torch.optim.SGD(params=resnet.parameters(), lr=lr, momentum=0.9, nesterov=True,
                                weight_decay=weight_decay)
    print(f'optim:{type(optim)}')
    sched_str = args.get('sched')
    if args.get('sched'):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=nepochs, eta_min=0, last_epoch=-1, verbose='deprecated')
    loss_obj = torch.nn.CrossEntropyLoss()
    resnet = resnet.to(device)
    model_path = os.path.join(checkpoint_dir, 'resnet.pt')
    log_obj.info('------------- ARGS ------------')
    log_obj.info(str(args))
    log_obj.info('-------------------------------')
    for ep in range(nepochs):
        train_loss, tr_acc = train_epoch(net=resnet, loader=tr_loader, optim=optim,
                                         loss_obj=loss_obj, device=device)
        test_acc,_ = compute_acc(net=resnet, loader=test_ld, device=device)
        model_path = os.path.join(checkpoint_dir, f'resnet_epoch_{ep}.pt')
        save_name = resnet.save(file_path=model_path, optim=optim, sched=sched, epoch=ep)
        #test_net(save_name, loader=test_ld)
        sched.step()
        log_str = f"epoch:{ep}, train_loss:{train_loss},train_acc:{tr_acc}, test_acc:{test_acc}, lr:{sched.get_last_lr()}"
        print(log_str)
        log_obj.info(log_str)
    test_acc, cmat = compute_acc(net=resnet, loader=test_ld, calc_conf_mat=True, device=device)
    model_path = os.path.join(checkpoint_dir, 'resnet.pt')
    resnet.save(file_path=model_path, optim=optim, sched=sched, epoch=ep)
    print(f'Final accuracy:{test_acc}, conf_mat:\n{cmat}\nmodel:{model_path}')
    dmat = sklearn.metrics.ConfusionMatrixDisplay(cmat)
    dmat.plot()
    plt.show()


############################ Test Path #####################
train_set_dir = r"C:\Users\shimon.cohen\data\medica\imgdb\imgdb\train_set"
test_set_dir = r"C:\Users\shimon.cohen\data\medica\imgdb\imgdb\test_set"
def get_train_valid_paths():
    train_good = os.path.join(train_set_dir,"GoodFocus")
    train_bad = os.path.join(train_set_dir, "BadFocus")
    train_relv = os.path.join(train_set_dir, "NotRelevant")
    test_good = os.path.join(test_set_dir, "GoodFocus")
    test_bad = os.path.join(test_set_dir, "BadFocus")
    return (train_good, train_bad, train_relv), (test_good, test_bad)

import argparse
def get_args():
    ap = argparse.ArgumentParser('train 1')
    ap.add_argument('--yaml_path', type=str, required=True, help="Full path of yaml file")
    args = ap.parse_args()
    yaml_obj = slidecore.net.yaml_obj.YamlObj(yaml_path=args.yaml_path)
    yaml_args = yaml_obj.get_params()
    return yaml_args

if __name__ == "__main__":
    train_data, test_data = get_train_valid_paths()
    args = get_args()
    logger = logging.getLogger(__name__)
    chk_pnts = args['checkpoints_dir']
    #os.rmdir(chk_pnts, )
    if os.path.exists(chk_pnts):
        shutil.rmtree(chk_pnts, ignore_errors=True)
    os.makedirs(chk_pnts, exist_ok=True)
    log_file = os.path.join(chk_pnts, 'log.txt')
    if os.path.exists(log_file):
        os.remove(log_file)
    print(f'log file: {log_file}')
    #logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG)
    logging.basicConfig(filename=log_file,level=logging.DEBUG)
    train(args=args, log_obj=logger)