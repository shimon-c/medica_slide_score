import slidecore
import argparse

def parse_args():
    ap = argparse.ArgumentParser('GetNetParams')
    ap.add_argument('--net_chk_pnt', type=str, required=True, help='Full net path')
    args = ap.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    net, margs, optim_params, sched_params, epoch = slidecore.resnet.ResNet.load(file_path=args.net_chk_pnt)
    file_name = f'{args.net_chk_pnt}.args.txt'
    with open(file_name, "w") as file:
        for k,v in margs.items():
            file.write(f'{k}:  {v}\n')