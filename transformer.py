# 2017.12.16 by xiaohang
import sys
from caffenet import *
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from functools import partial
import pickle
import time

def create_network(protofile, weightfile):
    net = CaffeNet(protofile)
    if args.cuda:
        net.cuda()
    print(net)
    net.load_weights(weightfile)
    return net

def modify(net):
    state_dict = net.state_dict()
    ts = set()
    for k, v in state_dict.items():
        ts.add(type(v))
        if type(v) == torch.nn.modules.container.Sequential or type(v) == Concat:
            del state_dict[k]
    print('Type in the state_dict: ', ts)
    return state_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert caffe to pytorch')
    parser.add_argument('-p', '--protofile', default='', type=str)
    parser.add_argument('-m', '--weightfile', default='', type=str)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')

    args = parser.parse_args()
    print(args)
    
    protofile = args.protofile
    weightfile = args.weightfile

    net = create_network(protofile, weightfile)
    ss  = modify(net)
    torch.save(ss, args.weightfile + '.pth')
    print("Save successfully ", args.weightfile + '.pth')
