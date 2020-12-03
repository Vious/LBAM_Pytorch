import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import random
import inspect, re
import math
import collections
from torchvision.utils import save_image


def wrapper_gmask(args):
    mask_global = torch.ByteTensor(1, 1, \
                                        args.imgSize, args.imgSize)

    res = args.res  # the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
    density = args.density
    MAX_SIZE = 350
    maxPartition = 30
    low_pattern = torch.rand(1, 1, int(res * MAX_SIZE), int(res * MAX_SIZE)).mul(255)
    pattern = nn.functional.interpolate(low_pattern, (MAX_SIZE, MAX_SIZE), mode='bilinear').detach()
    low_pattern = None
    pattern.div_(255)
    pattern = torch.lt(pattern, density).byte()  # 25% 1s and 75% 0s
    pattern = torch.squeeze(pattern).byte()

    gMask_opts = {}
    gMask_opts['pattern'] = pattern
    gMask_opts['MAX_SIZE'] = MAX_SIZE
    gMask_opts['fineSize'] = args.imgSize
    gMask_opts['maxPartition'] = maxPartition
    gMask_opts['mask_global'] = mask_global
    return create_gMask(gMask_opts)  # create an initial random mask

def create_gMask(gMask_opts, limit_cnt=1):
    pattern = gMask_opts['pattern']
    mask_global = gMask_opts['mask_global']
    MAX_SIZE = gMask_opts['MAX_SIZE']
    fineSize = gMask_opts['fineSize']
    maxPartition=gMask_opts['maxPartition']
    if pattern is None:
        raise ValueError
    wastedIter = 0
    while wastedIter <= limit_cnt:
        x = random.randint(1, MAX_SIZE-fineSize)
        y = random.randint(1, MAX_SIZE-fineSize)
        mask = pattern[y:y+fineSize, x:x+fineSize] # need check
        area = mask.sum()*100./(fineSize*fineSize)
        if area>20 and area<maxPartition:
            break
        wastedIter += 1
    if mask_global.dim() == 3:
        mask_global = mask.expand(1, mask.size(0), mask.size(1))
    else:
        mask_global = mask.expand(1, 1, mask.size(0), mask.size(1))
    return mask_global


parser = argparse.ArgumentParser()
parser.add_argument('--imgSize', type=int, default=256)
parser.add_argument('--loadSize', type=int, default=350)
parser.add_argument('--numbers', type=int, default=6000)
parser.add_argument('--density', type=float, default=0.25, help='region to be masked as 0')
parser.add_argument('--res', type=float, default=0.025, help='the lower it is, \
        the more continuous the output will be. 0.01 is too small and 0.1 is too large')
parser.add_argument('--save_dir', type=str, default='./NewMask3')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

for i in range(args.numbers):
    mask = 1 - wrapper_gmask(args)
    mask = mask.view(1, mask.size()[2], mask.size()[3])
    if (mask.sum() >= 61250):
        i -= 1
        continue
    else:
        print(i, mask.size(), mask.sum())
        save_image(mask // 1, '{:s}/{:06d}.png'.format(args.save_dir, i))

