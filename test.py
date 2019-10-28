import os
import math
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from torchvision import datasets
from models.LBAMModel import LBAMModel
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage
from data.basicFunction import CheckImageFile
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='', help='input damaged image')
parser.add_argument('--mask', type=str, default='', help='input mask')
parser.add_argument('--output', type=str, default='output', help='output file name')
parser.add_argument('--pretrained', type=str, default='', help='load pretrained model')
parser.add_argument('--loadSize', type=int, default=350,
                    help='image loading size')
parser.add_argument('--cropSize', type=int, default=256,
                    help='image training size')

args = parser.parse_args()


ImageTransform = Compose([
    Resize(size=args.cropSize, interpolation=Image.NEAREST),
    ToTensor(),
])

MaskTransform = Compose([
    Resize(size=args.cropSize, interpolation=Image.NEAREST),
    ToTensor(),
])

if not CheckImageFile(args.input):
    print('Input file is not image file!')
elif not CheckImageFile(args.mask):
    print('Input mask is not image file!')
elif args.pretrained == '':
    print('Provide pretrained model!')
else:

    image = ImageTransform(Image.open(args.input).convert('RGB'))
    mask = MaskTransform(Image.open(args.mask).convert('RGB'))
    threshhold = 0.5
    ones = mask >= threshhold
    zeros = mask < threshhold

    mask.masked_fill_(ones, 1.0)
    mask.masked_fill_(zeros, 0.0)

    mask = 1 - mask
    sizes = image.size()
    
    image = image * mask
    inputImage = torch.cat((image, mask[0].view(1, sizes[1], sizes[2])), 0)
    inputImage = inputImage.view(1, 4, sizes[1], sizes[2])

    
    mask = mask.view(1, sizes[0], sizes[1], sizes[2])
    
    netG = LBAMModel(4, 3)

    netG.load_state_dict(torch.load(args.pretrained))
    for param in netG.parameters():
        param.requires_grad = False
    netG.eval()
    print(netG.reverseConv5.updateMask.alpha)
    if torch.cuda.is_available():
        netG = netG.cuda()
        inputImage = inputImage.cuda()
        mask = mask.cuda()
    
    output = netG(inputImage, mask)
    output = output * (1 - mask) + inputImage[:, 0:3, :, :] * mask

    save_image(output, args.output + '.png')