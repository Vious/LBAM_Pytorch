import torch
from torch.utils.data import Dataset
from PIL import Image
from os import listdir, walk
from os.path import join
from random import randint
from data.basicFunction import CheckImageFile, ImageTransform, MaskTransform

class GetData(Dataset):
    def __init__(self, dataRoot, maskRoot, loadSize, cropSize):
        super(GetData, self).__init__()

        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.masks = [join (dataRootK, files) for dataRootK, dn, filenames in walk(maskRoot) \
            for files in filenames if CheckImageFile(files)]
        self.numOfMasks = len(self.masks)
        self.loadSize = loadSize
        self.cropSize = cropSize
        self.ImgTrans = ImageTransform(loadSize, cropSize)
        self.maskTrans = MaskTransform(cropSize)
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        mask = Image.open(self.masks[randint(0, self.numOfMasks - 1)])

        groundTruth = self.ImgTrans(img.convert('RGB'))
        mask = self.maskTrans(mask.convert('RGB'))

        # we add this threshhold to force the input mask to be binary 0,1 values
        # the threshhold value can be changeble, i think 0.5 is ok
        threshhold = 0.5
        ones = mask >= threshhold
        zeros = mask < threshhold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)

        # here, we suggest that the white values(ones) denotes the area to be inpainted, 
        # and dark values(zeros) is the values remained. 
        # Therefore, we do a reverse step let mask = 1 - mask, the input = groundTruth * mask, :).
        mask = 1 - mask
        inputImage = groundTruth * mask
        inputImage = torch.cat((inputImage, mask[0].view(1, self.cropSize[0], self.cropSize[1])), 0)

        return inputImage, groundTruth, mask
    
    def __len__(self):
        return len(self.imageFiles)
