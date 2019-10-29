from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, Resize, RandomHorizontalFlip

def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])

def ImageTransform(loadSize, cropSize):
    return Compose([
        Resize(size=loadSize, interpolation=Image.BICUBIC),
        RandomCrop(size=cropSize),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
    ])

def MaskTransform(cropSize):
    return Compose([
        Resize(size=cropSize, interpolation=Image.NEAREST),
        ToTensor(),
    ])

# this was image transforms function for paired image and mask, which means that damaged image and the 
# mask are in pairs, the input image already contains damaged area with (ones or zeros),
#  we suggest that you resize the input image with "NEAREST" not BICUBIC(or other) algorithm,
#  is is not guaranteed, but in some cases, the damaged portion might go out of the mask region, if you perform other resize methods
def PairedImageTransform(cropSize):
    return Compose([
        Resize(size=cropSize, interpolation=Image.NEAREST),
        ToTensor(),
    ])