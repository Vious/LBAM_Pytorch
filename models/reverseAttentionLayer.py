import math
import torch
from torch import nn
from models.ActivationFunction import GaussActivation, MaskUpdate
from models.weightInitial import weights_init


# learnable reverse attention conv
class ReverseMaskConv(nn.Module):
    def __init__(self, inputChannels, outputChannels, kernelSize=4, stride=2, 
        padding=1, dilation=1, groups=1, convBias=False):
        super(ReverseMaskConv, self).__init__()

        self.reverseMaskConv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, \
            dilation, groups, bias=convBias)

        self.reverseMaskConv.apply(weights_init())

        self.activationFuncG_A = GaussActivation(1.1, 1.0, 0.5, 0.5)
        self.updateMask = MaskUpdate(0.8)
    
    def forward(self, inputMasks):
        maskFeatures = self.reverseMaskConv(inputMasks)

        maskActiv = self.activationFuncG_A(maskFeatures)

        maskUpdate = self.updateMask(maskFeatures)

        return maskActiv, maskUpdate

# learnable reverse attention layer, including features activation and batchnorm
class ReverseAttention(nn.Module):
    def __init__(self, inputChannels, outputChannels, bn=False, activ='leaky', \
        kernelSize=4, stride=2, padding=1, outPadding=0,dilation=1, groups=1,convBias=False, bnChannels=512):
        super(ReverseAttention, self).__init__()

        self.conv = nn.ConvTranspose2d(inputChannels, outputChannels, kernel_size=kernelSize, \
            stride=stride, padding=padding, output_padding=outPadding, dilation=dilation, groups=groups,bias=convBias)
        
        self.conv.apply(weights_init())

        if bn:
            self.bn = nn.BatchNorm2d(bnChannels)
        
        if activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2, False)
        elif activ == 'relu':
            self.activ = nn.ReLU()
        elif activ == 'sigmoid':
            self.activ = nn.Sigmoid()
        elif activ == 'tanh':
            self.activ = nn.Tanh()
        elif activ == 'prelu':
            self.activ = nn.PReLU()
        else:
            pass

    def forward(self, ecFeaturesSkip, dcFeatures, maskFeaturesForAttention):
        nextDcFeatures = self.conv(dcFeatures)
        
        # note that encoder features are ahead, it's important tor make forward attention map ahead 
        # of reverse attention map when concatenate, we do it in the LBAM model forward function
        concatFeatures = torch.cat((ecFeaturesSkip, nextDcFeatures), 1)
        
        outputFeatures = concatFeatures * maskFeaturesForAttention

        if hasattr(self, 'bn'):
            outputFeatures = self.bn(outputFeatures)
        if hasattr(self, 'activ'):
            outputFeatures = self.activ(outputFeatures)

        return outputFeatures