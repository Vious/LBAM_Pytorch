import math
import torch
from torch import nn
from models.ActivationFunction import GaussActivation, MaskUpdate
from models.weightInitial import weights_init

# learnable forward attention conv layer
class ForwardAttentionLayer(nn.Module):
    def __init__(self, inputChannels, outputChannels, kernelSize, stride, 
        padding, dilation=1, groups=1, bias=False):
        super(ForwardAttentionLayer, self).__init__()

        self.conv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, dilation, \
            groups, bias)

        if inputChannels == 4:
            self.maskConv = nn.Conv2d(3, outputChannels, kernelSize, stride, padding, dilation, \
                groups, bias)
        else:
            self.maskConv = nn.Conv2d(inputChannels, outputChannels, kernelSize, stride, padding, \
                dilation, groups, bias)
        
        self.conv.apply(weights_init())
        self.maskConv.apply(weights_init())

        self.activationFuncG_A = GaussActivation(1.1, 2.0, 1.0, 1.0)
        self.updateMask = MaskUpdate(0.8)

    def forward(self, inputFeatures, inputMasks):
        convFeatures = self.conv(inputFeatures)
        maskFeatures = self.maskConv(inputMasks)
        #convFeatures_skip = convFeatures.clone()

        maskActiv = self.activationFuncG_A(maskFeatures)
        convOut = convFeatures * maskActiv

        maskUpdate = self.updateMask(maskFeatures)

        return convOut, maskUpdate, convFeatures, maskActiv

# forward attention gather feature activation and batchnorm
class ForwardAttention(nn.Module):
    def __init__(self, inputChannels, outputChannels, bn=False, sample='down-4', \
        activ='leaky', convBias=False):
        super(ForwardAttention, self).__init__()

        if sample == 'down-4':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 4, 2, 1, bias=convBias)
        elif sample == 'down-5':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 5, 2, 2, bias=convBias)
        elif sample == 'down-7':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 7, 2, 3, bias=convBias)
        elif sample == 'down-3':
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 3, 2, 1, bias=convBias)
        else:
            self.conv = ForwardAttentionLayer(inputChannels, outputChannels, 3, 1, 1, bias=convBias)
        
        if bn:
            self.bn = nn.BatchNorm2d(outputChannels)
        
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
    
    def forward(self, inputFeatures, inputMasks):
        features, maskUpdated, convPreF, maskActiv = self.conv(inputFeatures, inputMasks)

        if hasattr(self, 'bn'):
            features = self.bn(features)
        if hasattr(self, 'activ'):
            features = self.activ(features)
        
        return features, maskUpdated, convPreF, maskActiv