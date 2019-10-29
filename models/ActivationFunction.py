import math
import torch
from torch.nn.parameter import Parameter
from torch import nn
from torchvision import models

# asymmetric gaussian shaped activation function g_A 
class GaussActivation(nn.Module):
    def __init__(self, a, mu, sigma1, sigma2):
        super(GaussActivation, self).__init__()

        self.a = Parameter(torch.tensor(a, dtype=torch.float32))
        self.mu = Parameter(torch.tensor(mu, dtype=torch.float32))
        self.sigma1 = Parameter(torch.tensor(sigma1, dtype=torch.float32))
        self.sigma2 = Parameter(torch.tensor(sigma2, dtype=torch.float32))

    
    def forward(self, inputFeatures):

        self.a.data = torch.clamp(self.a.data, 1.01, 6.0)
        self.mu.data = torch.clamp(self.mu.data, 0.1, 3.0)
        self.sigma1.data = torch.clamp(self.sigma1.data, 0.5, 2.0)
        self.sigma2.data = torch.clamp(self.sigma2.data, 0.5, 2.0)

        lowerThanMu = inputFeatures < self.mu
        largerThanMu = inputFeatures >= self.mu

        leftValuesActiv = self.a * torch.exp(- self.sigma1 * ( (inputFeatures - self.mu) ** 2 ) )
        leftValuesActiv.masked_fill_(largerThanMu, 0.0)

        rightValueActiv = 1 + (self.a - 1) * torch.exp(- self.sigma2 * ( (inputFeatures - self.mu) ** 2 ) )
        rightValueActiv.masked_fill_(lowerThanMu, 0.0)

        output = leftValuesActiv + rightValueActiv

        return output

# mask updating functions, we recommand using alpha that is larger than 0 and lower than 1.0
class MaskUpdate(nn.Module):
    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()

        self.updateFunc = nn.ReLU(True)
        #self.alpha = Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.alpha = alpha
    def forward(self, inputMaskMap):
        """ self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        print(self.alpha) """

        return torch.pow(self.updateFunc(inputMaskMap), self.alpha)