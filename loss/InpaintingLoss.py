import torch
from torch import nn
from torch import autograd
from tensorboardX import SummaryWriter
from models.discriminator import DiscriminatorDoubleColumn

# modified from WGAN-GP
def calc_gradient_penalty(netD, real_data, fake_data, masks, cuda, Lambda):
    BATCH_SIZE = real_data.size()[0]
    DIM = real_data.size()[2]
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(real_data.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, 3, DIM, DIM)
    if cuda:
        alpha = alpha.cuda()
    
    fake_data = fake_data.view(BATCH_SIZE, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    if cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, masks)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if cuda else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * Lambda
    return gradient_penalty.sum().mean()


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


#tv loss
def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
        torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss



class InpaintingLossWithGAN(nn.Module):
    def __init__(self, logPath, extractor, Lamda, lr, betasInit=(0.5, 0.9)):
        super(InpaintingLossWithGAN, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.discriminator = DiscriminatorDoubleColumn(3)
        self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=betasInit)
        self.cudaAvailable = torch.cuda.is_available()
        self.numOfGPUs = torch.cuda.device_count()
        """ if (self.numOfGPUs > 1):
            self.discriminator = self.discriminator.cuda()
            self.discriminator = nn.DataParallel(self.discriminator, device_ids=range(self.numOfGPUs)) """
        self.lamda = Lamda
        self.writer = SummaryWriter(logPath)

    def forward(self, input, mask, output, gt, count, epoch):
        self.discriminator.zero_grad()
        D_real = self.discriminator(gt, mask)
        D_real = D_real.mean().sum() * -1
        D_fake = self.discriminator(output, mask)
        D_fake = D_fake.mean().sum() * 1
        gp = calc_gradient_penalty(self.discriminator, gt, output, mask, self.cudaAvailable, self.lamda)
        D_loss = D_fake - D_real + gp
        self.D_optimizer.zero_grad()
        D_loss.backward(retain_graph=True)
        self.D_optimizer.step()

        self.writer.add_scalar('LossD/Discrinimator loss', D_loss.item(), count)
        
        output_comp = mask * input + (1 - mask) * output

        holeLoss = 6 * self.l1((1 - mask) * output, (1 - mask) * gt)
        validAreaLoss = self.l1(mask * output, mask * gt)   

        if output.shape[1] == 3:
            feat_output_comp = self.extractor(output_comp)
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output_comp = self.extractor(torch.cat([output_comp]*3, 1))
            feat_output = self.extractor(torch.cat([output]*3, 1))
            feat_gt = self.extractor(torch.cat([gt]*3, 1))
        else:
            raise ValueError('only gray an')

        prcLoss = 0.0
        for i in range(3):
            prcLoss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            prcLoss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])

        styleLoss = 0.0
        for i in range(3):
            styleLoss += 120 * self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))
            styleLoss += 120 * self.l1(gram_matrix(feat_output_comp[i]),
                                          gram_matrix(feat_gt[i]))
        
        """ if self.numOfGPUs > 1:
            holeLoss = holeLoss.sum() / self.numOfGPUs
            validAreaLoss = validAreaLoss.sum() / self.numOfGPUs
            prcLoss = prcLoss.sum() / self.numOfGPUs
            styleLoss = styleLoss.sum() / self.numOfGPUs """
        self.writer.add_scalar('LossG/Hole loss', holeLoss.item(), count)    
        self.writer.add_scalar('LossG/Valid loss', validAreaLoss.item(), count)    
        self.writer.add_scalar('LossPrc/Perceptual loss', prcLoss.item(), count)    
        self.writer.add_scalar('LossStyle/style loss', styleLoss.item(), count)    

        GLoss = holeLoss + validAreaLoss + prcLoss + styleLoss + 0.1 * D_fake
        self.writer.add_scalar('Generator/Joint loss', GLoss.item(), count)    
        return GLoss.sum()