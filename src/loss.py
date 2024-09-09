import torch
import torch.nn as nn
from torch import optim

EPSILON = 1e-40  # value to use to approximate zero (to prevent undefined results)


class GLoss(nn.Module):
    ''' C-RNN-GAN generator loss
    '''

    def __init__(self):
        super(GLoss, self).__init__()

    def forward(self, logits_gen):
        logits_gen = torch.clamp(logits_gen, EPSILON, 1.0)
        batch_loss = -torch.log(logits_gen)

        return torch.mean(batch_loss)


class DLoss(nn.Module):
    ''' C-RNN-GAN discriminator loss
    '''

    def __init__(self, label_smoothing=False):
        super(DLoss, self).__init__()
        self.label_smoothing = label_smoothing

    def forward(self, logits_real, logits_gen):
        ''' Discriminator loss

        logits_real: logits from D, when input is real
        logits_gen: logits from D, when input is from Generator

        loss = -(ylog(p) + (1-y)log(1-p))

        '''
        logits_real = torch.clamp(logits_real, EPSILON, 1.0)
        d_loss_real = -torch.log(logits_real)

        if self.label_smoothing:
            p_fake = torch.clamp((1 - logits_real), EPSILON, 1.0)
            d_loss_fake = -torch.log(p_fake)
            d_loss_real = 0.9 * d_loss_real + 0.1 * d_loss_fake

        logits_gen = torch.clamp((1 - logits_gen), EPSILON, 1.0)
        d_loss_gen = -torch.log(logits_gen)

        batch_loss = d_loss_real + d_loss_gen
        return torch.mean(batch_loss)


class CRGLoss:
    """
    Wrapper class for the generator and discriminator losses
    """

    def __init__(self, g_loss, d_loss: DLoss):
        self.g_loss = g_loss
        self.d_loss = d_loss

    @property
    def gen(self):
        return self.g_loss

    @property
    def disc(self):
        return self.d_loss
