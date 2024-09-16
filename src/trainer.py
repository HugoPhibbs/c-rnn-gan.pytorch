import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from c_rnn_gan.generate import CKPT_DIR
from c_rnn_gan.src.CRG_model import CRGModel
from c_rnn_gan.src.loss import CRGLoss
from c_rnn_gan.src.run_training import CRGOptimizer

G_FN = 'c_rnn_gan_g.pth'
D_FN = 'c_rnn_gan_d.pth'

G_LRN_RATE = 0.001
D_LRN_RATE = 0.001
MAX_GRAD_NORM = 5.0
# following values are modified at runtime
MAX_SEQ_LEN = 200
BATCH_SIZE = 32

EPSILON = 1e-40  # value to use to approximate zero (to prevent undefined results)


class CRGTrainer:
    """
    Trainer class for C-RNN-GAN (CRG) model
    """

    def __init__(self, model: CRGModel, data_loader: DataLoader, optimizer: CRGOptimizer, loss: CRGLoss,
                 num_batches: int, num_epochs: int, num_features: int,
                 save_g=False,
                 save_d=False):
        self.model = model
        self.data_loader = data_loader
        self.num_batches = num_batches
        self.optimizer = optimizer
        self.loss = loss
        self.num_epochs = num_epochs
        self.num_features = num_features

        self.save_g = save_g
        self.save_d = save_d

    def run_training(self):
        """
        Runs the entire training process
        """
        for ep in range(self.num_epochs):
            _ = self.run_epoch(ep)

        self.save_models()

    def save_models(self):
        """
        Saves the models if needed
        """
        if self.save_g:
            torch.save(self.model.gen.state_dict(), os.path.join(CKPT_DIR, G_FN))
            print("Saved generator: %s" % os.path.join(CKPT_DIR, G_FN))

        if self.save_d:
            torch.save(self.model.disc.state_dict(), os.path.join(CKPT_DIR, D_FN))
            print("Saved discriminator: %s" % os.path.join(CKPT_DIR, D_FN))

    def run_epoch(self, epoch_idx):
        """
        Run a single training epoch

        """
        trn_g_loss, trn_d_loss, trn_acc = self.run_epoch_helper()

        # TODO: implement validation

        print("Epoch %d/%d " % (epoch_idx + 1, self.num_epochs))

        print("\t[Training] G_loss: %0.8f, D_loss: %0.8f, D_acc: %0.2f\n", (trn_g_loss, trn_d_loss, trn_acc))

        return trn_acc

    def run_epoch_helper(self):
        """
        Helper method for running an epoch

        Does the actual running of the epoch - i.e. going thru batches and updating weights etc
        """
        num_features = self.num_features

        self.model.gen.train()
        self.model.disc.train()

        loss_calculated = {}
        g_loss_total = 0.0
        d_loss_total = 0.0
        num_corrects = 0

        log_sum_real = 0.0
        log_sum_gen = 0.0

        for (batch_data, batch_labels) in self.data_loader:
            for (k, v) in batch_data:
                continue # TODO loop through the batch data, and refactor the below code to be in this loop


            # Reset the hidden states for each batch
            g_states = self.model.gen.init_hidden()
            d_states = self.model.disc.init_hidden()

            ## Generator

            # Removed GLoss from loss, thought it wasn't needed for an MVP

            self.optimizer.gen.zero_grad()
            z = torch.empty([MAX_SEQ_LEN, num_features]).uniform_()

            g_feats, _ = self.model.gen(z, g_states)
            # feed real and generated input to discriminator
            d_logits_gen, _, new_d_states = self.model.disc(g_feats, d_states)
            loss_calculated['gen'] = self.model.gen(d_logits_gen)

            loss_calculated['gen'].backward()
            self.optimizer.gen.step()

            ## Discriminator

            self.optimizer.disc.zero_grad()

            # feed real and generated input to discriminator
            # TODO change how the batch_data is fed to the discriminator
            # It seems there is a disconnect between how much data the discriminator is using, and how much the generator is using
            d_logits_real, _, _ = self.model.disc(batch_data, d_states)
            # need to detach from operation history to prevent backpropagating to generator
            d_logits_gen, _, _ = self.model.disc(g_feats.detach(), d_states)
            # calculate loss, backprop, and update weights of D
            loss_calculated['disc'] = self.loss.disc(d_logits_real, d_logits_gen)

            log_sum_real += d_logits_real.sum().item()
            log_sum_gen += d_logits_gen.sum().item()

            loss_calculated['disc'].backward()
            nn.utils.clip_grad_norm_(self.model.disc.parameters(), max_norm=MAX_GRAD_NORM)
            self.optimizer.disc.step()

            g_loss_total += loss_calculated['gen'].item()
            d_loss_total += loss_calculated['disc'].item()
            num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()

        g_loss_avg, d_loss_avg = 0.0, 0.0
        d_acc = 0.0
        num_sample = BATCH_SIZE * self.num_batches

        if num_sample > 0:
            g_loss_avg = g_loss_total / num_sample
            d_loss_avg = d_loss_total / num_sample
            d_acc = 100 * num_corrects / (2 * num_sample)  # 2 because (real + generated)

            print("Trn: ", log_sum_real / num_sample, log_sum_gen / num_sample)

        return g_loss_avg, d_loss_avg, d_acc
