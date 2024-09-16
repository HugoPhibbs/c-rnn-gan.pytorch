import os

import torch
from torch import nn
from torch.utils.data import DataLoader

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from c_rnn_gan.src.CRG_model import CRGModel
    from c_rnn_gan.src.loss import CRGLoss
    from c_rnn_gan.src.run_training import CRGOptimizer

G_FN = 'c_rnn_gan_g.pth'
D_FN = 'c_rnn_gan_d.pth'


class CRGTrainer:
    """
    Trainer class for C-RNN-GAN (CRG) model
    """

    def __init__(self, model, data_loader, optimizer, loss,
                 num_batches: int, num_epochs: int, num_features: int, batch_size: int, max_grad_norm: float,
                 max_seq_length: int, model_save_path: str = None,
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
        self.model_save_path = model_save_path

        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.max_seq_length = max_seq_length

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
            g_path = os.path.join(self.model_save_path, G_FN)

            torch.save(self.model.gen.state_dict(), g_path)
            print(f"Saved generator: {g_path}")

        if self.save_d:
            d_path = os.path.join(self.model_save_path, D_FN)

            torch.save(self.model.disc.state_dict(), d_path)
            print(f"Saved discriminator: {d_path}")

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

        for (batch_data, _) in self.data_loader:
            # Reset the hidden states for each batch
            g_states = self.model.gen.init_hidden(self.batch_size)
            d_states = self.model.disc.init_hidden(self.batch_size)

            ## Generator

            # Removed GLoss from loss, thought it wasn't needed for an MVP

            self.optimizer.gen.zero_grad()
            z = torch.empty([self.batch_size, self.max_seq_length, num_features]).uniform_()

            g_feats, _ = self.model.gen(z, g_states)
            # feed real and generated input to discriminator
            d_logits_gen, _, _ = self.model.disc(g_feats, d_states)
            loss_calculated['gen'] = self.loss.gen(d_logits_gen)

            loss_calculated['gen'].backward()
            self.optimizer.gen.step()

            ## Discriminator

            self.optimizer.disc.zero_grad()

            # feed real and generated input to discriminator
            # It seems there is a disconnect between how much data the discriminator is using, and how much the generator is using
            d_logits_real, _, _ = self.model.disc(batch_data, d_states)
            # need to detach from operation history to prevent backpropagating to generator
            d_logits_gen, _, _ = self.model.disc(g_feats.detach(), d_states)
            # calculate loss, backprop, and update weights of D
            loss_calculated['disc'] = self.loss.disc(d_logits_real, d_logits_gen)

            log_sum_real += d_logits_real.sum().item()
            log_sum_gen += d_logits_gen.sum().item()

            loss_calculated['disc'].backward()
            nn.utils.clip_grad_norm_(self.model.disc.parameters(), max_norm=self.max_grad_norm)
            self.optimizer.disc.step()

            g_loss_total += loss_calculated['gen'].item()
            d_loss_total += loss_calculated['disc'].item()
            num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()

        g_loss_avg, d_loss_avg = 0.0, 0.0
        d_acc = 0.0
        num_sample = self.batch_size * self.num_batches

        if num_sample > 0:
            g_loss_avg = g_loss_total / num_sample
            d_loss_avg = d_loss_total / num_sample
            d_acc = 100 * num_corrects / (2 * num_sample)  # 2 because (real + generated)

            print("Trn: ", log_sum_real / num_sample, log_sum_gen / num_sample)

        return g_loss_avg, d_loss_avg, d_acc
