import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import pandas as pd

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from c_rnn_gan.src.CRG_model import CRGModel
    from c_rnn_gan.src.loss import CRGLoss
    from c_rnn_gan.src.CRG_run import CRGOptimizer

G_FN = 'c_rnn_gan_g.pth'
D_FN = 'c_rnn_gan_d.pth'


class CRGTrainer:
    """
    Trainer class for C-RNN-GAN (CRG) model
    """

    def __init__(self, model, data_loader: DataLoader, optimizer, loss, num_epochs: int, num_features: int,
                 batch_size: int, max_grad_norm: float,
                 max_seq_length: int, model_save_path: str = None, per_nth_batch_print_memory: int = -1,
                 writer_path: str = None, seq_length: int = 10,
                 save_g=False,
                 save_d=False):
        self.model = model
        self.data_loader = data_loader
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

        self.per_nth_batch_print_memory = per_nth_batch_print_memory

        self.seq_length = seq_length
        # self.train_data_df = pd.DataFrame(columns=['epoch', 'g_loss', 'd_loss', 'd_acc'])

        self.writer = SummaryWriter(writer_path)

    def run_training(self):
        """
        Runs the entire training process
        """
        # self.writer.add_graph(self.model.gen, torch.empty([self.batch_size, self.seq_length, self.num_features]))
        # self.writer.add_graph(self.model.disc, torch.empty([self.batch_size, self.seq_length, self.num_features]))

        for ep in tqdm(range(self.num_epochs), desc=f"Epoch Progress (Total: {self.num_epochs})"):
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

        print("Epoch %d/%d " % (epoch_idx + 1, self.num_epochs))
        trn_g_loss, trn_d_loss, trn_d_acc = self.run_epoch_helper(epoch_idx)

        # TODO: implement validation

        print(f"[Training] G_loss: {trn_g_loss:.8f} %, D_loss: {trn_d_loss:.8f}, D_acc: {trn_d_acc:.2f}\n")

        self.writer.add_scalar('Loss/g_loss', trn_g_loss, epoch_idx)
        self.writer.add_scalar('Loss/d_loss', trn_d_loss, epoch_idx)
        self.writer.add_scalar('Accuracy/d_acc', trn_d_acc, epoch_idx)

        return trn_d_acc

    def run_epoch_helper(self, epoch_idx, start_training_disc_at=0):
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

        num_batches = len(self.data_loader)

        progress_bar = tqdm(self.data_loader, desc=f"Epoch {epoch_idx + 1}", leave=False, dynamic_ncols=True)

        for i, (batch_data, _) in enumerate(progress_bar):
            # if self.per_nth_batch_print_memory > 0 and i % self.per_nth_batch_print_memory == 0:
            #     self.memory_summary()

            # Batch_data has shape (batch_size, seq_len, num_features)

            # Reset the hidden states for each batch
            g_states = self.model.gen.init_hidden(self.batch_size)
            d_states = self.model.disc.init_hidden(self.batch_size)

            ## Generator

            # Removed GLoss from loss, thought it wasn't needed for an MVP

            self.optimizer.gen.zero_grad()
            z = torch.empty([self.batch_size, self.seq_length, num_features]).uniform_()

            g_feats, _ = self.model.gen(z, g_states)
            # feed real and generated input to discriminator
            d_logits_gen, _, _ = self.model.disc(g_feats, d_states)
            loss_calculated['gen'] = self.loss.gen(d_logits_gen)

            loss_calculated['gen'].backward()
            nn.utils.clip_grad_norm_(self.model.gen.parameters(), max_norm=self.max_grad_norm)
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

            if i >= start_training_disc_at:
                loss_calculated['disc'].backward()
                nn.utils.clip_grad_norm_(self.model.disc.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.disc.step()

            g_loss_total += loss_calculated['gen'].item()
            d_loss_total += loss_calculated['disc'].item()
            num_corrects += (d_logits_real > 0.5).sum().item() + (d_logits_gen < 0.5).sum().item()

            if i % 100 == 0:
                global_idx = num_batches * epoch_idx + i
                np.save(f"gen_output_{global_idx}.npy", g_feats[-1].detach().cpu().numpy())

            progress_bar.refresh()

        num_sample = self.batch_size * num_batches
        assert num_sample > 0, "No samples taken from the dataset"

        g_loss_avg = g_loss_total / num_batches
        d_loss_avg = d_loss_total / num_batches
        d_acc = 100 * num_corrects / (2 * num_sample)  # 2 because (real + generated)

        print("Trn: ", log_sum_real / num_sample, log_sum_gen / num_sample)

        return g_loss_avg, d_loss_avg, d_acc

    def memory_summary(self):
        """
        Prints a summary of memory usage
        """
        assert torch.cuda.is_available(), "CUDA not available"

        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)  # Total memory in MB
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Allocated memory in MB

        # Calculate percentage
        percentage = (allocated_memory / total_memory) * 100
        print(f"\nCUDA Memory Usage: {allocated_memory:.2f} MB / {total_memory:.2f} MB, {percentage:.2f}%")
