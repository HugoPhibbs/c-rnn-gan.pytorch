# Copyright 2019 Christopher John Bayron
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been created by Christopher John Bayron based on "rnn_gan.py"
# by Olof Mogren. The referenced code is available in:
#
#     https://github.com/olofmogren/c-rnn-gan

import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from c_rnn_gan.src.crgmodel import CRGModel, Generator, Discriminator
from c_rnn_gan.src.loss import CRGLoss, GLoss, DLoss
from c_rnn_gan.src.optimizer import CRGOptimizer
from c_rnn_gan.src.trainer import CRGTrainer
from utils.data_utils import SingleGameDatasetRandomChoose

G_FN = 'c_rnn_gan_g.pth'
D_FN = 'c_rnn_gan_d.pth'


MAX_GRAD_NORM = 5.0
# following values are modified at runtime
MAX_SEQ_LEN = 200
BATCH_SIZE = 32

EPSILON = 1e-40  # value to use to approximate zero (to prevent undefined results)


class C_RNN_GAN:

    def __init__(self, model: CRGModel, trainer: CRGTrainer, data_loader: DataLoader, optimizer: CRGOptimizer,
                 loss):
        self._model = model
        self._trainer = trainer
        self._data_loader = data_loader
        self._optimizer = optimizer
        self._loss = loss

    @property
    def model(self):
        """Model"""
        return self._model

    @property
    def data_l(self):
        """Data loader"""
        return self._data_loader

    @property
    def opt(self):
        """Optimizer"""
        return self._optimizer

    @property
    def loss(self):
        """loss"""
        return self._loss

    @property
    def trainer(self):
        return self._trainer

    def train(self):
        return self.trainer.run_training()

    @staticmethod
    def get_instance(num_feats, gen_params, disc_params, optimizer_params, data_loader, num_epochs, num_batches,
                     label_smoothing=False, use_cuda=False):
        generator = Generator(num_feats, gen_params["hidden_units"], gen_params["drop_prob"], use_cuda)
        discriminator = Discriminator(num_feats, disc_params["hidden_units"], disc_params["drop_prob"],
                                      use_cuda)
        model = CRGModel(generator, discriminator)

        optimizer = CRGOptimizer(model,
                                 gen_learn_rate=optimizer_params["g_learn_rate"],
                                 disc_learn_rate=optimizer_params["d_learn_rate"],
                                 use_sgd=optimizer_params["use_sgd"]
                                 )

        loss = CRGLoss(GLoss(), DLoss(label_smoothing))

        trainer = CRGTrainer(model, data_loader, optimizer, loss, num_batches, num_epochs, num_feats, save_g=False,
                             save_d=False)

        return C_RNN_GAN(model, trainer, data_loader, optimizer, loss)


def main(args):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    gen_params = {
        "hidden_units": args.g_hidden_units,
        "drop_prob": args.g_drop_prob
    }

    disc_params = {
        "hidden_units": args.d_hidden_units,
        "drop_prob": args.d_drop_prob
    }

    optimizer_params = {
        "g_learn_rate": args.g_lrn_rate,
        "d_learn_rate": args.d_lrn_rate,
        "use_sgd": args.use_sgd
    }

    cols_to_predict = args.cols_to_predict.split(',')

    dataset = SingleGameDatasetRandomChoose(args.game_name, frame_count=args.seq_size, cols_to_predict=cols_to_predict)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    crg = C_RNN_GAN.get_instance(dataset.num_pred_features, gen_params, disc_params, optimizer_params, data_loader,
                                 num_epochs=args.num_epochs,
                                 num_batches=args.num_batches,
                                 label_smoothing=args.label_smoothing,
                                 use_cuda=use_cuda)

    crg.train()

if __name__ == "__main__":
    ARG_PARSER = ArgumentParser()
    ARG_PARSER.add_argument('--load_g', action='store_true')
    ARG_PARSER.add_argument('--load_d', action='store_true')
    ARG_PARSER.add_argument('--no_save_g', action='store_true')
    ARG_PARSER.add_argument('--no_save_d', action='store_true')

    ARG_PARSER.add_argument('--num_epochs', default=300, type=int)
    ARG_PARSER.add_argument('--seq_len', default=256, type=int)
    ARG_PARSER.add_argument('--batch_size', default=16, type=int)
    ARG_PARSER.add_argument('--g_lrn_rate', default=0.001, type=float)
    ARG_PARSER.add_argument('--d_lrn_rate', default=0.001, type=float)

    ARG_PARSER.add_argument('--no_pretraining', action='store_true')
    ARG_PARSER.add_argument('--g_pretraining_epochs', default=5, type=int)
    ARG_PARSER.add_argument('--d_pretraining_epochs', default=5, type=int)
    # ARG_PARSER.add_argument('--freeze_d_every', default=5, type=int)
    ARG_PARSER.add_argument('--use_sgd', action='store_true')
    ARG_PARSER.add_argument('--conditional_freezing', action='store_true')
    ARG_PARSER.add_argument('--label_smoothing', action='store_true')
    ARG_PARSER.add_argument('--feature_matching', action='store_true')

    ARG_PARSER.add_argument("--col_to_predict", type=str, help="Comma-separated list argument")

    ARGS = ARG_PARSER.parse_args()
    MAX_SEQ_LEN = ARGS.seq_len
    BATCH_SIZE = ARGS.batch_size

    main(ARGS)
