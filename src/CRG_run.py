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
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from c_rnn_gan.src.C_RNN_GAN import C_RNN_GAN

import c_rnn_gan.src.training_constants as tc
from utils.data_utils import DataUtils

from utils.datasets import SingleGameControlDataset

G_FN = 'c_rnn_gan_g.pth'
D_FN = 'c_rnn_gan_d.pth'


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
        "g_learn_rate": args.g_learn_rate,
        "d_learn_rate": args.d_learn_rate,
        "use_sgd": args.use_sgd
    }

    training_constants = tc.TrainingConstants(g_learn_rate=args.g_learn_rate, d_learn_rate=args.d_learn_rate,
                                              batch_size=args.batch_size, eps=args.epsilon, seq_length=args.seq_len,
                                              num_epochs=args.num_epochs,
                                              g_hidden_units=args.g_hidden_units, d_hidden_units=args.d_hidden_units,
                                              g_drop_prob=args.g_drop_prob, d_drop_prob=args.d_drop_prob,
                                              max_grad_norm=args.max_grad_norm, max_seq_length=args.max_seq_length,
                                              per_nth_batch_print_memory=args.per_nth_batch_print_memory)

    cols_to_keep = args.cols_to_keep.split(',') if args.cols_to_keep is not None else None

    if args.train_session_set is not None:
        args.train_session_set = DataUtils.read_txt(f"{args.dataset_root}{args.train_session_set}")

    parquet_folder_path = f"{args.dataset_root}/parquet"

    dataset = SingleGameControlDataset(args.game_name, session_set=args.train_session_set, cols_to_keep=cols_to_keep,
                                       frame_count=args.seq_len, parquet_folder_path=parquet_folder_path)

    num_features = dataset.num_features



    if args.subset_size > 0:
        subset_indices = list(range(0, args.subset_size - args.subset_size % args.batch_size))
        dataset = torch.utils.data.Subset(dataset, subset_indices)

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    crg = C_RNN_GAN.get_instance(num_features, gen_params, disc_params, optimizer_params, data_loader,
                                 num_epochs=args.num_epochs,
                                 training_constants=training_constants,
                                 label_smoothing=args.label_smoothing,
                                 use_cuda=use_cuda,
                                 writer_path=args.writer_path)

    crg.train()

    # TODO add testing code


if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--load_g', action='store_true', default=False)
    arg_parser.add_argument('--load_d', action='store_true', default=False)
    arg_parser.add_argument('--save_g', action='store_true', default=False)
    arg_parser.add_argument('--save_d', action='store_true', default=False)
    arg_parser.add_argument("--model_save_path", type=str, help="Path to where models are saved",
                            default=r"C:\Users\hugop\Documents\Uni\Graphics\COMPSCI715\c_rnn_gan\models")

    arg_parser.add_argument('--num_epochs', default=tc.NUM_EPOCHS_DEFAULT, type=int)
    arg_parser.add_argument('--seq_len', default=tc.SEQ_LENGTH_DEFAULT, type=int)
    arg_parser.add_argument('--batch_size', default=tc.BATCH_SIZE_DEFAULT, type=int)
    arg_parser.add_argument('--g_learn_rate', default=tc.G_LRN_RATE_DEFAULT, type=float)
    arg_parser.add_argument('--d_learn_rate', default=tc.D_LRN_RATE_DEFAULT, type=float)
    arg_parser.add_argument("--epsilon", default=tc.EPSILON_DEFAULT, type=float)

    arg_parser.add_argument("--max_grad_norm", default=tc.MAX_GRAD_NORM_DEFAULT, type=float)
    arg_parser.add_argument("--max_seq_length", default=tc.MAX_SEQ_LEN_DEFAULT, type=int)
    arg_parser.add_argument("--per_nth_batch_print_memory", default=tc.PER_NTH_BATCH_PRINT_MEMORY_DEFAULT, type=int)

    arg_parser.add_argument("--subset_size", type=int, help="Size of subset to use for training", default=-1)

    arg_parser.add_argument('--g_hidden_units', default=tc.G_HIDDEN_UNITS_DEFAULT, type=int)
    arg_parser.add_argument('--d_hidden_units', default=tc.D_HIDDEN_UNITS_DEFAULT, type=int)
    arg_parser.add_argument('--g_drop_prob', default=tc.G_DROP_PROB_DEFAULT, type=float)
    arg_parser.add_argument('--d_drop_prob', default=tc.D_DROP_PROB_DEFAULT, type=float)

    arg_parser.add_argument('--no_pretraining', action='store_true', default=False)
    arg_parser.add_argument('--g_pretraining_epochs', default=5, type=int)
    arg_parser.add_argument('--d_pretraining_epochs', default=5, type=int)
    # ARG_PARSER.add_argument('--freeze_d_every', default=5, type=int)
    arg_parser.add_argument('--use_sgd', action='store_true', default=False)
    arg_parser.add_argument('--conditional_freezing', action='store_true', default=False)
    arg_parser.add_argument('--label_smoothing', action='store_true', default=False)
    arg_parser.add_argument('--feature_matching', action='store_true', default=False)

    # For data loading
    arg_parser.add_argument("--dataset_root", type=str, help="Root directory of dataset",
                            default=r"C:\Users\hugop\Documents\Uni\Graphics\COMPSCI715\datasets")

    # For logs
    arg_parser.add_argument("--writer_path", type=str, help="Path to save tensorboard logs",
                            default=r"C:\Users\hugop\Documents\Uni\Graphics\COMPSCI715\c_rnn_gan\logs\experiment0")

    # For Single Game Training
    arg_parser.add_argument("--game_name", type=str, help="Game name to run training on", default='Flight_Squad')
    arg_parser.add_argument("--train_session_set", type=str, help=".txt file path containing list of training sessions",
                            default=None)
    arg_parser.add_argument("--cols_to_keep", type=str, help="Comma-separated list argument of columns to keep",
                            default="head_pos_x,head_pos_y")

    args = arg_parser.parse_args()

    print(args)

    main(args)
