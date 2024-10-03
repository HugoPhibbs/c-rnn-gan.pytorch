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

import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhScaled(nn.Module):
    def __init__(self, scale=10):
        super(TanhScaled, self).__init__()
        self.scale = scale

    def forward(self, x):
        return torch.tanh(x) * self.scale


class Generator(nn.Module):
    """
    Generator model for C-RNN-GAN
    """

    def __init__(self, num_feats, hidden_units=256, drop_prob=0.1, use_cuda=False, tanh_scale=10):
        super(Generator, self).__init__()

        # params
        self.hidden_dim = hidden_units
        self.use_cuda = use_cuda
        self.num_feats = num_feats

        # Set the layers
        self.fc_layer1a = nn.Linear(in_features=(num_feats * 2),
                                    out_features=3 * hidden_units)  # 2 for z and previous output
        self.ln_f1a = nn.LayerNorm(3 * hidden_units)
        self.fc_layer1b = nn.Linear(in_features=3 * hidden_units,
                                    out_features=hidden_units)
        self.ln_f1b = nn.LayerNorm(hidden_units)

        self.lstm_cell1 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.ln1 = nn.LayerNorm(hidden_units)

        self.dropout = nn.Dropout(p=drop_prob)

        self.lstm_cell2 = nn.LSTMCell(input_size=hidden_units, hidden_size=hidden_units)
        self.ln2 = nn.LayerNorm(hidden_units)

        self.fc_layer2a = nn.Linear(in_features=hidden_units, out_features=3 * hidden_units)
        self.ln_f2a = nn.LayerNorm(3 * hidden_units)
        self.fc_layer2b = nn.Linear(in_features=3 * hidden_units, out_features=num_feats)
        self.ln_f2b = nn.LayerNorm(num_feats)

        self.activation = TanhScaled(1)

        if use_cuda:
            self.cuda()

    def forward_helper(self, input, state1, state2):
        """
        Does the actual forward prop for the generator

        Made into its own function for easy reading and refactoring

        Args:
            input: Concatenated input features (z and previous timestep output)
            state1: Hidden state for LSTM cell 1
            state2: Hidden state for LSTM cell 2
        """
        out_1a = self.fc_layer1a(input)
        out_1a = F.leaky_relu(out_1a, negative_slope=0.01)
        out_1b = self.fc_layer1b(out_1a)
        out_1b = F.leaky_relu(out_1b, negative_slope=0.01)

        h1, c1 = self.lstm_cell1(out_1b, state1)
        # h1 = self.ln1(h1)
        # h1 = self.dropout(h1)  # feature dropout only (no recurrent dropout)
        h2, c2 = self.lstm_cell2(h1, state2)
        # h2 = self.ln2(h2)

        out_f2a = self.fc_layer2a(h2)
        out_f2a = F.leaky_relu(out_f2a, negative_slope=0.01)
        out_f2b = self.fc_layer2b(out_f2a)
        # out_f2b = F.leaky_relu(out_f2b, negative_slope=0.01)

        out = self.activation(out_f2b)

        return out, (h1, c1), (h2, c2)

    def forward(self, z, states):
        """
        Forward prop

        Args:
            z: Random noise tensor of shape (batch_size, seq_len, num_feats)
            states: ((h1, c1), (h2, c2)) Initial states for the two LSTM cells
        """
        if self.use_cuda:
            z = z.cuda()

        # z: (seq_len, num_feats)
        # z here is the uniformly random vector
        batch_size, seq_len, num_feats = z.shape

        # split to seq_len * (batch_size * num_feats)
        z = torch.split(z, 1, dim=1)
        z = [z_step.squeeze(dim=1) for z_step in z]

        # create dummy-previous-output for first timestep
        prev_gen = torch.empty([batch_size, num_feats]).normal_()
        if self.use_cuda:
            prev_gen = prev_gen.cuda()

        # manually process each timestep
        state1, state2 = states  # (h1, c1), (h2, c2)
        gen_feats = []
        for z_step in z:
            # concatenate current input features and previous timestep output features
            concat_in = torch.cat((z_step, prev_gen), dim=-1)

            prev_gen, (h1, c1), (h2, c2) = self.forward_helper(concat_in, state1, state2)

            gen_feats.append(prev_gen)

            state1 = (h1, c1)
            state2 = (h2, c2)

        # seq_len * (batch_size * num_feats) -> (batch_size * seq_len * num_feats)
        gen_feats = torch.stack(gen_feats, dim=1)

        states = (state1, state2)
        return gen_feats, states

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        if (self.use_cuda):
            hidden = ((weight.new(batch_size, self.hidden_dim).uniform_().cuda(),
                       weight.new(batch_size, self.hidden_dim).uniform_().cuda()),
                      (weight.new(batch_size, self.hidden_dim).uniform_().cuda(),
                       weight.new(batch_size, self.hidden_dim).uniform_().cuda()))
        else:
            hidden = ((weight.new(batch_size, self.hidden_dim).uniform_(),
                       weight.new(batch_size, self.hidden_dim).uniform_()),
                      (weight.new(batch_size, self.hidden_dim).uniform_(),
                       weight.new(batch_size, self.hidden_dim).uniform_()))

        return hidden


class Discriminator(nn.Module):
    ''' C-RNN-GAN discrminator
    '''

    def __init__(self, num_feats, hidden_units=256, drop_prob=0.6, use_cuda=False):

        super(Discriminator, self).__init__()

        # params
        self.hidden_dim = hidden_units
        self.num_layers = 2
        self.use_cuda = use_cuda

        self.dropout = nn.Dropout(p=drop_prob)
        self.lstm = nn.LSTM(input_size=num_feats, hidden_size=hidden_units,
                            num_layers=self.num_layers, batch_first=True, dropout=drop_prob,
                            bidirectional=True)
        self.fc_layer = nn.Linear(in_features=(2 * hidden_units), out_features=1)

        if use_cuda:
            self.cuda()

    def forward(self, note_seq, state):
        ''' Forward prop
        '''
        if self.use_cuda:
            note_seq = note_seq.cuda()

        # note_seq: (batch_size, seq_len, num_feats)
        drop_in = self.dropout(note_seq)  # input with dropout
        # (batch_size, seq_len, num_directions*hidden_size)
        lstm_out, state = self.lstm(drop_in, state)
        # (batch_size, seq_len, 1)
        out = self.fc_layer(lstm_out)
        out = torch.sigmoid(out)

        # Collapse to (batch_size, seq_len, 1) -> (batch_size, 1), care about per-sequence output
        num_dims = len(out.shape)
        reduction_dims = tuple(range(1, num_dims))
        # (batch_size)
        out = torch.mean(out, dim=reduction_dims)

        return out, lstm_out, state

    def init_hidden(self, batch_size):
        ''' Initialize hidden state '''
        # create NEW tensor with SAME TYPE as weight
        weight = next(self.parameters()).data

        layer_mult = 2  # for being bidirectional

        if self.use_cuda:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).uniform_().cuda(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).uniform_().cuda())
        else:
            hidden = (weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).uniform_(),
                      weight.new(self.num_layers * layer_mult, batch_size,
                                 self.hidden_dim).uniform_())

        return hidden


class CRGModel:
    """
    C-RNN-GAN model

    Wrapper class for Generator and Discriminator models
    """

    def __init__(self, generator: Generator, discriminator: Discriminator):
        self._generator = generator
        self._discriminator = discriminator

    @property
    def gen(self):
        return self._generator

    @property
    def disc(self):
        return self._discriminator
