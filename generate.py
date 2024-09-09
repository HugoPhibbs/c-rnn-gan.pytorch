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
import numpy as np
import torch

from c_rnn_gan.src.crgmodel import Generator
CKPT_DIR = 'models'
G_FN = 'c_rnn_gan_g.pth'
MAX_SEQ_LEN = 256
FILENAME = 'sample.mid'

def generate(n):
    pass

if __name__ == "__main__":
    ARG_PARSER = ArgumentParser()
    # number of times to execute generator model;
    # all generated data are concatenated to form a single longer sequence
    ARG_PARSER.add_argument('-n', default=1, type=int)
    ARGS = ARG_PARSER.parse_args()

    generate(ARGS.n)
