from torch import optim

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from c_rnn_gan.src.CRG_model import CRGModel

from c_rnn_gan.src.training_constants import G_LRN_RATE_DEFAULT, D_LRN_RATE_DEFAULT


class CRGOptimizer:

    def __init__(self, model, gen_learn_rate=G_LRN_RATE_DEFAULT, disc_learn_rate=D_LRN_RATE_DEFAULT,
                 use_sgd=True):
        self.use_sgd = use_sgd

        if self.use_sgd:
            self._generator = optim.SGD(model.gen.parameters(), lr=gen_learn_rate, momentum=0.9)
            self._discriminator = optim.SGD(model.disc.parameters(), lr=disc_learn_rate, momentum=0.9)
        else:
            self._generator = optim.Adam(model.gen.parameters(), gen_learn_rate)
            self._discriminator = optim.Adam(model.disc.parameters(), disc_learn_rate)

    @property
    def gen(self):
        return self._generator

    @property
    def disc(self):
        return self._discriminator
