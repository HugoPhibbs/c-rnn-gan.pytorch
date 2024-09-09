from torch import optim

from c_rnn_gan.src.crgmodel import CRGModel

G_LRN_RATE = 0.001
D_LRN_RATE = 0.001

class CRGOptimizer:

    def __init__(self, model: CRGModel, gen_learn_rate=G_LRN_RATE, disc_learn_rate=D_LRN_RATE, use_sgd=True):
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