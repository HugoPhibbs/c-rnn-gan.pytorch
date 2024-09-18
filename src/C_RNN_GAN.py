from torch.utils.data import DataLoader

from c_rnn_gan.src import training_constants as tc
from c_rnn_gan.src.CRG_model import CRGModel, Generator, Discriminator
from c_rnn_gan.src.loss import CRGLoss, GLoss, DLoss
from c_rnn_gan.src.optimizer import CRGOptimizer
from c_rnn_gan.src.trainer import CRGTrainer


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
    def get_instance(num_feats, gen_params, disc_params, optimizer_params, data_loader, num_epochs,
                     training_constants: tc.TrainingConstants,
                     label_smoothing=False, use_cuda=False, writer_path=None):
        generator = Generator(num_feats, gen_params["hidden_units"], gen_params["drop_prob"], use_cuda)
        discriminator = Discriminator(num_feats, disc_params["hidden_units"], disc_params["drop_prob"],
                                      use_cuda)
        model = CRGModel(generator, discriminator)

        optimizer = CRGOptimizer(model,
                                 gen_learn_rate=optimizer_params["g_learn_rate"],
                                 disc_learn_rate=optimizer_params["d_learn_rate"],
                                 use_sgd=optimizer_params["use_sgd"]
                                 )

        loss = CRGLoss(GLoss(eps=training_constants.eps), DLoss(label_smoothing))

        trainer = CRGTrainer(model, data_loader, optimizer, loss, num_epochs, num_feats,
                             training_constants.batch_size, training_constants.max_grad_norm,
                             training_constants.max_seq_length,
                             seq_length=training_constants.seq_length,
                             per_nth_batch_print_memory=training_constants.per_nth_batch_print_memory,
                             writer_path=writer_path)

        return C_RNN_GAN(model, trainer, data_loader, optimizer, loss)
