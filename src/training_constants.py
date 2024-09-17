from dataclasses import dataclass

G_LRN_RATE_DEFAULT = 0.001
D_LRN_RATE_DEFAULT = 0.001
MAX_GRAD_NORM_DEFAULT = 5.0
MAX_SEQ_LEN_DEFAULT = 256
BATCH_SIZE_DEFAULT = 128
EPSILON_DEFAULT = 1e-40
SEQ_LENGTH_DEFAULT = 10
NUM_EPOCHS_DEFAULT = 10
G_HIDDEN_UNITS_DEFAULT = 256
D_HIDDEN_UNITS_DEFAULT = 256
G_DROP_PROB_DEFAULT = 0.5
D_DROP_PROB_DEFAULT = 0.5
NUM_BATCHES_DEFAULT = 100
PER_NTH_BATCH_PRINT_MEMORY_DEFAULT = 10


@dataclass
class TrainingConstants:
    """
    Configuration class for the GAN model.

    I.e. contains hyper params for the model, optimizer, etc.
    """

    g_learn_rate: float = G_LRN_RATE_DEFAULT
    d_learn_rate: float = G_LRN_RATE_DEFAULT
    max_grad_norm: float = MAX_GRAD_NORM_DEFAULT
    max_seq_length: int = MAX_SEQ_LEN_DEFAULT
    batch_size: int = BATCH_SIZE_DEFAULT
    eps: float = EPSILON_DEFAULT
    seq_length: int = SEQ_LENGTH_DEFAULT
    num_epochs: int = NUM_EPOCHS_DEFAULT
    g_hidden_units: int = G_HIDDEN_UNITS_DEFAULT
    d_hidden_units: int = D_HIDDEN_UNITS_DEFAULT
    g_drop_prob: float = G_DROP_PROB_DEFAULT
    d_drop_prob: float = D_DROP_PROB_DEFAULT
    num_batches: int = NUM_BATCHES_DEFAULT
    per_nth_batch_print_memory: int = PER_NTH_BATCH_PRINT_MEMORY_DEFAULT

