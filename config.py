"""
    code by Murat Tut(@Mrtut)
    config file contains the hyperparameters and the device selection

    it is still under construction

"""
from torch.cuda import is_available as cuda_available
from torch.backends.mps import is_available as mps_available
import json


# write the path of the data root

def get_device():
    # select device
    if cuda_available():
        device = "cuda"
    elif mps_available():
        device = "mps"
    # if mps rise an error use cpu
    else:
        device = "cpu"
    return device


# hyperparameters
class hyperparameters_for_model:
    def __init__(
            self,
            batch_size=32,  # 32 64                original Gpt2 = 64 (need control)
            block_size=128,  # 8 64 128 256        original Gpt2 = 1024
            max_iters=10,  # 200  1000           original Gpt2 = 200
            eval_interval=100,  #                  original Gpt2 = 200
            n_embd=128,  # 128 256                 original Gpt2 = 768
            n_layer=4,  # 2                        original Gpt2 = 12
            n_head=4,  # 2                         original Gpt2 =1 2
            learning_rate=2e-3,  # 1e-2 1e-3       original Gpt2 = 5e-4 (need control)
            embd_pdrop=0.1,  # 0.1 0.2             original Gpt2 = 0.1 (need control)
            attn_pdrop=0.1,  # 0.1 0.2             original Gpt2 = 0.1 (need control)
            resid_pdrop=0.1,  # 0.1 0.2            original Gpt2 = 0.1 (need control)
            train_mode="developer"  # "performance" "developer" "research",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size

        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head

        self.learning_rate = learning_rate

        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop

        self.max_iters = max_iters
        self.eval_interval = eval_interval

        self.device = get_device()

        self.train_mode = train_mode


class hyperparameters_for_data:
    def __init__(
            self,
            data_root="Data/",
            word_files_root="openwebtext_token/min_word_count.json",
            train_files_root="openwebtext_only_english/train.txt",
            val_files_root="openwebtext_only_english/val.txt",

    ):
        super().__init__()
        self.data_root = data_root
        self.word_files_root = data_root + word_files_root
        self.train_files_root = data_root + train_files_root
        self.val_files_root = data_root + val_files_root
        self.vocab_size = 968  #  len(self.read_words().items())

    def read_words(self):
        with open(self.word_files_root, 'r') as file:
            return json.load(file)
