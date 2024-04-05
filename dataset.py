"""
    code by Murat Tut(@Mrtut)
    dataset file contains the data reading and encoding functions
    it is still under construction
    some functions are broken* and need to be fixed

    * read_large_text_file, it seems like it is not working properly
    * encode word part
    * decode word part
"""

import torch

from config import hyperparameters_for_data, hyperparameters_for_model

config_data = hyperparameters_for_data()
config_model = hyperparameters_for_model()

device = config_model.device
block_size = config_model.block_size
batch_size = config_model.batch_size

stoi = config_data.read_words()
itos = {ch: i for i, ch in stoi.items()}
itos[1] = " "
itos[2] = " "


def read_file(file_root, file_pointer):
    temp_list = []

    # Open the file in read mode
    with open(file_root, 'r') as f:
        f.seek(file_pointer)
        line = f.readline()
        while line:
            for word in line.split():
                if word in stoi:
                    temp_list.append(stoi["<start>"])
                    temp_list.append(stoi[word])
                    temp_list.append(stoi["<end>"])
                    if len(temp_list) >= block_size * batch_size + 1:
                        return temp_list, f.tell()
            line = f.readline()
    #return temp_list, None


def decode(l: list):
    """
    :param l: input list of integers
    :return: string decoded
    """

    return ''.join([itos[i] for i in l])


def encoded_data_batch(raw_data):
    temp_list = []
    encoded_data = torch.tensor(raw_data, dtype=torch.int32)
    for i in range(batch_size):
        step = i * block_size
        temp_list.append(encoded_data[step: step + block_size + 1])
    block_data = torch.stack(temp_list)
    temp_list.clear()
    return block_data


def encode(raw_data: str):
    # Convert characters to numeric base
    return [stoi[i] if i in stoi else stoi["<oov>"] for i in raw_data]


def get_batch_train(file_pointer):
    train_data, new_file_pointer = read_file(config_data.train_files_root, file_pointer)
    block_data = encoded_data_batch(train_data)
    return block_data[:, :-1], block_data[:, 1:], new_file_pointer


def get_batch_val(file_pointer):
    val_data, new_file_pointer = read_file(config_data.val_files_root, file_pointer)
    block_data = encoded_data_batch(val_data)
    return block_data[:, :-1], block_data[:, 1:], new_file_pointer



