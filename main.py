"""
    code by Murat Tut(@Mrtut)
"""
import torch
import torchinfo

from config import hyperparameters_for_data, hyperparameters_for_model
from model import GPT2MODEL
from utility import performance_mod, developer_mod, generate_token, reload_model, save_model

import time


def d_type_founder():
    if device == "cuda":
        return torch.float32
    elif device == "mps":
        return torch.float32  # you can also use torch.bfloat16
    else:
        return torch.float32


# load the hyperparameters
config_model = hyperparameters_for_model()
config_data = hyperparameters_for_data()

device = config_model.device
train_mode = config_model.train_mode
learning_rate = config_model.learning_rate

vocab_size = config_data.vocab_size


def main():
    print('Vocabulary size =', vocab_size)

    print('Device =', device)

    d_type = d_type_founder()

    model = GPT2MODEL().to(device=device, dtype=d_type)

    # model = reload_model().to(device=config.device)

    print(torchinfo.summary(model, device=device))

    # optimizer can be changed AdamW to. Adam, SGD, etc. but AdamW is recommended
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if train_mode == "performance":
        model, losses = performance_mod(model, optimizer)
    elif train_mode == "developer":
        model, losses = developer_mod(model, optimizer)
    else:
        raise ValueError("Train mode should be either 'performance' or 'developer'")

    while True:
        prompt = input("Please enter a prompt,"
                       " for exit write exit: ")
        if prompt == "exit":
            break
        generate_token(model, prompt)


start = time.process_time()
if __name__ == '__main__':
    main()

end = time.process_time()
print("total Elapsed time:", end - start)
