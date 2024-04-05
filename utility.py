"""
    code by Murat Tut(@Mrtut)
"""
import pickle

import numpy as np
import torch
from tqdm.auto import tqdm

from config import hyperparameters_for_model
from dataset import get_batch_train, get_batch_val, encode, decode

config_model = hyperparameters_for_model()

device = config_model.device
max_iters = config_model.max_iters
eval_interval = config_model.eval_interval


def performance_mod(model: torch.nn, optimizer: torch.optim):
    losses = {}
    temp_train_pointer = 0
    temp_val_pointer = 0
    for _ in tqdm(range(max_iters)):
        model.train()
        xb, yb, train_file_pointer = get_batch_train(temp_train_pointer)
        _, loss = model.forward(xb.to(device), yb.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses["train"] = loss.item()
        temp_train_pointer = train_file_pointer

    with torch.no_grad():
        model.eval()
        X, Y, val_file_pointer = get_batch_val(temp_val_pointer)
        _, loss = model.forward(X.to(device), Y.to(device))
        losses["val"] = loss.item()
        print(
            f"step {max_iters}"
            f": train loss {losses['train']:.4f},"
            f" train accuracy % {np.exp(-losses['train']) * 100:.4f}",
            f" val loss {losses['val']:.4f},"
            f" val accuracy % {np.exp(-losses['val']) * 100:.4f}")
        temp_val_pointer = val_file_pointer

    return model, losses


def developer_mod(model: torch.nn, optimizer: torch.optim):
    losses = {}
    temp_train_pointer = 0
    temp_val_pointer = 0
    model.train()
    for iter in tqdm(range(max_iters)):
        xb, yb, train_file_pointer = get_batch_train(temp_train_pointer)
        _, loss = model.forward(xb.to(device), yb.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses["train"] = loss.item()
        temp_train_pointer = train_file_pointer
        if iter % eval_interval == 0:
            with torch.no_grad():
                model.eval()
                X, Y, val_file_pointer = get_batch_val(temp_val_pointer)
                _, loss = model.forward(X.to(device), Y.to(device))
                losses["val"] = loss.item()
                print(
                    f"step {iter}"
                    f": train loss {losses['train']:.4f},"
                    f" train accuracy % {np.exp(-losses['train']) * 100:.4f}",
                    f" val loss {losses['val']:.4f},"
                    f" val accuracy % {np.exp(-losses['val']) * 100:.4f}")
                temp_val_pointer = val_file_pointer
            model.train()
    return model, losses


def generate_token(model: torch.nn, prompt: str):
    model.eval()
    context = torch.tensor(encode(prompt.lower()), dtype=torch.long, device=device)
    generated_data = model.generate(context.unsqueeze(0), max_new_tokens=50)[0].tolist()
    generated_words = decode(generated_data)
    print(generated_words)


def save_model(model: torch.nn):
    with open("Model/gpt2_v1.pkl", "wb") as file:
        pickle.dump(model, file)


def reload_model():
    with open("Model/gpt2_v1.pkl", "rb") as file:
        model = pickle.load(file)
    return model
