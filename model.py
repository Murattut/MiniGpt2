"""
    Code by Murat Tut(@Mrtut)
    This is a simple implementation of GPT2 model
    I used the code from the following OpenAI's blog paper as a reference
    https://openai.com/blog

    It is an implementation of a GPT2 model. However, it is not the same as the original model by OpenAI because,
    since it is originally written with tensorflow with version ~1.
    I write The code with PyTorch, and it is optimized for speed* and memory usage.

    * I use new PyTorch features which are highly optimized for speed and memory usage.

"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from config import hyperparameters_for_data, hyperparameters_for_model

config_data = hyperparameters_for_data()
config_model = hyperparameters_for_model()

device = config_model.device
n_emb = config_model.n_embd
block_size = config_model.block_size
attn_pdrop = config_model.attn_pdrop
n_layer = config_model.n_layer
n_head = config_model.n_head
resid_pdrop = config_model.resid_pdrop
embd_pdrop = config_model.embd_pdrop

# vocab_size = config_data.vocab_size
vocab_size = config_data.vocab_size

class GPT2MODEL(nn.Module):
    def __init__(self):
        super(GPT2MODEL, self).__init__()

        self.device = device
        self.block_size = block_size

        self.vocab_emd = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_emb)
        self.pos_emd = nn.Embedding(num_embeddings=block_size, embedding_dim=n_emb)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
        self.drop = nn.Dropout(embd_pdrop)
        self.normalize = nn.LayerNorm(n_emb)
        self.layer = nn.Linear(n_emb, vocab_size, bias=False)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('normalize.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        """
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)"""

    def forward(self, index, targets=None):
        """
        :param targets: torch.Tensor, shape (batch_size, block_size) B, T
        :param index: torch.Tensor, shape (batch_size, block_size) B, T
        index = x this is just name
        """

        _, T = index.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        posx = torch.arange(0, T, device=self.device, dtype=torch.long)
        x = self.vocab_emd(index) + self.pos_emd(posx)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.normalize(x)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.layer(x)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.layer(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    """
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss
    """

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=True, top_k=None):
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.attn = MultiHeadAttention()
        self.ln1 = nn.LayerNorm(n_emb)
        self.ffn = FeedForward()
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        """
        y = self.mha(x)
        x = self.ln1(x + y)
        y = self.ffn(x)
        x = self.ln2(x + y)
        return x
        not this order can be change
        """

        # x = self.ln1(x + self.attn(x))
        # return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self):
        super().__init__()
        # self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        assert n_emb % n_head == 0, "n_embd must be divisible by n_head"
        head_size = n_emb // n_head
        self.mha_flash_attn = Attention(head_size)
        # self.proj = nn.Linear(head_size * num_heads, n_embd, dtype=torch.float16)
        self.proj = nn.Linear(head_size, n_emb)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        return self.dropout(self.proj(self.mha_flash_attn(x)))


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.GELU(),
            nn.Linear(4 * n_emb, n_emb),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size: int):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.dropout_attn = config_model.attn_pdrop

    def forward(self, x):
        print(x.shape)
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        v = self.value(x)  # (B, T, hs)

        return F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout_attn)




"""
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        v = self.value(x)  # (B, T, hs)

        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = self.dropout(F.softmax(wei, dim=-1))  # (B, T, T))
        # perform the weighted aggregation of the values
        return wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
"""
