import torch
import gc
import json
import os
import math
import statistics
import matplotlib.pyplot as plt
from attention import dense
from attention import sparse
from masks import mask
from slidingmasks import sliding_win
from bigbirdmasks import bigbird
import urllib.request

if not os.path.exists('input.txt'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', 'input.txt')
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)
stoi = {ch:i for i , ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
#print(encode("hello"))
data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9*(len(data)))
train_data = data[:n]
val_data = data[n:]
max_iters = 10000
eval_interval = 300
eval_iters = 200
batch_size = 32 # number of independent sequences
block_size = 128#seq_len
n_embd = 128
n_head = 4
n_layer = 2
def get_batch(split):
    # generates a small btach of data
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data)- block_size,(batch_size,)) # random offsets 
    x = torch.stack([data[i:i+block_size]for i in ix] )
    y = torch.stack([data[i+1:i+block_size+1] for i in ix] )
    x, y = x.to(device), y.to(device)
    return x,y 
import torch.nn as nn

class GPT(nn.Module):
  def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )#[32, 128, 128]
        self.position_embedding = nn.Embedding(
            block_size,n_embd
        )
  class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        T = x.shape[1]
        attention=[mask(q),sliding_win(block_size,4,"cuda"),bigbird(q,k,v,4,2,2,"cuda")]
       names = ["Dense","Sliding Window","BigBird"]

    for i in range(len(attention)):

    out = dense(q, k, v, attention[i])

    print(names[i])
    print(out)
