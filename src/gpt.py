import os
import urllib.request
import torch
import torch.nn as nn
from attention import dense
from masks import mask
from slidingmasks import sliding_win
from bigbirdmasks import bigbird
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

if not os.path.exists("input.txt"):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "input.txt"
    )

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 32
block_size = 128
n_embd = 128
n_head = 4
n_layer = 2
max_iters = 1000
eval_interval = 100
eval_iters = 50
learning_rate = 3e-4

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

def run_attention(q, k, v, attention_type):
    if attention_type == "dense":
        attn_mask = mask(q)

    elif attention_type == "sliding":
        attn_mask = sliding_win(
            q.shape[-2],
            4,
            q.device
        )

    elif attention_type == "bigbird":
        attn_mask = bigbird(
            q,
            k,
            v,
            4,
            2,
            2,
            q.device
        )

    else:
        raise ValueError("Unknown attention type")

    return dense(q, k, v, attn_mask)

class Head(nn.Module):
    def __init__(self, head_size, attention_type):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.attention_type = attention_type

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        return run_attention(q, k, v, self.attention_type)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, attention_type):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, attention_type)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, attention_type):
        super().__init__()
        head_size = n_embd // n_head
        self.attention = MultiHeadAttention(
            n_head,
            head_size,
            attention_type
        )
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, attention_type):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[
            TransformerBlock(attention_type)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            loss = nn.functional.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss

@torch.no_grad()
def estimate_loss(model):
    model.eval()
    losses = []

    for _ in range(eval_iters):
        x, y = get_batch("val")
        _, loss = model(x, y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)
def train_model(attention_type):
    print("\nTraining:", attention_type)

    model = GPT(attention_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    steps = []
    train_losses = []
    val_losses = []

    for step in range(max_iters):
        x, y = get_batch("train")
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            val_loss = estimate_loss(model)

            steps.append(step)
            train_losses.append(loss.item())
            val_losses.append(val_loss)

            print(
                step,
                "train:",
                round(loss.item(), 4),
                "val:",
                round(val_loss, 4)
            )

    return model, steps, train_losses, val_losses


results = {}

for attention_type in ["dense", "sliding", "bigbird"]:
    model, steps, train_losses, val_losses = train_model(attention_type)

    results[attention_type] = {
        "steps": steps,
        "train": train_losses,
        "val": val_losses
    }

plt.figure(figsize=(8, 5))

for name in results:
    plt.plot(
        results[name]["steps"],
        results[name]["train"],
        label=name
    )

plt.xlabel("Training step")
plt.ylabel("Training loss")
plt.title("Training Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))

for name in results:
    plt.plot(
        results[name]["steps"],
        results[name]["val"],
        label=name
    )

plt.xlabel("Training step")
plt.ylabel("Validation loss")
plt.title("Validation Loss")
plt.legend()
plt.grid()
plt.show()

print("\nFinal validation losses:")

for name in results:
    print(
        name,
        round(results[name]["val"][-1], 4)
    )
