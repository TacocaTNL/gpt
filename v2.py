import requests
import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # how many samples to process at once?
block_size = 8 # what is the maximum length of a prediction?
max_iters = 5000 # how many iterations to train for?
eval_interval = 500 # how often to evaluate the model?
lr = 1e-3 # learning rate
test_size = 0.1 # what fraction of the data to use for testing?
device = 'cuda' if torch.cuda.is_available() else 'cpu' # use the GPU if you have one
eval_iters = 100 # how many iterations to evaluate for?
n_embd = 32 # number of embedding dimensions

torch.manual_seed(42) # for reproducibility

# Load the data
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
data = response.text

# Create a vocabulary
chars = sorted(list(set(data)))
vocab_size = len(chars)
char_to_int = {ch: i for i, ch in enumerate(chars)}
int_to_char = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [char_to_int[ch] for ch in s] # encoder: string -> list of ints
decode = lambda x: ''.join([int_to_char[i] for i in x]) # decoder: list of ints -> string

# Train / test split
train_data = torch.tensor(encode(data), dtype=torch.long)
n = int(test_size * len(train_data))
train_data = train_data[n:]
test_data = train_data[:n]

# Data loader
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Estimate the loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    ''' one head of self-attention'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # public information about token itself
        self.query = nn.Linear(n_embd, head_size, bias=False) # public information that token wants to have
        self.value = nn.Linear(n_embd, head_size, bias=False) # private information about token
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # lower triangular matrix

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # shape: (B, T, C)
        q = self.query(x) # shape: (B, T, C)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, T, C) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # create decoder-block. (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # perform weighted aggregation of values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd) 
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx: (batch_size, block_size) (B, T)
        # targets: (batch_size, block_size) (B, T)
        tok_emb = self.token_embedding(idx) # shape: (batch_size, block_size, n_embd) (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T, device=device)) # shape: (block_size, n_embd) (T, C)
        concat_emb = tok_emb + pos_emb # shape: (B, T, C)
        concat_emb = self.sa_head(concat_emb) # apply one head of self-attention (B, T, C)
        logits = self.lm_head(concat_emb) # shape: (batch_size, block_size, vocab_size) (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(-1)

            # How well are we predicting the next token based on the logits?
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, steps):
        # idx: (batch_size, block_size) (B, T)
        for _ in range(steps):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # get the last time step
            logits = logits[:, -1, :] # shape: (B, C)
            # get the probabilities
            probs = F.softmax(logits, dim=-1) # shape: (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # shape: (B, 1)
            # append to the sequence
            idx = torch.cat((idx, idx_next), dim=1) # shape: (B, T+1)
        return idx
    
model = BigramLanguageModel().to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Training loop
for i in range(max_iters):
    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f'Iteration {i}, train loss: {losses["train"]}, test loss: {losses["test"]}')

    # Get a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate some text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, steps=500)[0].tolist()))