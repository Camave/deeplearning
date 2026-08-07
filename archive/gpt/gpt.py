import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import tiktoken

torch.cuda.empty_cache()


# --------------hyperparameters-----------------
block_size = 64
batch_size = 32
lr = 3e-4
num_epoch = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
eval_interval = 500
n_embd = 128
n_head = 4
dropout = 0.2
n_layer = 2

torch.manual_seed(1337)

# --------------data preparation-----------------
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
tokenizer = tiktoken.get_encoding("gpt2")
vocab_size = tokenizer.n_vocab

# === Exemple de texte ===
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
# train and validation split
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# --------------data loading-----------------
def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()  #  désactive le calcul des gradients (plus rapide)
def estimate_loss():
    out = {}
    model.eval()  #  passe en mode "évaluation" (désactive dropout, etc.)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)  # extrait un mini-batch
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  #  repasse en mode entraînement
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.Q = nn.Linear(n_embd, head_size, bias=False)
        self.K = nn.Linear(n_embd, head_size, bias=False)
        self.V = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        q = self.Q(x)  # (B, T, head_size)
        k = self.K(x)  # (B, T, head_size)
        wei =  q @ k.transpose(-2, -1) * k.shape[-1]**0.5 # (B, T, head_size) * (B, head_size, T)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim =-1)
        wei = self.drop(wei) # (B, T, T)
        v = self.V(x)  # (B, T, head_size)
        out =  wei @ v 
        
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads,n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.drop(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd,4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd,n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self,n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
# --------------model definition-----------------
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, target=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        postion = self.position_embedding_table(torch.arange(T, device=device).clamp(max=block_size-1)) # (T,C)
        x = token_emb + postion # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits,target)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

model = GPT(vocab_size)
model = model.to(device)

# create a Pytorch optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    # evaluate modele
    if epoch % eval_interval == 0 or epoch == num_epoch - 1:
        losses = estimate_loss()
        print(epoch, losses)

    # get batch
    xb, yb = get_batch("train")

    # training
    logits,loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()    
    if epoch == 0 :    
        print(tokenizer.decode(model.generate(idx = torch.zeros((1,1), dtype= torch.long, device=device),max_new_tokens=500)[0].tolist()))


print(tokenizer.decode(model.generate(idx = torch.zeros((1,1), dtype= torch.long, device=device),max_new_tokens=500)[0].tolist()))
