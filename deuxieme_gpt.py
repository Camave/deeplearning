import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# --------------hyperparameters-----------------
torch.manual_seed(1337)
block_size = 8
batch_size = 32
lr = 1e-3
num_epoch = 100000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
eval_interval = 1000
n_embd = 32

# --------------data preparation-----------------
text = open(r"c:\perso\projet_python\IA\LLM\Video\deeplearning\input.txt").read()
# all the unique characters in the text
vocab = sorted(set(text))
vocab_size = len(vocab)

# create a mapping from characters to integers
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for i, ch in enumerate(vocab)}
encode = lambda text : [char2idx[c] for c in text]
decode = lambda enc : ''.join([idx2char[i] for i in enc])

# train and validation split
data = torch.tensor(encode(text), dtype= torch.long)
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

# --------------model definition-----------------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        

    def forward(self, idx, target=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx) # (B,T,C)
        postion = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_emb + postion # (B,T,C)
        logits = self.lm_head(token_emb) # (B,T,vocab_size)
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
            logits, loss = self(idx)
            logits = logits[:,-1,:]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
model = model.to(device)

# create a Pytorch optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for epoch in range(num_epoch):
    # evaluate modele
    if epoch%eval_interval == 0:
        losses = estimate_loss()
        print(epoch, losses)

    # get batch
    xb, yb = get_batch("train")

    # training
    logits,loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if epoch ==0 :
        print(decode(model.generate(idx = torch.zeros((1,1), dtype= torch.long, device=device),max_new_tokens=500)[0].tolist()))


print(decode(model.generate(idx = torch.zeros((1,1), dtype= torch.long, device=device),max_new_tokens=500)[0].tolist()))