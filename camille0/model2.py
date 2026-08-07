import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== PARAMÈTRES RÉDUITS =====

batch_size = 64       
block_size = 128     
n_embd = 384         
n_head = 6            
n_layer = 6          
dropout = 0.4
lr = 3e-4
epochs = 1000

eval_iters = 200
eval_interval = 500

# ===== FIX FRAGMENTATION =====

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
tokens = enc.encode(text)
data = torch.tensor(tokens, dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    if split == "train":
        data = train_data
    else:
        data = val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()  #  désactive le calcul des gradients (plus rapide)
def estimate_loss():
    out = {}
    model.eval()  #  passe en mode "évaluation" (désactive dropout, etc.)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)  # extrait un mini-batch
            with torch.amp.autocast("cuda"):
                logits = model(X)
                B, T, C = logits.shape
                logit = logits.view(B*T, C)
                target = Y.view(B*T)
                loss = F.cross_entropy(logit, target)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()  #  repasse en mode entraînement
    return out

# ===== TON MODÈLE =====
class attention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.drop = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, Q, K, V):
        B, T, C = Q.shape
        wei = Q @ K.transpose(-2, -1) / self.head_size**0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        out = wei @ V
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.Q = nn.Linear(n_embd, head_size, bias=False)
        self.K = nn.Linear(n_embd, head_size, bias=False)
        self.V = nn.Linear(n_embd, head_size, bias=False)
        self.Heads = nn.ModuleList([attention(head_size) for i in range(n_head)])
        self.lin = nn.Linear(head_size*n_head, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)
        out = torch.cat([head(Q, K, V) for head in self.Heads], dim=-1)
        out = self.drop(out)
        out = self.lin(out)
        return out

class Feed_Forward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.FN = nn.Sequential(
            nn.Linear(n_embd, n_embd*4),
            nn.GELU(),
            nn.Linear(n_embd*4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.FN(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.MHA = MultiHeadAttention(head_size)
        self.FNN = Feed_Forward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.MHA(self.ln1(x))
        x = x + self.FNN(self.ln2(x))
        return x

class Camille0(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding_tok = nn.Embedding(vocab_size, n_embd)
        self.embedding_pos = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for i in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T = x.shape
        token_emb = self.embedding_tok(x)
        postion = self.embedding_pos(torch.arange(T, device=device).clamp(max=block_size-1))
        x = token_emb + postion
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

# ===== INITIALISATION =====
model = Camille0(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scaler = torch.amp.GradScaler("cuda")

print(f"✅ Modèle sur: {next(model.parameters()).device}")
n_params = sum(p.numel() for p in model.parameters())
print(f"✅ Paramètres: {n_params/1e6:.1f}M")
print(f"✅ Batch size: {batch_size}, Block size: {block_size}")
print("="*60)


# ===== BOUCLE D'ENTRAÎNEMENT (FIXÉE) =====
for epoch in range(epochs):

    if epoch % eval_interval == 0 or epoch == epochs - 1:
        losses = estimate_loss()
        print(epoch, losses)

    xb, yb = get_batch("train")
    
    #forward pass en FP16
    with torch.amp.autocast("cuda"):
        logits = model(xb)
        B, T, C = logits.shape
        logit = logits.view(B*T, C)
        target = yb.view(B*T)
        loss = F.cross_entropy(logit, target)

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    #multiplie la loss par un facteur avant le backward pour augmenter les gradients
    scaler.scale(loss).backward()
    #divise les gradients par le facteur de scaling pour revenir à l’échelle normale
    scaler.unscale_(optimizer)
    #limite la norme des gradients à 1.0 pour pas explose les gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #applique la mise à jour des poids si les gradients sont valides
    scaler.step(optimizer)
    #ajuste automatiquement le facteur de scaling pour le prochain batch.
    scaler.update()

    # ===== MONITORING (APRÈS backward) =====
    if epoch % 100 == 0 or epoch == epochs-1:
        # Maintenant la mémoire inclut les gradients
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | "
              f"VRAM: {allocated:.2f}/{reserved:.2f} GB")
    
    # Libère la mémoire régulièrement
    if epoch % 500 == 0 and epoch > 0:
        torch.cuda.empty_cache()

# ===== GÉNÉRATION =====
print("\n" + "="*60)
print("🎨 GÉNÉRATION DE TEXTE")
print("="*60)

model.eval()  # Mode évaluation
with torch.no_grad():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(idx=context, max_new_tokens=500)
    text = enc.decode(generated[0].tolist())
    print(text)