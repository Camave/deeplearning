import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
from datasets import load_from_disk
from transformers import AutoTokenizer
from datasets import load_dataset

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===== PARAMÈTRES RÉDUITS =====

batch_size = 32       
block_size = 64     
n_embd = 194        
n_head = 4            
n_layer = 2         
dropout = 0.4
lr = 3e-4
epochs = 5000

eval_iters = 200
eval_interval = 500

temperature = 0.9

# ===== FIX FRAGMENTATION =====

torch.manual_seed(1337)

dataset = load_dataset("DKYoon/SlimPajama-6B")
ds_local = dataset["train"]

# 2. Initialiser le tokenizer (celui que vous utilisez pour GPT-2 est bon)
enc = AutoTokenizer.from_pretrained("gpt2") 
vocab_size = enc.vocab_size

# ===== NOUVELLE FONCTION DE PRÉTRAITEMENT =====

BLOCK_SIZE = block_size # Reprend la variable de votre script
# Fonction de tokenisation et de groupement (nécessaire pour transformer le texte en blocs)
def tokenize_and_group(examples):
    # 1. Tokenisation du texte
    tokenized = enc(
        examples["text"],
        truncation=False,
        return_overflowing_tokens=True,
        padding=False,
        # Ajoutez ceci pour simplifier la sortie:
        return_attention_mask=False 
    )
    
    # 2. Concaténation et groupement en blocs de BLOCK_SIZE
    # Cette étape est simplifiée car 'tokenized' ne contient que des 'input_ids' et 'overflow_mapping'
    
    # Nous allons concaténer toutes les listes d'input_ids générées par le tokenizer
    all_input_ids = tokenized["input_ids"]
    
    # 'sum(all_input_ids, [])' est la syntaxe correcte pour concaténer une liste de listes
    concatenated_ids = sum(all_input_ids, [])
    total_length = len(concatenated_ids)
    
    # Suppression des jetons à la fin pour ne garder que des blocs complets
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    
    # Division en blocs de BLOCK_SIZE
    result = {
        "input_ids": [
            concatenated_ids[i : i + BLOCK_SIZE]
            for i in range(0, total_length, BLOCK_SIZE)
        ]
    }
    
    # Le label est l'input (standard pour l'entraînement LLM)
    result["labels"] = result["input_ids"].copy()
    
    return result

# 3. Appliquer la tokenisation
tokenized_dataset = ds_local.map(
    tokenize_and_group, 
    batched=True, 
    remove_columns=ds_local.column_names,
    desc="Tokenisation et groupement en blocs..."
)


# 4. Séparation en données d'entraînement et de validation
# Note : Nous travaillons maintenant avec des blocs de tokens, pas du texte brut
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=1337)
train_dataset = tokenized_dataset["train"]
val_dataset = tokenized_dataset["test"]

print("✅ Préparation des données terminée.")
print(f"Entraînement : {len(train_dataset)} blocs | Validation : {len(val_dataset)} blocs")

def get_batch(split):
    if split == "train":
        data = train_dataset
    else:
        data = val_dataset
        
    # 1. Sélectionner un index aléatoire à partir du dataset
    ix = torch.randint(len(data), (batch_size,))
    
    # 2. Extraire les exemples
    # On utilise la méthode 'select' du dataset pour choisir les indices
    batch_data = data.select(ix.tolist())
    
    # 3. Convertir en tensors PyTorch (la clé est 'input_ids' et 'labels')
    # Les données sont déjà découpées en blocs de 'block_size'
    
    X = torch.tensor(batch_data["input_ids"], dtype=torch.long)
    Y = torch.tensor(batch_data["labels"], dtype=torch.long)
    
    X, Y = X.to(device), Y.to(device)
    
    # 4. Ajouter les dimensions manquantes si nécessaire (la fonction map fait déjà le travail)
    # Dans votre cas, la forme est déjà (batch_size, block_size)
    
    return X, Y

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

    def generate(self, idx, max_new_tokens, top_k=None, repetition_penalty=1.2):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            
            # --- AJOUT DE LA PÉNALITÉ DE RÉPÉTITION ---
            # 1. Identifier les jetons déjà dans le contexte (idx)
            # 2. Créer un masque/une pénalité pour ces jetons
            for i in range(idx.size(0)): # Pour chaque exemple dans le batch (ici 1)
                context_tokens = idx[i].unique()
                for token_id in context_tokens:
                    # Pénaliser les jetons déjà vus dans le contexte
                    if logits[i, token_id] > 0:
                        logits[i, token_id] /= repetition_penalty # Diminuer le score si le logit est positif
                    else:
                        logits[i, token_id] *= repetition_penalty # Augmenter la pénalité si le logit est négatif
            # ------------------------------------------

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

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
        initial_prompt = "The quick brown fox jumps over the lazy"
        initial_tokens = enc.encode(initial_prompt)
        context = torch.tensor([initial_tokens], dtype=torch.long, device=device)

        # Appelez la génération avec le nouveau contexte
        print(enc.decode(model.generate(idx=context, max_new_tokens=500, top_k=50, repetition_penalty=1.2)[0].tolist()))

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
    generated = model.generate(idx=context, max_new_tokens=500, top_k = 50, repetition_penalty=1.2)
    text = enc.decode(generated[0].tolist())
    print(text)


MODEL_DIR = "./checkpoints"
MODEL_NAME = "camille0_weight_petit.pth"

os.makedirs(MODEL_DIR, exist_ok=True) 

PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# --- Sauvegarde ---
# On enregistre le dictionnaire d'état du modèle
torch.save(model.state_dict(), PATH)

print(f"✅ Modèle sauvegardé avec succès à : {PATH}")