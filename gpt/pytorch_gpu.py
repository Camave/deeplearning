import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


# Embedding des tokens + positions
class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.embedding_tok = nn.Embedding(vocab_size, embed_dim)
        self.embedding_pos = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        batch_size, seq_length = x.size()
        token_emb = self.embedding_tok(x)
        pos = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        token_pos = self.embedding_pos(pos)
        embeddings = token_emb + token_pos
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, S, E = x.shape
        Q = self.Q(x)
        K = self.K(x)
        V = self.V(x)

        Q = Q.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:S, :S], float('-inf'))

        p = F.softmax(scores, dim=-1)
        p = self.attn_dropout(p)

        output_head = p @ V
        output = output_head.permute(0, 2, 1, 3).reshape(B, S, E)

        output = self.out(output)
        output = self.resid_dropout(output)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, max_seq_len, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads, max_seq_len, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(ff_hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.dropout(self.fc2(self.activation(self.fc1(self.norm2(x)))))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.emb = GPTEmbedding(vocab_size, embed_dim, max_seq_len)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_hidden_dim, max_seq_len, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, vocab_size, bias=False)
        self.proj.weight = self.emb.embedding_tok.weight

    def forward(self, x):
        x = self.emb(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)
        x = self.proj(x)
        return x


# ---- ✅ FONCTION EVALUATION ----
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity.item()


# ---- ✅ BOUCLE TRAIN + VAL ----
def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        val_loss, val_ppl = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss:.4f}, Train PPL={train_ppl:.2f} | "
            f"Val Loss={val_loss:.4f}, Val PPL={val_ppl:.2f}"
        )


# ---- DATASET ----
def build_dataset(encoded_text, seq_length):
    inputs, targets = [], []
    for i in range(len(encoded_text) - seq_length):
        x = encoded_text[i : i + seq_length]
        y = encoded_text[i+1 : i + seq_length + 1]
        inputs.append(x)
        targets.append(y)
    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)
    return X, Y


def generate(model, start_tokens, max_new_tokens, char2idx, idx2char, device, temperature=1.0):
    input = start_tokens.to(device)
    max_seq_len = model.emb.embedding_pos.num_embeddings  # récupère max_seq_len du modèle

    for _ in range(max_new_tokens):
        # Tronquer si la séquence devient trop longue
        if input.size(1) > max_seq_len:
            input = input[:, -max_seq_len:]

        logits = model(input)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        input = torch.cat([input, next_token], dim=1)

    output_text = "".join([idx2char[int(i)] for i in input[0]])
    return output_text



# ---- MAIN ----
text = open("input.txt").read()
vocab = sorted(set(text))
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for i, ch in enumerate(vocab)}

encoded = [char2idx[c] for c in text]
X, Y = build_dataset(encoded, seq_length=16)

vocab_size = len(vocab)
embed_dim = 256
num_heads = 8
ff_hidden_dim = 512
num_layers = 4
max_seq_len = 64
dropout = 0.1
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)
model = MiniGPT(vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len, dropout)
model.to(device)

dataset = torch.utils.data.TensorDataset(X, Y)

# ✅ Split 90% train / 10% val
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=200)

torch.save(model.state_dict(), "shakespeare_gpt.pth")

# (Optionnel) Sauvegarder le vocabulaire
import pickle
with open("char_mapping.pkl", "wb") as f:
    pickle.dump((char2idx, idx2char), f)
