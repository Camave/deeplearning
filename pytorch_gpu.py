import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
from google.colab import files



# Embedding des tokens + positions
class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        # TODO: définir token embedding + positional embedding
        self.embedding_tok = nn.Embedding(vocab_size,embed_dim)
        self.embedding_pos = nn.Embedding(max_seq_len,embed_dim)


    def forward(self, x):
        # TODO: renvoyer embeddings + positions
        batch_size, seq_length = x.size()
        token_emb = self.embedding_tok(x)
        pos = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, seq_length)
        token_pos = self.embedding_pos(pos)
        embeddings = token_emb + token_pos
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Définir les projections
        self.Q = nn.Linear(embed_dim, embed_dim)
        self.K = nn.Linear(embed_dim, embed_dim)
        self.V = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        B, S, E = x.shape  # batch_size, seq_len, embed_dim

        # Projections linéaires
        Q = self.Q(x)  # [B, S, E]
        K = self.K(x)
        V = self.V(x)

        # Split en têtes
        Q = Q.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, S, D]
        K = K.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        #calcul
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()  # True pour bloquer
        scores = scores.masked_fill(mask, float('-inf'))
        p = F.softmax(scores, dim=-1)
        output_head = p @ V
        output = output_head.permute(0, 2, 1, 3).reshape(B, S, E)
        output = self.out(output)

        return output


# Bloc Transformer (un bloc de GPT)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        # TODO: attention + layernorm + feedforward + résidu
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.P_couche = nn.Linear(embed_dim, ff_hidden_dim)
        self.activation = nn.GELU()
        self.D_couche = nn.Linear(ff_hidden_dim, embed_dim)

    def forward(self, x, mask=None):
        # TODO: forward complet du bloc
        x_norm1 = self.norm1(x)
        att_out = self.attention(x_norm1)
        x = x + att_out
        x_norm2 = self.norm2(x)
        ff_out = self.P_couche(x_norm2)     # Linear(embed_dim → ff_hidden_dim)
        ff_out = self.activation(ff_out)   # GELU
        ff_out = self.D_couche(ff_out)     # Linear(ff_hidden_dim → embed_dim)
        x = x + ff_out
        return x


# Le modèle complet GPT
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len):
        super().__init__()
        # TODO: embedding, plusieurs blocs transformer, couche finale
        self.emb = GPTEmbedding(vocab_size, embed_dim, max_seq_len)
        self.list_trans = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)]
            )
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        # TODO: appliquer embeddings -> blocs -> projection finale
        x = self.emb(x)
        for trans in self.list_trans:
            x = trans(x, mask=mask)
        x = self.proj(x)
        return x


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    num_epochs = 10  # ou plus
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader):
            inputs, targets = batch
            # TODO: envoyer sur device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # TODO: forward
            logits = model(inputs)
            # TODO: calcul loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            # TODO: backward + step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def build_dataset(encoded_text, seq_length):
    inputs = []
    targets = []
    for i in range(len(encoded_text) - seq_length):
        x = encoded_text[i : i + seq_length]
        y = encoded_text[i+1 : i + seq_length + 1]  # décalé d’un cran
        inputs.append(x)
        targets.append(y)
    # Convertir en tenseurs
    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)
    return X, Y

def generate(model, start_tokens, max_new_tokens, char2idx, idx2char, device, temperature=1.0):
    """
    Génère du texte à partir d'un prompt initial.
    """
    # 1. Mettre start_tokens dans un tenseur sur le bon device
    input = start_tokens.to(device)
    # 2. Boucle de génération
    for _ in range(max_new_tokens):
        # a. Passer input dans le modèle → logits
        logits = model(input)
        # b. Prendre seulement les logits du dernier token
        logits = logits[:, -1, :]
        # c. Appliquer la température
        logits = logits/temperature
        # d. Calculer probabilités avec softmax
        probs = F.softmax(logits, dim=-1)
        # e. Échantillonner un token (ou argmax)
        next_token = torch.multinomial(probs, 1)
        # f. Ajouter ce token à la séquence
        input = torch.cat([input, next_token], dim=1)
    # 3. Décoder la séquence complète en texte
    output_text = "".join([idx2char[int(i)] for i in input[0]])
    return output_text

uploaded = files.upload()
text = open("input.txt").read()[:100000]
vocab = sorted(set(text))
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for i, ch in enumerate(vocab)}

encoded = [char2idx[c] for c in text]

X, Y = build_dataset(encoded, seq_length=3)

vocab_size = len(vocab)
embed_dim = 8
num_heads = 2
ff_hidden_dim = 32
num_layers = 1
max_seq_len = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)
model = MiniGPT(vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers, max_seq_len)
model.to(device)

batch_size = 2
seq_len = max_seq_len
num_batches = 5

dataset = torch.utils.data.TensorDataset(X, Y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train(model, dataloader, optimizer, criterion, device)

logits = model(X)
print(logits.shape)  # doit afficher [2, 10, 50]

loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
perplexity = torch.exp(loss)

encoded = [char2idx[c] for c in "hello"]

start_tokens = torch.tensor([encoded], dtype=torch.long)

text =generate(model, start_tokens, 5, char2idx, idx2char, device, temperature=0.7)
print(loss)
print(perplexity)
print(text)