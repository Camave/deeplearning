"""
    NeuralFlow pour Camille0 - Visualisation des couches
"""
import os
import datetime
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer

# ==========================================
# 1. CONFIGURATION ET HYPERPARAMÈTRES
# ==========================================
# Doivent être identiques à ceux de ton entraînement !
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "./checkpoints/camille0_weight.pth" # Chemin vers ton fichier .pth
OUTPUT_FOLDER = "./visualisations/"             # Dossier de sortie
PROBE_STRING = "L'intelligence artificielle est" # Le texte à visualiser

# Hyperparamètres de Camille0 (copiés de ton script)
block_size = 128     
n_embd = 384         
n_head = 6            
n_layer = 6          
dropout = 0.0 # On met 0 pour la visu (pas besoin de dropout en inférence)

# ==========================================
# 2. DÉFINITION DE TON ARCHITECTURE
# (Copié-collé de ton code pour que le chargement fonctionne)
# ==========================================

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
        
        # Note: self.blocks est la partie importante pour la visualisation

# ==========================================
# 3. LOGIQUE DE VISUALISATION (NeuralFlow)
# ==========================================

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    print("1. Chargement du Tokenizer (GPT2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size

    print("2. Initialisation du modèle Camille0...")
    model = Camille0(vocab_size)
    
    print(f"3. Chargement des poids depuis {MODEL_PATH}...")
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        print("   Modèle chargé avec succès !")
    except Exception as e:
        print(f"ERREUR : Impossible de charger le modèle. Vérifiez le chemin. \n{e}")
        return

    print(f"4. Analyse de la phrase : '{PROBE_STRING}'")
    probe_results = []
    # On calcule les sorties intermédiaires
    probe_result = compute_custom_model_output(model, tokenizer, PROBE_STRING)
    probe_results.append(probe_result)

    print("5. Génération du GIF...")
    plot_embedding_flow(probe_results)
    print(f"✅ Terminé ! Vérifiez le dossier {OUTPUT_FOLDER}")


def compute_custom_model_output(model, tokenizer, text):
    """
    Cette fonction remplace la forward pass standard.
    Elle capture l'état (hidden_state) après chaque bloc.
    """
    with torch.no_grad():
        layer_outputs = []

        # A. Tokenisation
        input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
        B, T = input_ids.shape

        # B. Embedding initial (Tok + Pos)
        # On reproduit exactement le début de ton forward()
        token_emb = model.embedding_tok(input_ids)
        position = model.embedding_pos(torch.arange(T, device=device).clamp(max=block_size-1))
        x = token_emb + position
        x = model.drop(x)

        # On ajoute l'état initial (embedding)
        layer_outputs.append(x)

        # C. Boucle sur les blocs (self.blocks)
        for i, block in enumerate(model.blocks):
            x = block(x)
            # On stocke le résultat de ce bloc
            layer_outputs.append(x)
        
        # Optionnel : On peut ajouter la sortie finale après LayerNorm si on veut
        x = model.ln_f(x)
        layer_outputs.append(x)

        return layer_outputs


# --- Fonctions utilitaires de NeuralFlow (Couleur & Image) ---

def vectorized_get_color_rgb(value_tensor, max_value=1.0):
    # Transformation des valeurs numériques en couleurs HSL -> RGB
    h = (value_tensor * 1.0) / max_value
    s = torch.ones_like(h)
    v = torch.ones_like(h)
    c = v * s
    x = c * (1 - torch.abs((h * 6) % 2 - 1))
    m = v - c
    h1 = (h * 6).int()
    rgb = torch.stack((
        torch.where((h1 == 0) | (h1 == 5), c, torch.where((h1 == 1) | (h1 == 4), x, 0)),
        torch.where((h1 == 1) | (h1 == 2), c, torch.where((h1 == 0) | (h1 == 3), x, 0)),
        torch.where((h1 == 3) | (h1 == 4), c, torch.where((h1 == 2) | (h1 == 5), x, 0)),
    ), dim=-1) + m.unsqueeze(-1)
    return rgb

def generate_filename(prefix, extension):
    formatted_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{formatted_time}.{extension}"

def plot_layers(all_words, title, file_path):
    # Dimensions
    # all_words est une liste de tenseurs [Couche1, Couche2, ...]
    # Chaque tenseur est de forme (Batch, Seq_Len, Embed_Dim)
    sequence_length = all_words[0].shape[1]
    paths = []

    # Normalisation globale pour que les couleurs soient cohérentes
    all_words_cat = torch.cat(all_words, dim=0)
    global_mean = torch.mean(all_words_cat)
    global_var = torch.var(all_words_cat) * 25 # Facteur d'échelle pour le contraste
    min_val = global_mean - global_var
    max_val = global_mean + global_var

    # Pour chaque token de la phrase (axe du temps)
    for i in range(sequence_length):
        list_of_tensors = []
        for tensor in all_words:
            # On prend le vecteur d'embedding de ce token spécifique pour cette couche
            list_of_tensors.append(tensor[:, i, :])

        # Concaténation verticale des couches pour l'image
        # Shape: [Nombre_Couches, Embed_Dim]
        full_tensor = torch.cat(list_of_tensors, dim=0) 
        
        # Redimensionnement pour l'esthétique (split si trop large, sinon brut)
        # Ici on garde simple : on utilise la valeur absolue pour la couleur
        reshaped_tensor = torch.abs(full_tensor)

        # Normalisation
        normalized_data = (reshaped_tensor - min_val) / (max_val - min_val)
        # Clamp pour éviter les débordements de couleur
        normalized_data = torch.clamp(normalized_data, 0, 1)
        
        color_tensor = vectorized_get_color_rgb(normalized_data)

        # Création de l'image
        array = (color_tensor.cpu().numpy() * 255).astype(np.uint8)
        
        # On redimensionne l'image pour qu'elle soit visible (zoom x10 par exemple)
        image = Image.fromarray(array, 'RGB')
        image = image.resize((image.width * 4, image.height * 10), Image.NEAREST)

        # Sauvegarde frame par frame
        tmp_name = f"token_{i}"
        filename = f"{title}_{tmp_name}.png"
        full_path = os.path.join(file_path, filename)
        image.save(full_path)
        paths.append(full_path)

    # Création du GIF
    gif_name = f"{title}_flow.gif"
    gif_path = os.path.join(file_path, gif_name)
    with imageio.get_writer(gif_path, mode='I', fps=2, loop=0) as writer:
        for filename in paths:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Nettoyage des fichiers temporaires
    for filename in paths:
        try:
            os.remove(filename)
        except:
            pass
            
    return gif_path

def plot_embedding_flow(probe_results):
    # Adaptation structurelle pour plot_layers
    # probe_results[0] contient la liste des tenseurs par couche
    layer_embeddings = probe_results[0]
    
    # Plot
    plot_layers(layer_embeddings, "camille_flow", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()