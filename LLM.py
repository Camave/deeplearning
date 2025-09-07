import numpy as np
text = "hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello hello"

def build_vocab(text):
    # 1. caractères uniques
    vocab = sorted(list(set(text.lower())))
    
    # 2. dictionnaires
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    idx2char = {i: ch for i, ch in enumerate(vocab)}
    
    return vocab, char2idx, idx2char

def endode(text, char2idx):
    text = text.lower()
    encoded = []
    for c in text:
        encoded.append(char2idx[c])
    return encoded

def decode(encoded, idx2char):
    text = ""
    for i in encoded:
        text += idx2char[i]
    return text

def matrice(encoded, vocab_size):
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for i in range(len(encoded) - 1):
        c1 = encoded[i]
        c2 = encoded[i + 1]
        counts[c1, c2] += 1
    # Lissage
    counts += 1
    # Normalisation
    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs

def generate_text(start_char, length, char2idx, idx2char, probs):
    text = start_char
    current_idx = char2idx[start_char]
    
    for _ in range(length):
        # distribution pour le caractère courant
        p = probs[current_idx]
        
        # tirer un caractère aléatoire selon p
        next_idx = np.random.choice(len(p), p=p)
        
        # ajouter au texte
        text += idx2char[next_idx]
        
        # mettre à jour
        current_idx = next_idx
    
    return text

def perplexity(text, char2idx, probs):
    encoded = endode(text, char2idx)
    log_prob = 0.0
    for i in range(len(encoded) - 1):
        c1 = encoded[i]
        c2 = encoded[i + 1]
        log_prob += np.log(probs[c1, c2])
    n = len(encoded) - 1
    perplexity = np.exp(-log_prob / n)
    return perplexity

vocab, char2idx, idx2char = build_vocab(text)
encoded = endode(text, char2idx)
decoded = decode(encoded, idx2char)
one_hot_matrix = matrice(encoded, len(vocab))

print(char2idx)
print(encoded)
print(decoded)
print(one_hot_matrix)

print(generate_text("h", 4, char2idx, idx2char, one_hot_matrix))
print(perplexity(text, char2idx, one_hot_matrix))
