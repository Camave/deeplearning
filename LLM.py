import numpy as np
text = "hello"

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

def matrice(encoded):
    one_hot = np.ones((len(encoded), len(encoded)), dtype=np.float32)
    for i, idx in enumerate(encoded):
        one_hot[i, idx] += 1.0
    return one_hot

vocab, char2idx, idx2char = build_vocab(text)
encoded = endode(text, char2idx)
decoded = decode(encoded, idx2char)
one_hot_matrix = matrice(encoded)

print(char2idx)
print(encoded)
print(decoded)
print(one_hot_matrix)
