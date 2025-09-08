import numpy as np
text = "bonjour je suis en train d’apprendre un RNN"

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

def sequence(text):
    seq_length = 5
    step = 1
    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_length, step):
        sequences.append(text[i: i + seq_length])
        next_chars.append(text[i + seq_length])
    return sequences, next_chars

X, Y = sequence(text)
vocab, char2idx, idx2char = build_vocab(text)

for i in range(3):
    print("X (indices) :", X[i], " -> ", decode(endode(X[i], char2idx), idx2char))
    print("Y (indice) :", Y[i], " -> ", idx2char[char2idx[Y[i]]])