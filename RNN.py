import numpy as np
text = "bonjour je suis en train d'apprendre un RNN"

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

def sequence(text, seq_length=5):
    step = 1
    sequences = []
    next_chars = []
    for i in range(0, len(text) - seq_length, step):
        sequences.append(text[i: i + seq_length])
        next_chars.append(text[i + seq_length])
    return sequences, next_chars

def matrice(vocab_size, seq_length=5, X=None, Y=None):
    counts_X = np.zeros((len(X), seq_length, vocab_size), dtype=np.float32)
    for i in range(len(X)):
        for j in range(len(X[i])):
            t = X[i][j]
            counts_X[i, j, t] += 1

    count_Y = np.zeros((len(Y), vocab_size), dtype=np.float32)
    for i in range(len(Y)):
        t = Y[i]
        count_Y[i, t] += 1

    return counts_X, count_Y

def init_rnn(vocab_size, hidden_size):
    parameters = {}
    # poids
    parameters["Wxh"] = np.random.randn(hidden_size, vocab_size) * 0.01  # input to hidden
    parameters["Whh"] = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden
    parameters["Why"] = np.random.randn(vocab_size, hidden_size) * 0.01  # hidden to output
    # biais
    parameters["bh"] = np.zeros((hidden_size, 1))  # hidden bias
    parameters["by"] = np.zeros((vocab_size, 1))  # output bias
    return parameters

def forward_rnn(inputs, h_prev, parameters):
    Wxh, Whh, Why = parameters["Wxh"], parameters["Whh"], parameters["Why"]
    bh, by = parameters["bh"], parameters["by"]
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(h_prev)
    for t in range(len(inputs)):
        xs[t] = inputs[t].reshape(-1, 1)  # one-hot vector
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t - 1]) + bh)  # hidden state
        ys[t] = np.dot(Why, hs[t]) + by  # unnormalized log probabilities for next chars
        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # probabilities for next chars
    return xs, hs, ys, ps

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # stabilité numérique
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def hidden_test(parameters, one_hot_X):
    

def forward(parameters,one_hot_X):
    h_t = {}
    y_t = {}
    h_t["h"+ str(0)] = np.zeros((parameters["Whh"].shape[0], 1))
    for t in range(one_hot_X.shape[1]):

        


seq_length = 6
hidden_size = 32
X, Y = sequence(text, seq_length)
vocab, char2idx, idx2char = build_vocab(text)

# Encoder X en indices
X_encoded = [endode(seq, char2idx) for seq in X]
Y_encoded = [char2idx[c.lower()] for c in Y]

One_hot_X, One_hot_Y = matrice(len(vocab), seq_length, X_encoded, Y_encoded)
params = init_rnn(len(vocab), hidden_size)

