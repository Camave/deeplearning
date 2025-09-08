import numpy as np
text = "La partie européenne de la France, appelée France métropolitaine, est située à l'extrémité occidentale de l'Europe et sa délimitation est restée inchangée depuis la fin de la Seconde Guerre mondiale et le traité de Paris en 1947. Elle est bordée par la mer du Nord au nord, la Manche au nord-nord-ouest, la mer Celtique à l'ouest-nord-ouest, le golfe de Gascogne à l'ouest et la mer Méditerranée au sud-est. Elle est frontalière de la Belgique au nord-nord-est, du Luxembourg au nord-est, de l'Allemagne à l'est-nord-est, de la Suisse à l'est, de l'Italie à l'est-sud-est, de Monaco au sud-est et de l'Espagne et Andorre au sud-sud-ouest."

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

def matrice3(encoded, vocab_size):
    counts = np.zeros((vocab_size, vocab_size, vocab_size), dtype=np.float32)
    for i in range(len(encoded) - 2):
        c1 = encoded[i]
        c2 = encoded[i + 1]
        c3 = encoded[i + 2]
        counts[c1, c2, c3] += 1
    # Lissage
    counts += 1
    # Normalisation
    probs = counts / counts.sum(axis=2, keepdims=True)
    return probs


def generate_text3(start_char, length, char2idx, idx2char, probs):
    text = start_char
    current_idx = (char2idx[text[0]], char2idx[text[1]])

    for _ in range(length):
        # distribution pour le caractère courant
        p = probs[current_idx[0], current_idx[1], :]
        
        # tirer un caractère aléatoire selon p
        next_idx = np.random.choice(len(p), p=p)
        
        # ajouter au texte
        text += idx2char[next_idx]
        
        # mettre à jour
        current_idx = (current_idx[1], next_idx)
    
    return text

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


def perplexity3(text, char2idx, probs):
    encoded = endode(text, char2idx)
    log_prob = 0.0
    for i in range(len(encoded) - 2):
        c1 = encoded[i]
        c2 = encoded[i + 1]
        c3 = encoded[i + 2]
        log_prob += np.log(probs[c1, c2, c3])
    n = len(encoded) - 2
    perplexity = np.exp(-log_prob / n)
    return perplexity


split_idx = int(len(text) * 0.8)
text_train = text[:split_idx]
text_test = text[split_idx:]


vocab, char2idx, idx2char = build_vocab(text)
encoded = endode(text_train, char2idx)

proba_train_3 = matrice3(encoded, len(vocab))
perplex_train_3 = perplexity3(text_train, char2idx, proba_train_3)
perplex_test_3 = perplexity3(text_test, char2idx, proba_train_3)
g3 = generate_text3("he", 50, char2idx, idx2char, proba_train_3)

proba_train_2 = matrice(encoded, len(vocab))
perplex_train_2 = perplexity(text_train, char2idx, proba_train_2)
perplex_test_2 = perplexity(text_test, char2idx, proba_train_2)
g2 = generate_text("h", 50, char2idx, idx2char, proba_train_2)


print("Train text 3:", perplex_train_3)
print("Test text 3:", perplex_test_3)
print(g3)

print("Train text 2:", perplex_train_2)
print("Test text 2:", perplex_test_2)
print(g2)

# vocab, char2idx, idx2char = build_vocab(text)
# encoded = endode(text, char2idx)
# decoded = decode(encoded, idx2char)
# proba = matrice3(encoded, len(vocab))
# text = generate_text("he", 5, char2idx, idx2char, proba)
# perplex = perplexity(text, char2idx, proba)

# print(char2idx)
# print(encoded)
# print(decoded)
# print(proba)
# print(text)
# print(perplex)

# # print(generate_text("h", 4, char2idx, idx2char, one_hot_matrix))
# # print(perplexity(text, char2idx, one_hot_matrix))

