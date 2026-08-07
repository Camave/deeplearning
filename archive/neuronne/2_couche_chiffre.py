import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def initialize_weights(n0, n1, n2):
    W1 = np.random.rand(n1,n0) * np.sqrt(2/n0)
    b1 = np.random.randn(n1,1)
    W2 = np.random.rand(n2,n1) * np.sqrt(2/n0)
    b2 = np.random.randn(n2,1)

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parameters

def Forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = W1.dot(X) + b1
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = W2.dot(A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))

    activation = {
        'A1': A1,
        'A2': A2
    }
    return activation

def Back_propagation(X, y, activation, parameters):
    W2 = parameters["W2"]
    A1 = activation["A1"]
    A2 = activation["A2"]

    m = y.shape[1]

    dZ2 = A2 - y
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }
    return gradients

def ubdate(gradients, parameters, a):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = gradients["dW1"]
    db1 = gradients["db1"]
    dW2 = gradients["dW2"]
    db2 = gradients["db2"]

    W1 -= a * dW1
    b1 -= a * db1
    W2 -= a * dW2
    b2 -= a * db2

    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }
    return parameters

def L(Y, A):
    epsilon = 1e-15  # To avoid log(0)
    return -np.mean(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

def Predict(X, parameters):
    activation = Forward_propagation(X, parameters)
    A2 = activation["A2"]
    return A2 >= 0.5

def neurol_network(X_train, y_train, n1, a, epochs):
    n0 = X_train.shape[0]
    n2 = y_train.shape[0] 
    parameters = initialize_weights(n0, n1, n2)

    train_loss = []
    train_acc = []

    for i in tqdm(range(epochs)):

        activation = Forward_propagation(X_train,parameters) 
        gradients = Back_propagation(X_train, y_train, activation, parameters)
        parameters = ubdate(gradients, parameters, a)

        if i % 10 ==0 :
            # Train
            train_loss.append(L(y_train, activation["A2"]))
            y_pred = Predict(X_train, parameters)
            train_acc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))

    plt.figure(figsize=(14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.legend()
    plt.title("Train Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.legend()
    plt.title("Train Accuracy")

    plt.tight_layout()
    plt.show()

    return parameters


digits = load_digits()

X = digits.data / 16.0
y = (digits.target == 3).astype(int)

# Indices des classes
idx_pos = np.where(y == 1)[0]  # tous les 3
idx_neg = np.where(y == 0)[0]  # tous les autres

# Échantillonnage aléatoire de négatifs pour équilibrer
np.random.seed(42)
idx_neg_sampled = np.random.choice(idx_neg, size=len(idx_pos), replace=False)

# Fusionner indices équilibrés
idx_balanced = np.concatenate([idx_pos, idx_neg_sampled])
np.random.shuffle(idx_balanced)

# Dataset équilibré
X_balanced = X[idx_balanced]
y_balanced = y[idx_balanced]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, shuffle=True
)
print(X_train.shape)
print(y_train.shape)
# Adapter au format réseau
X_train, X_test = X_train.T, X_test.T
y_train, y_test = y_train.reshape(1, -1), y_test.reshape(1, -1)

# Entraînement
params = neurol_network(X_train, y_train, n1=32, a=0.1, epochs=5000)

# Test
y_pred = Predict(X_test, params)
acc = accuracy_score(y_test.flatten(), y_pred.flatten())
print("Accuracy sur test :", acc)

# ---- TEST SUR LA PREMIÈRE IMAGE ----
for i in range(0,20):
    image = X_test[:, i].reshape(-1, 1)  # 64x1
    true_label = y_test[0, i]

    # Affichage
    plt.imshow(image.reshape(8, 8), cmap='gray')
    plt.title(f"Label réel : {true_label}")
    plt.axis('off')
    plt.show()

    # Prédiction
    pred = Predict(image, params)[0, 0]
    print("Prédiction du réseau :", int(pred), "(1 = c'est un 3, 0 = pas un 3)")