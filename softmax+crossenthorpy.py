import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def initialize_weights(dimension):
    parameters = {}
    c = len(dimension)
    
    for i in range(1, c):
        W= np.random.rand(dimension[i], dimension[i-1]) * np.sqrt(2/dimension[i-1])
        b = np.random.randn(dimension[i], 1)
        parameters['W' + str(i)] = W 
        parameters['b' + str(i)] = b

    return parameters


def Forward_propagation(X, parameters):
    activation = {}
    activation['A0'] = X
    c = len(parameters) // 2

    for i in range(1, c):
        Z = parameters["W" + str(i)].dot(activation['A' + str(i-1)]) + parameters["b" + str(i)]
        A = 1 / (1 + np.exp(-Z))   # sigmoïde pour couches cachées
        activation['A' + str(i)] = A

    # dernière couche → softmax
    ZL = parameters["W" + str(c)].dot(activation['A' + str(c-1)]) + parameters["b" + str(c)]
    AL = softmax(ZL)
    activation['A' + str(c)] = AL

    return activation


def Back_propagation(y, activation, parameters):
    gradients = {}
    m = y.shape[1]
    c = len(activation) - 1  # nombre de couches

    # dernière couche (softmax)
    dZ = activation['A' + str(c)] - y  

    for i in reversed(range(1, c+1)):
        dW = 1/m * dZ.dot(activation['A' + str(i-1)].T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        gradients['dW' + str(i)] = dW
        gradients['db' + str(i)] = db

        if i > 1:  # pas pour la dernière couche
            dZ = np.dot(parameters['W' + str(i)].T, dZ) * activation['A' + str(i-1)] * (1 - activation['A' + str(i-1)])

    return gradients

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # stabilité numérique
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def cross_entropy(Y, A):
    m = Y.shape[1]
    epsilon = 1e-15
    return -np.sum(Y * np.log(A + epsilon)) / m

def ubdate(gradients, parameters, a):
    c = len(gradients) // 2
    for i in range(1, c + 1):
        parameters['W' + str(i)] -= a * gradients['dW' + str(i)]
        parameters['b' + str(i)] -= a * gradients['db' + str(i)] 

    return parameters

def L(Y, A):
    return cross_entropy(Y, A)

def Predict(X, parameters):
    activation = Forward_propagation(X, parameters)
    A = activation["A" + str(len(parameters) // 2)]
    return np.argmax(A, axis=0)

def neurol_network(X_train, y_train,dim = (32,32,32), a = 0.1 , epochs=5000):
    np.random.seed(0)
    dimension = list(dim)
    dimension.insert(0, X_train.shape[0])
    dimension.append(y_train.shape[0])
    parameters = initialize_weights(dimension)

    train_loss = []
    train_acc = []

    for i in tqdm(range(epochs)):

        activation = Forward_propagation(X_train,parameters) 
        gradients = Back_propagation(y_train, activation, parameters)
        parameters = ubdate(gradients, parameters, a)

        if i % 10 ==0 :
            # Train
            c = len(parameters) // 2
            train_loss.append(L(y_train, activation["A" + str(c)]))
            y_pred = Predict(X_train, parameters)
            # Correction : compare les classes (entiers) et non les one-hot
            y_true = np.argmax(y_train, axis=0)
            train_acc.append(accuracy_score(y_true, y_pred))

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
y = digits.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(y)
# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=True
)
# Adapter au format réseau
X_train, X_test = X_train.T, X_test.T
y_train, y_test = y_train.T, y_test.T  # Correction ici

# Entraînement
dim = (64,32)
params = neurol_network(X_train, y_train, dim, a=0.1, epochs=12000)

# Test
y_pred = Predict(X_test, params)
y_true_test = np.argmax(y_test, axis=0)  # Correction ici
acc = accuracy_score(y_true_test, y_pred)
print(f"Test Accuracy: {acc * 100:.2f}%")


# ---- TEST SUR LA PREMIÈRE IMAGE ----
for i in range(0,20):
    image = X_test[:, i].reshape(-1, 1)  # 64x1
    true_label = y_true_test[i]  # étiquette réelle

    # Affichage
    plt.imshow(image.reshape(8, 8), cmap='gray')
    plt.title(f"Label réel : {true_label}")
    plt.axis('off')
    plt.show()

    # Prédiction
    pred = Predict(image, params)
    print("Prédiction du réseau :", int(pred), )