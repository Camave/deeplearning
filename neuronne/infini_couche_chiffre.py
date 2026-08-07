import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def initialize_weights(dimension):
    parameters = {}
    c = len(dimension)
    
    for i in range(1, c):
        W= np.random.rand(dimension[i], dimension[i-1]) * np.sqrt(2/dimension[i-1])
        b = np.random.randn(dimension[i], 1)
        parameters['W' + str(i)] = W 
        parameters['b' + str(i)] = b

    return parameters


def Forward_propagation(X,parameters):
    activation = {}
    activation['A0'] = X
    c = len(parameters) // 2

    for i in range(1,c+1):        
        Z = parameters["W" + str(i)].dot(activation['A' + str(i-1)]) + parameters["b" + str(i)]
        A = 1 / (1 + np.exp(-Z))
        activation['A' + str(i)] = A

    return activation

def Back_propagation(y, activation, parameters):
    gradients = {}
    m = y.shape[1]
    c = len(parameters)//2
    dZ = activation['A' +str(c)] - y

    for i in reversed(range(1, c + 1)):
        dW = 1/m * dZ.dot(activation['A' + str(i-1)].T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.dot(parameters['W' + str(i)].T, dZ) * activation['A' + str(i-1)] * (1 - activation['A' + str(i-1)])
        gradients['dW' + str(i)] = dW
        gradients['db' + str(i)] = db

    return gradients

def ubdate(gradients, parameters, a):
    c = len(gradients) // 2
    for i in range(1, c + 1):
        parameters['W' + str(i)] -= a * gradients['dW' + str(i)]
        parameters['b' + str(i)] -= a * gradients['db' + str(i)] 

    return parameters

def L(Y, A):
    epsilon = 1e-15  # To avoid log(0)
    return -np.mean(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon))

def Predict(X, parameters):
    activation = Forward_propagation(X, parameters)
    A = activation["A" + str(len(parameters) // 2)]
    return A >= 0.5

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
dim = (128, 64)
params = neurol_network(X_train, y_train, dim, a=0.1, epochs=10000)

# Test
y_pred = Predict(X_test, params)
acc = accuracy_score(y_test.flatten(), y_pred.flatten())
print(f"Test Accuracy: {acc * 100:.2f}%")   

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