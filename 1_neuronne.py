import numpy as np 
import matplotlib.pyplot as plt


def Z(X,W,b):
    """
    Computes the linear combination of inputs and weights plus bias.

    Parameters:
    X (numpy.ndarray): Input data of shape (n_samples, n_features).
    W (numpy.ndarray): Weights of shape (n_features,).
    b (float): Bias term.

    Returns:
    numpy.ndarray: Resulting linear combination of shape (n_samples,).
    """
    return np.dot(X, W) + b

def A(Z):
    """
    Computes the sigmoid activation function.

    Parameters:
    z (numpy.ndarray): Input data of any shape.

    Returns:
    numpy.ndarray: Sigmoid activation of the input data, same shape as input.
    """
    return 1 / (1 + np.exp(-Z))

def L(Y, A):
    """
    Computes the binary cross-entropy loss.

    Parameters:
    Y (numpy.ndarray): True labels of shape (n_samples,).
    Y_hat (numpy.ndarray): Predicted probabilities of shape (n_samples,).

    Returns:
    float: Binary cross-entropy loss.
    """
    return -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))

def update_weights(X, Y, A, W, a, m):
    """
    Computes the gradient of the loss with respect to the weights.

    Parameters:
    X (numpy.ndarray): Input data of shape (n_samples, n_features).
    Y (numpy.ndarray): True labels of shape (n_samples,).
    A (numpy.ndarray): Predicted probabilities of shape (n_samples,).
    W (numpy.ndarray): Weights of shape (n_features,).
    a (float): Learning rate.
    m (int): Number of samples.

    Returns:
    numpy.ndarray: Updated weights after applying gradient descent.
    """
    dW = np.dot(X.T, (A - Y)) / m
    return W - a * dW

def update_bias(Y, A, b, a, m):
    """
    Computes the gradient of the loss with respect to the bias.

    Parameters:
    Y (numpy.ndarray): True labels of shape (n_samples,).
    A (numpy.ndarray): Predicted probabilities of shape (n_samples,).
    b (float): Bias term.
    a (float): Learning rate.
    m (int): Number of samples.

    Returns:
    float: Updated bias after applying gradient descent.
    """
    db = np.mean(A - Y)
    return b - a * db


X = np.array([[1.2, 4.3], 
              [2.3, 1.9], 
              [0.9, 1.2]])
weights = np.array([0.1, 0.2])
bias = 0.3
Y = np.array([0, 1, 1])

Z_val = Z(X, weights, bias)
A_val = A(Z_val)
loss = L(Y, A_val)

print("Z:", Z_val)
print("A:", A_val)
print("Loss:", loss)

a = 0.01
m = X.shape[0]
# Gradient descent loop
for i in range(1000):
    Z_val = Z(X, weights, bias)
    A_val = A(Z_val)
    loss = L(Y, A_val)

    weights = update_weights(X, Y, A_val, weights, a, m)
    bias = update_bias(Y, A_val, bias, a, m)

    print(f"Iteration {i+1}: Loss = {loss}, Weights = {weights}, Bias = {bias}")

# Affichage de la frontière de décision après l'entraînement
plt.figure()
# Afficher les points avec leur label
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap='bwr', edgecolors='k', label='Données')

# Calcul de la droite de séparation : w1*x1 + w2*x2 + b = 0  <=>  x2 = -(w1*x1 + b)/w2
x1_vals = np.linspace(X[:,0].min()-0.5, X[:,0].max()+0.5, 100)
x2_vals = -(weights[0]*x1_vals + bias)/weights[1]
plt.plot(x1_vals, x2_vals, 'g-', label='Frontière de décision')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Frontière de décision du modèle')
plt.legend()
plt.grid()
plt.show()


