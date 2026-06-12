import numpy as np


# ── funções de ativação ──────────────────────────────────────────────────────

def sigmoid_binary(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_binary_deriv(fx):
    return fx * (1.0 - fx)


def sigmoid_bipolar(x):
    return 2.0 / (1.0 + np.exp(-x)) - 1.0

def sigmoid_bipolar_deriv(fx):
    return 0.5 * (1.0 - fx ** 2)


def tanh(x):
    return np.tanh(x)

def tanh_deriv(fx):
    return 1.0 - fx ** 2


ACTIVATIONS = {
    "binary":  (sigmoid_binary,  sigmoid_binary_deriv),
    "bipolar": (sigmoid_bipolar, sigmoid_bipolar_deriv),
    "tanh":    (tanh,            tanh_deriv),
}


# ── rede MLP ─────────────────────────────────────────────────────────────────

class MLP:
    """
    Rede MLP com 1 camada intermediária.
    Implementação manual do Backpropagation — sem frameworks.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs,
                 learning_rate=0.2, activation="bipolar", seed=None):

        if seed is not None:
            np.random.seed(seed)

        self.lr = learning_rate
        self.act, self.act_d = ACTIVATIONS[activation]

        # pesos e bias: inicialização randômica em [-0.5, 0.5]
        self.W1 = np.random.uniform(-0.5, 0.5, (n_inputs, n_hidden))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, n_hidden))

        self.W2 = np.random.uniform(-0.5, 0.5, (n_hidden, n_outputs))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, n_outputs))

    # ── forward pass ─────────────────────────────────────────────────────────

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.act(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.act(self.z2)
        return self.a2

    # ── backward pass ────────────────────────────────────────────────────────

    def backward(self, X, y):
        # erro na saída
        delta2 = (y - self.a2) * self.act_d(self.a2)

        # erro na camada intermediária
        delta1 = (delta2 @ self.W2.T) * self.act_d(self.a1)

        # atualização dos pesos
        self.W2 += self.lr * self.a1.T @ delta2
        self.b2 += self.lr * np.sum(delta2, axis=0, keepdims=True)

        self.W1 += self.lr * X.T @ delta1
        self.b1 += self.lr * np.sum(delta1, axis=0, keepdims=True)

    # ── treinamento ──────────────────────────────────────────────────────────

    def train(self, X, y, tolerance=0.001, max_epochs=10_000):
        history = []
        for epoch in range(1, max_epochs + 1):
            self.forward(X)
            self.backward(X, y)

            total_error = np.sum(np.abs(y - self.a2))
            history.append(total_error)

            if total_error < tolerance:
                return epoch, history

        return max_epochs, history

    # ── predição ─────────────────────────────────────────────────────────────

    def predict(self, X):
        return self.forward(X)
