import numpy as np
import csv
import matplotlib.pyplot as plt
import os

# DATA
def load_csv(ruta):
    data = []
    with open(ruta) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            row_to = [float(row[head]) for head in headers]
            data.append(row_to)
    return data, headers

# Normalizacion Z-Score
def z_score_normalization(X):
    means = np.mean(X, axis=0)
    sds = np.std(X, axis=0)
    sds_new = np.where(sds == 0, 1, sds)
    X_normal = (X - means) / sds_new
    return X_normal, means, sds

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predecir(X, W):
    return sigmoid(X @ W)

def calcular_bce(X, y, W):
    p = predecir(X, W)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    bce = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return bce

def gradiente_bce(X, y, W):
    N = len(y)
    p = predecir(X, W)
    grad_W = (1 / N)*(X.T @ (p - y))
    return grad_W

# GRADIENT DESCENT (GD)
def gradient_descent(X, y, lr=0.01, epochs=1000):
    W = np.zeros((X.shape[1], 1))

    history = []

    for i in range(epochs):
        bce = calcular_bce(X, y, W)
        if (i > 0 and abs(bce-history[-1][2])< 1e-10):
            history.append((i, W.copy(), bce))
            break
        history.append((i, W.copy(), bce))
        grad_W = gradiente_bce(X, y, W)
        W = W - lr * grad_W

    return W.copy(), history

def calculate_accuracy(X, y, W, threshold=0.5):
    p = predecir(X, W)
    y_pred = (p >= threshold).astype(int)
    
    TP = np.sum((y_pred == 1) & (y == 1))
    TN = np.sum((y_pred == 0) & (y == 0))
    FP = np.sum((y_pred == 1) & (y == 0))
    FN = np.sum((y_pred == 0) & (y == 1))
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy, TP, TN, FP, FN

def graficar(data, W, W_normal,  label_x, label_y, case, mode):
    dirname = f"outputs/{case}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    x_vals = [d[0] for d in data]
    y_vals = [d[1] for d in data]
    plt.figure(figsize=(10, 5))
    plt.scatter(x_vals, y_vals, color='blue', label='Data')

    x_line = np.linspace(min(x_vals), max(x_vals), 300)
    z_line = np.log(x_line)

    y_line_gd = W[0][0] + W[1][0] * z_line
    plt.plot(x_line, y_line_gd, color='red', linewidth=3, 
             label=f'GD: y={W[1][0]:.2f}x + {W[0][0]:.2f}')
    
    y_line_norm = W_normal[0][0] + W_normal[1][0] * z_line
    plt.plot(x_line, y_line_norm, color='green', linestyle='--', linewidth=2,
             label=f'Normal Ec.: y={W_normal[1][0]:.2f}x + {W_normal[0][0]:.2f}')
    
    plt.xscale('log')
    plt.title(f'Modelo Logarítmico {mode} — {case}')
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dirname}/{label_x}_{mode}.png")
    print(f"Save: {label_x} with {mode}.png")
    plt.close()

def print_hist(hist):
    print()
    for i in range(0, len(hist), 200):
        epoch, pesos, mse = hist[i]
        pesos_str = " | ".join([f"w{j}: {pesos[j][0]:.4f}" for j in range(len(pesos))])
        print(f"Iter {epoch:4d} | BCE: {mse:.8f} | {pesos_str}")
    
    if len(hist) % 200 != 1:
        epoch, pesos, mse = hist[-1]
        pesos_str = " | ".join([f"w{j}: {pesos[j][0]:.4f}" for j in range(len(pesos))])
        print(f"Iter {epoch:4d} | BCE: {mse:.8f} | {pesos_str} (Final)")

# MAIN
if __name__ == "__main__":
    # Preprocessing
    data, headers = load_csv('dataset.csv')
    data_array = np.array(data)

    # Normalization Z-score
    data_normal, means, sds = z_score_normalization(data_array[:,:-1])

    # X without normalization
    X = (np.hstack((np.ones((len(data),1)),data_array[:,:-1])))
    print(f"X (no normalization):\n{X}")

    # X with normalization Z-score
    X_normal = (np.hstack((np.ones((len(data),1)),data_normal)))
    print(f"\nX (normalization z-score):\n{X_normal}")

    # Y
    y = data_array[:, -1].reshape(-1, 1)
    y_bin = (y >200000).astype(int)
    print(f"\nY:\n{y_bin}")

    # Gradient Descent (Vector W)
    W_grad_desc, history = gradient_descent(X_normal, y_bin, 0.01, 2000)
    print(f"\nW (gradient descent):\n{W_grad_desc}")

    # Accuracy
    acc, _, _, _, _ = calculate_accuracy(X_normal, y_bin, W_grad_desc)
    print(f"Accuracy of model: {acc}")

    # History (MSE)
    print_hist(history)

    # New prediction
    x_new = np.array([175, 4, 8, 6], dtype=float)
    x_norm = (x_new - means) / sds
    x_norm = np.concatenate([[1], x_norm]).reshape(1, -1)
    z_new = x_norm @ W_grad_desc
    prob_new = sigmoid(z_new)
    print(f"\nProbability from new value:\n{prob_new}")