import random
import csv
import matplotlib.pyplot as plt
import os
import math

# DATA
def cargar_csv(ruta):
    data = []
    with open(ruta) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            row_to = [float(row[head]) for head in headers]
            data.append(row_to)
    return data, headers

def predecir(X, w, b):
    result = []
    for row in X:
        sum_row = 0
        for j in range(len(w)):
            sum_row += row[j] * w[j] 
        result.append(sum_row + b)
    return result

def calcular_mse(data, W, b):
    y = [row[-1] for row in data]
    y_pred = predecir(data, W, b)
    return sum((r-p)**2 for r, p in zip(y,y_pred))/len(data)

def calcular_mae(data, W, b):
    y = [row[-1] for row in data]
    y_pred = predecir(data, W, b)
    return sum(abs(r-p) for r, p in zip(y, y_pred))/len(data)

def gradiente_mse(data, w, b):
    n = len(data)
    grad_w = [0.0] * len(w)
    grad_b = 0.0
    y_pred = predecir(data, w, b)
    for i in range(n):
        error = data[i][-1] - y_pred[i]
        for j in range(len(w)):
            grad_w[j] += -2 * data[i][j] * error 
        grad_b += -2 * error
    grad_res_w = [gw/n for gw in grad_w]
    return grad_res_w, grad_b / n

def gradiente_una_muestra(x, y, w, b):
    y_pred = sum(x_i * w_i for x_i,w_i in zip(x, w))+b
    error = y - y_pred
    grad_w = [-2 * xj * (error) for xj in x]
    grad_b = -2 * error
    return grad_w, grad_b

def solucion_analitica(data):
    n = len(data)
    sx  = sum(x for x, y in data)
    sy  = sum(y for x, y in data)
    sxy = sum(x*y for x, y in data)
    sx2 = sum(x**2 for x, y in data)

    m = (n * sxy - sx * sy) / (n * sx2 - sx**2)
    b = (sy - m * sx) / n
    return m, b

# GRADIENT DESCENT (GD)
def gradient_descent(data, lr=0.01, iteraciones=1000, w_init = None, b_init=0.0):
    w = list(w_init) if w_init else [0.0] * (len(data[0]) - 1)
    b = b_init
    historial = []
    for i in range(iteraciones):
        mse = calcular_mse(data, w, b)
        historial.append((i, w[:], b, mse))
        grad_w, grad_b = gradiente_mse(data, w, b)
        for j in range(len(grad_w)):
            w[j] = w[j] - lr * grad_w[j]
        b = b - lr * grad_b
    return w, b, historial

def calcular_estadisticas(data):
    n = len(data)
    num_columnas = len(data[0])
    medias = []
    desviaciones = []
    
    for j in range(num_columnas - 1):
        columna = [row[j] for row in data]
        media = sum(columna) / n
        
        varianza = sum((x - media)**2 for x in columna) / n
        desviacion = math.sqrt(varianza)
        
        medias.append(media)
        desviaciones.append(desviacion)
        
    return medias, desviaciones

# Normalizacion Z-Score
def normalizar_z_score(data, medias, desviaciones):
    data_normalizada = []
    for row in data:
        nueva_fila = []
        for j in range(len(row) - 1):
            if desviaciones[j] > 0:
                z = (row[j] - medias[j]) / desviaciones[j]
            else:
                z = 0.0
            nueva_fila.append(z)
        
        nueva_fila.append(row[-1])
        data_normalizada.append(nueva_fila)
        
    return data_normalizada

def graficar(hist, case, mode):
    dirname = f"outputs/{case}"
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    mse_data = [d[3] for d in hist]
    plt.figure(figsize=(10, 6))
    plt.plot(mse_data, color='red', linewidth=1, label=f"MSE {mode} data")
    plt.title("Convergence Graph "+ mode)
    plt.xlabel("Iterations")
    plt.ylabel("Error MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{dirname}/{case}_convergence_{mode}.png")
    print(f"Save: Convergence graph with {mode}.png")
    plt.close()

# MAIN
if __name__ == "__main__":

    print("LINEAR REGRESSION WITH GRADIENT DESCENT (MULTIVARIABLE)")
    filename = "caso3_energia.csv"
    data_raw, headers = cargar_csv(filename)

    medias, desviaciones = calcular_estadisticas(data_raw)
    
    data = normalizar_z_score(data_raw, medias, desviaciones)

    # GD weights, bias and historial
    w_GD, b_GD, hist_GD = gradient_descent(data, lr=0.05, iteraciones=2000)
    print("Final weights with GD:")
    [print(f"{wi} ") for wi in w_GD]
    print(f"Bias: {b_GD}")
    # Graphs about convergence
    #graficar(hist_GD, filename, "GD")
    #graficar(hist_SGD, filename, "SGD")