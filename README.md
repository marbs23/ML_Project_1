# 🛩️ Mantenimiento Predictivo de Turbofanes NASA FD002

**Clasificación Binaria de Estado Crítico mediante Random Forest y Regresión Logística**

> Proyecto 1 — CS3061 Machine Learning · UTEC 2025  
> Autor: Martin Jesús Bonilla Sarmiento

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Kaggle Score](https://img.shields.io/badge/Kaggle%20F1-0.90-brightgreen?logo=kaggle)](https://www.kaggle.com/competitions/mantenimiento-predictivo-aeronautico)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Descripción del Problema

El dataset **NASA CMAPSS FD002** registra la vida completa de **260 turbofanes** bajo **seis condiciones operacionales** distintas (combinaciones de altitud y número de Mach). Cada motor comienza completamente sano en el ciclo 1 y opera hasta la falla total.

El objetivo es construir un **sistema de alerta temprana** que, dado el estado actual de los sensores de un motor, prediga si éste se encuentra en estado **Crítico** (≤ 30 ciclos para la falla) o **Sano** (> 30 ciclos restantes).

```
y = 1  →  Crítico: RUL ≤ 30 ciclos  (requiere mantenimiento urgente)
y = 0  →  Sano:    RUL > 30 ciclos  (puede seguir operando)
```

**Desafíos principales:**
- 6 condiciones operacionales que introducen correlación espuria en los sensores
- Desbalance de clases: ~85% Sano / ~15% Crítico
- 21 sensores sin etiquetas físicas interpretables
- La métrica de evaluación es **F1-score** (no accuracy)

---

## 📊 Resultados

| Modelo | F1 | Accuracy | Recall | Precision | Threshold |
|--------|-----|----------|--------|-----------|-----------|
| **RF-All** (sin deconfounding) | **0.8598** | **0.9554** | 87.3% | 84.7% | 0.661 |
| RF-Sel (con deconfounding) | 0.8562 | 0.9540 | 87.5% | 83.8% | 0.652 |
| LR-Sel (con deconfounding) | 0.8310 | 0.9459 | 84.9% | 81.3% | 0.714 |

🏆 **Score público en Kaggle: F1 = 0.90** (submission con RF-All, threshold = 0.661)

---

## 🗂️ Estructura del Repositorio

```
ML_Project_1/
│
├── NASA/
│   ├── NASA_problem.py          # Script principal — pipeline completo
│   ├── train_FD002.txt          # Dataset de entrenamiento (260 motores)
│   ├── test_FD002.txt           # Dataset de test (259 motores, historia parcial)
│   ├── RUL_FD002.txt            # RUL real del test (referencia)
│   ├── submission.csv           # Predicciones para Kaggle
│   └── outputs/
│       ├── variance.png         # Gráfico de varianza de sensores
│       ├── precision-recall.png # Curvas Precision-Recall
│       └── confussion-matrices.png  # Matrices de confusión
│
├── report/
│   └── ML_Proj_1.pdf            # Informe completo (IEEEtran)
│
└── README.md
```

---

## ⚙️ Pipeline de Preprocesamiento

El preprocesamiento se diseñó para evitar **data leakage** en todas sus etapas. Todos los parámetros se estiman exclusivamente sobre el conjunto de entrenamiento.

### 1. División por Motor (no por fila)

```python
# CORRECTO: todas las filas de un motor van al mismo split
train_engines, val_engines = train_test_split(
    all_engines, test_size=0.2, random_state=42
)
# 208 motores → train | 52 motores → validación
```

Si se dividiera por filas, el modelo vería ciclos tardíos del mismo motor en training y ciclos tempranos en validación — esto constituye leakage temporal.

### 2. Definición del Target

```python
train_df['RUL']   = train_df['max_cycle'] - train_df['cycle']
train_df['label'] = (train_df['RUL'] <= 30).astype(int)
```

### 3. Eliminación por Baja Varianza

Sensores con varianza < 0.1 son prácticamente constantes y no aportan información discriminativa.

```
Eliminados: sensor_16 (var = 2.2e-5), sensor_10 (var = 0.016)
```

### 4. Desconfoundización Operacional

Las 6 condiciones operacionales explican el 96–100% de la varianza de los sensores (R² ≈ 1.0), enmascarando la señal de degradación real.

**Solución:** para cada sensor, ajustar una regresión lineal sobre las condiciones operacionales y conservar el residuo:

```python
def deconfound(df_source, df_target, sensors_to_fix, op_cols):
    for sensor in sensors_to_fix:
        reg = LinearRegression()
        reg.fit(df_source[op_cols], df_source[sensor])
        residual = df_target[sensor] - reg.predict(df_target[op_cols])
        result[sensor] = residual
```

**Impacto medido:**

| Sensor | Correlación RAW con label | Correlación DECONFOUNDED |
|--------|--------------------------|--------------------------|
| sensor_15 | 0.03 | **0.53** |
| sensor_11 | 0.05 | **0.38** |
| sensor_4  | 0.04 | **0.32** |
| sensor_13 | 0.00 | **0.29** |

### 5. Selección por Correlación

Sobre los residuos desconfoundizados, se eliminan sensores con |ρ| < 0.03 con la etiqueta binaria. Sobreviven **11 sensores** de 19.

**Sensores finales (RF-Sel y LR-Sel):**

| Sensor | \|ρ\| | Sensor | \|ρ\| |
|--------|--------|--------|--------|
| sensor_15 | 0.5315 | sensor_17 | 0.1959 |
| sensor_11 | 0.3836 | sensor_3  | 0.1940 |
| sensor_4  | 0.3201 | sensor_9  | 0.1242 |
| sensor_13 | 0.2939 | sensor_21 | 0.0528 |
| sensor_14 | 0.2650 | sensor_20 | 0.0526 |
| sensor_2  | 0.0333 | | |

---

## 🤖 Modelos

### Random Forest (RF-All y RF-Sel)

```python
RandomForestClassifier(
    n_estimators=100,       # 100 árboles — estabilidad del ensamble
    max_depth=10,           # evita memorizar los 208 motores de training
    min_samples_leaf=20,    # cada hoja ≥ 20 ciclos — estabilidad
    class_weight='balanced', # compensa desbalance 85/15
    random_state=42
)
```

**¿Por qué RF?** Captura interacciones no lineales entre sensores, maneja las 6 condiciones operacionales de forma nativa (sin KMeans ni clustering previo), y el bagging lo hace resistente al overfitting.

### Logistic Regression (LR-Sel)

```python
Pipeline([
    ('scaler', StandardScaler()),     # Z-score: (x - μ) / σ
    ('clf', LogisticRegression(
        C=1.0,                        # regularización L2, λ = 1/C
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])
```

**¿Por qué LR?** Sirve como línea base interpretable. El `StandardScaler` es obligatorio porque sensor_9 opera en miles y sensor_15 en unidades — sin escalado el gradiente diverge. El `Pipeline` garantiza que el scaler no se fitea sobre datos de validación.

---

## 📈 Optimización del Umbral

En lugar del threshold por defecto (0.5), se maximiza F1 sobre la curva Precision-Recall:

```python
precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
f1_per_threshold = 2 * precisions * recalls / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_per_threshold[:-1])]
```

El threshold óptimo de RF-All es **0.661** — el modelo necesita ≥66.1% de confianza para declarar un motor Crítico, lo que reduce falsas alarmas manteniendo alto recall.

---

## 💰 Análisis de Costo-Beneficio

| Evento | Costo USD |
|--------|-----------|
| Falso Negativo (motor crítico no detectado → falla catastrófica) | $500,000 |
| Falso Positivo (mantenimiento innecesario) | $15,000 |
| Verdadero Positivo (mantenimiento correcto) | $15,000 |

**Asimetría 33:1** entre FN y FP.

| Modelo | FN | FP | VP | Costo FN | Costo Mant. | **Total** | **Ahorro vs sin modelo** |
|--------|----|----|-----|----------|-------------|-----------|--------------------------|
| RF-All | 205 | 254 | 1407 | $102.5M | $24.9M | $127.4M | **$678.6M** |
| RF-Sel | 201 | 273 | 1411 | $100.5M | $25.3M | $125.8M | **$680.2M** |
| LR-Sel | 243 | 314 | 1369 | $121.5M | $25.3M | $146.8M | $659.2M |
| Sin modelo | 1612 | 0 | 0 | $806.0M | $0 | $806.0M | — |

---

## 🚀 Reproducibilidad

### Instalación

```bash
git clone https://github.com/marbs23/ML_Project_1
cd ML_Project_1/NASA
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Ejecutar el pipeline completo

```bash
python NASA_problem.py
```

El script ejecuta en orden:
1. Carga de datos
2. Creación de RUL y label
3. Split por motor (seed = 42)
4. Eliminación por varianza
5. Desconfoundización
6. Selección por correlación
7. Entrenamiento de RF-All, RF-Sel y LR-Sel
8. Optimización de threshold vía curva Precision-Recall
9. Exportación de gráficos a `outputs/`
10. Generación de `submission.csv`

**Seed fijo:** `random_state=42` en `train_test_split`, `RandomForestClassifier` y `LogisticRegression`.

---

## 📁 Datos

El dataset NASA CMAPSS FD002 está disponible en:
- [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [Kaggle — Mantenimiento Predictivo Aeronáutico](https://www.kaggle.com/competitions/mantenimiento-predictivo-aeronautico/data)

**Formato de `train_FD002.txt`:** sin header, separado por espacios.
```
engine_id | cycle | op_setting_1 | op_setting_2 | op_setting_3 | sensor_1 ... sensor_21
```

---

## 📄 Informe

El informe completo en formato IEEEtran está disponible en [`report/ML_Proj_1.pdf`](report/ML_Proj_1.pdf).

Contiene: ingeniería de datos, metodología, resultados, análisis de costo-beneficio, bias-variance tradeoff y evidencia de submission en Kaggle (F1 público = **0.90**).

---

## ✍️ Autor

**Martin Jesús Bonilla Sarmiento**  
Facultad de Ingeniería — Universidad de Ingeniería y Tecnología (UTEC)  
Lima, Perú · 2025