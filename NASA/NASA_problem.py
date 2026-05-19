import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os


os.makedirs("outputs", exist_ok=True)
# Load data
col_names = (
    ['engine_id', 'cycle'] +
    ['op_setting_1', 'op_setting_2', 'op_setting_3'] +
    [f'sensor_{i}' for i in range(1, 22)]
)

train_df = pd.read_csv('train_FD002.txt', sep=r'\s+', header=None, names=col_names)

# Create RUL and binary label
max_cycles = train_df.groupby('engine_id')['cycle'].max().reset_index()
max_cycles.columns = ['engine_id', 'max_cycle']
train_df = train_df.merge(max_cycles, on='engine_id')
train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
train_df['label'] = (train_df['RUL'] <= 30).astype(int)

# Split engines 80% train, 20% validation
all_engines = train_df['engine_id'].unique()

train_engines, val_engines = train_test_split(
    all_engines,
    test_size=0.2,
    random_state=42  # seed
)

df_train = train_df[train_df['engine_id'].isin(train_engines)].copy()
df_val   = train_df[train_df['engine_id'].isin(val_engines)].copy()

print(f"Training engines:   {len(train_engines)} ({len(df_train):,} rows)")
print(f"Validation engines: {len(val_engines)}  ({len(df_val):,} rows)")
print(f"\nTrain label distribution:")
print(df_train['label'].value_counts(normalize=True).round(3))
print(f"\nValidation label distribution:")
print(df_val['label'].value_counts(normalize=True).round(3))

# PREPROCESSING
# 1. Eliminate variables about variance
sensors = [f'sensor_{i}' for i in range(1, 22)]

sensor_variance = df_train[sensors].var().sort_values()
print("Sensor variance (sorted from lowest to highest):")
print(sensor_variance.round(6))

# Visualization about Variance
fig, ax = plt.subplots(figsize=(12, 5))
colors = ['tomato' if v < 0.1 else 'steelblue' for v in sensor_variance.values]
bars = ax.bar(sensor_variance.index, sensor_variance.values, color=colors)
ax.axhline(y=0.1, color='red', linestyle='--', linewidth=1.5, 
           label='Variance threshold = 0.1')
ax.set_xlabel('Sensor')
ax.set_ylabel('Variance')
ax.set_title('Sensor Variance — Red bars are candidates for removal')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"outputs/variance.png")
plt.close()

# Drop sensors with variance below threshold
VARIANCE_THRESHOLD = 0.1
low_variance_sensors = sensor_variance[sensor_variance < VARIANCE_THRESHOLD].index.tolist()
print(f"\nSensors to DROP (low variance): {low_variance_sensors}")

# 2. Decofound data

def deconfound(df_source, df_target, sensors_to_fix, op_cols):
    result = df_target.copy()
    
    op_train  = df_source[op_cols].values   # used to FIT the model
    op_target = df_target[op_cols].values   # used to PREDICT and subtract
    
    for sensor in sensors_to_fix:
        
        reg = LinearRegression()
        reg.fit(op_train, df_source[sensor].values)
        predicted_by_ops = reg.predict(op_target)
        result[sensor] = df_target[sensor].values - predicted_by_ops
    
    return result


op_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']

sensors_after_variance = [s for s in sensors if s not in low_variance_sensors]

df_tr_deconf = deconfound(df_train, df_train, sensors_after_variance, op_cols)
df_v_deconf  = deconfound(df_train, df_val,   sensors_after_variance, op_cols)

print(f"\nDeconfounding applied to {len(sensors_after_variance)} sensors")
print(f"Shapes — train: {df_tr_deconf.shape}, val: {df_v_deconf.shape}")

# Compute correlations with deconf
label_corr_deconf = (
    df_tr_deconf[sensors_after_variance]
    .corrwith(df_tr_deconf['label'])
    .abs()
    .sort_values(ascending=False)
)

print("\nCorrelation with Critical label AFTER deconfounding:")
print(label_corr_deconf.round(4))

LOW_CORR_THRESHOLD = 0.05
low_corr_sensors = label_corr_deconf[
    (label_corr_deconf < LOW_CORR_THRESHOLD) | (label_corr_deconf.isna())
].index.tolist()
survive_sensors = label_corr_deconf[label_corr_deconf >= LOW_CORR_THRESHOLD].index.tolist()

# Final set of sensors to use
final_sensors = [s for s in survive_sensors]
print(f"\n{'='*50}")
print(f"FINAL SENSOR SELECTION SUMMARY")
print(f"{'='*50}")
print(f"Started with:              21 sensors")
print(f"Dropped (low variance):    {len(low_variance_sensors)} sensors → {low_variance_sensors}")
print(f"Dropped (low label corr):  {len(low_corr_sensors)} sensors → {low_corr_sensors}")
print(f"Final survive sensors:     {len(final_sensors)} sensors → {final_sensors}")


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, classification_report,
                              confusion_matrix, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

features_A = op_cols + sensors_after_variance
features_B = op_cols + final_sensors

def train_and_evaluate(model, X_tr, y_tr, X_v, y_v, model_name):
    # Train model LR or RF
    model.fit(X_tr, y_tr)
    
    # Probabilities of validation
    proba = model.predict_proba(X_v)[:, 1]
    
    # Find possible thresholds / Trade-off curve
    precisions, recalls, thresholds = precision_recall_curve(y_v, proba)
    
    # F1 for each threshold
    f1_per_threshold = (
        2 * precisions * recalls / (precisions + recalls + 1e-8)
    )
    
    # BEST F1 and Threshold
    best_idx       = np.argmax(f1_per_threshold[:-1])
    best_threshold = thresholds[best_idx]
    best_f1        = f1_per_threshold[best_idx]
    
    # Apply threshold
    preds = (proba >= best_threshold).astype(int)
    
    print(f"\n{'─'*45}")
    print(f"  {model_name}")
    print(f"{'─'*45}")
    print(f"  Threshold óptimo : {best_threshold:.3f}")
    print(f"  F1 Score         : {best_f1:.4f}")
    print(f"  Accuracy         : {(preds == y_v).mean():.4f}")
    
    return {
        'model'     : model,
        'name'      : model_name,
        'threshold' : best_threshold,
        'f1'        : best_f1,
        'preds'     : preds,
        'proba'     : proba,
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls'   : recalls,
        'f1_curve'  : f1_per_threshold
    }

# Model 1: Random Forest all features
rf_all = RandomForestClassifier(
    n_estimators=100,      # 100 árboles — balance entre velocidad y estabilidad
    class_weight='balanced', # penaliza más los errores en la clase minoritaria (Crítico)
    max_depth=10,          # profundidad máxima de cada árbol
                           # sin límite → árboles memorizan training (overfitting)
    min_samples_leaf=20,   # cada hoja necesita mínimo 20 ejemplos
                           # evita que el árbol divida sobre un solo ciclo
    random_state=42,
    n_jobs=-1
)

result_rf_all = train_and_evaluate(
    model      = rf_all,
    X_tr       = df_train[features_A],
    y_tr       = df_train['label'],
    X_v        = df_val[features_A],
    y_v        = df_val['label'],
    model_name = 'RF — All features (without deconfounding)'
)

# Model 2: Random Forest with survive features
rf_selected = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    max_depth=10,
    min_samples_leaf=20,
    random_state=42,
    n_jobs=-1
)

result_rf_sel = train_and_evaluate(
    model      = rf_selected,
    X_tr       = df_tr_deconf[features_B],
    y_tr       = df_train['label'],
    X_v        = df_v_deconf[features_B],
    y_v        = df_val['label'],
    model_name = 'RF — Survive features (with deconfounding)'
)

# ── Modelo 3: Logistic Regression con features SELECCIONADAS ─────────────────
# LR asume que la frontera de decisión es lineal en el espacio de features
# Necesita escalado porque es sensible a la magnitud de los valores
# Pipeline garantiza que el scaler se fitea solo en train (no hay leakage)

lr_selected = Pipeline([
    ('scaler', StandardScaler()),
    # StandardScaler: x_scaled = (x - mean) / std
    # Necesario para LR porque los sensores tienen escalas muy diferentes
    # sensor_9 puede tener valores en miles, sensor_15 en unidades
    # Sin escalar, el gradiente estaría dominado por los sensores grandes
    
    ('clf', LogisticRegression(
        class_weight='balanced',
        C=1.0,         # C = 1/λ donde λ es la fuerza de regularización L2
                       # C grande = menos regularización (más flexible)
                       # C pequeño = más regularización (más conservador)
        max_iter=1000,
        random_state=42
    ))
])

result_lr_sel = train_and_evaluate(
    model      = lr_selected,
    X_tr       = df_tr_deconf[features_B],
    y_tr       = df_train['label'],
    X_v        = df_v_deconf[features_B],
    y_v        = df_val['label'],
    model_name = 'LR — Features seleccionadas (con deconfounding)'
)

# Confussion matrices Visualization
results = [result_rf_all, result_rf_sel, result_lr_sel]
y_v = df_val['label']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, result in zip(axes, results):
    cm = confusion_matrix(y_v, result['preds'])
    
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    labels = np.array([
        [f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" 
         for j in range(2)] 
        for i in range(2)
    ])
    
    sns.heatmap(
        cm_pct,
        annot=labels,
        fmt='',
        cmap='Blues',
        xticklabels=['Healthy', 'Critical'],
        yticklabels=['Healthy', 'Critical'],
        ax=ax,
        vmin=0, vmax=100,
        linewidths=1
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    f1 = f1_score(y_v, result['preds'])
    ax.set_title(f"{result['name']}\nF1 = {f1:.4f} | Threshold = {result['threshold']:.3f}")

plt.suptitle('Comparación de Modelos — Validation Set', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(f"outputs/confussion-matrices.png")
plt.close()

# Precision-Recall Visualization
fig, ax = plt.subplots(figsize=(9, 6))

colors = ['steelblue', 'tomato', 'green']
for result, color in zip(results, colors):
    ax.plot(
        result['recalls'][:-1],
        result['precisions'][:-1],
        color=color,
        linewidth=2,
        label=f"{result['name']} (F1={result['f1']:.3f})"
    )
    # Marcar el punto óptimo en cada curva
    best_idx = np.argmax(result['f1_curve'][:-1])
    ax.scatter(
        result['recalls'][best_idx],
        result['precisions'][best_idx],
        color=color, s=100, zorder=5,
        marker='*'
    )


baseline = df_val['label'].mean()
ax.axhline(y=baseline, color='gray', linestyle='--',
           alpha=0.7, label=f'Baseline (random) = {baseline:.2f}')

ax.set_xlabel('Recall (fracción de motores críticos detectados)', fontsize=11)
ax.set_ylabel('Precision (confiabilidad de las alarmas)', fontsize=11)
ax.set_title('Curvas Precision-Recall\n'
             '★ = punto óptimo de F1 para cada modelo', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(f"outputs/precision-recall.png")
plt.close()


# SUMMARY FINAL TABLE

print("\n" + "="*70)
print("RESUMEN COMPARATIVO DE MODELOS")
print("="*70)
print(f"{'Modelo':<45} {'F1':>6} {'Threshold':>10} {'Features':>8}")
print("-"*70)

for result in results:
    f1 = f1_score(y_v, result['preds'])
    n_features = (
        len(features_A) if 'Todas' in result['name'] 
        else len(features_B)
    )
    print(f"{result['name']:<45} {f1:>6.4f} {result['threshold']:>10.3f} {n_features:>8}")

print("="*70)

# Best model
best_result = max(results, key=lambda r: f1_score(y_v, r['preds']))
print(f"\n→ Mejor modelo: {best_result['name']}")
print(f"  F1 = {f1_score(y_v, best_result['preds']):.4f}")
print(f"  Threshold = {best_result['threshold']:.3f}")



#dasfdsafdas

test_df = pd.read_csv('test_FD002.txt', sep=r'\s+', header=None, names=col_names)

print(f"Test data shape: {test_df.shape}")
print(f"Número de motores únicos en test: {test_df['engine_id'].nunique()}")
last_rows = (
    test_df
    .sort_values(['engine_id', 'cycle'])
    .groupby('engine_id')
    .last()                    # toma la fila con el cycle más alto
    .reset_index()             # convierte engine_id de índice a columna normal
)

print(f"\nDespués de tomar última fila por motor: {last_rows.shape}")
print(f"Primeros 5 motores con sus cycles finales:")
print(last_rows[['engine_id', 'cycle']].head())

print(f"\nMejor modelo: {best_result['name']}")
print(f"Threshold óptimo: {best_result['threshold']:.3f}")
print(f"Features usadas: {len(features_A)}")

X_test = last_rows[features_A]

test_proba = best_result['model'].predict_proba(X_test)[:, 1]

print(f"\nDistribución de probabilidades en test:")
print(f"  Min:    {test_proba.min():.3f}")
print(f"  Max:    {test_proba.max():.3f}")
print(f"  Mean:   {test_proba.mean():.3f}")
print(f"  Median: {np.median(test_proba):.3f}")

optimal_threshold = best_result['threshold']
test_preds = (test_proba >= optimal_threshold).astype(int)

print(f"\nThreshold aplicado: {optimal_threshold:.3f}")
print(f"Predicciones en test:")
print(f"  Sano (0):    {(test_preds == 0).sum()} motores")
print(f"  Crítico (1): {(test_preds == 1).sum()} motores")
print(f"  Proporción crítica: {test_preds.mean():.1%}")

submission = pd.DataFrame({
    'engine_id': last_rows['engine_id'].values,
    'Critico':   test_preds
})

submission = submission.sort_values('engine_id').reset_index(drop=True)

submission.to_csv('submission.csv', index=False)