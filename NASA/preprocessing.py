import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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
plt.show()

# Drop sensors with variance below threshold
VARIANCE_THRESHOLD = 0.1
low_variance_sensors = sensor_variance[sensor_variance < VARIANCE_THRESHOLD].index.tolist()
print(f"\nSensors to DROP (low variance): {low_variance_sensors}")
