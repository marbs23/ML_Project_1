import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

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

