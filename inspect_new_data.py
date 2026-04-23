import pandas as pd

print('=== NEW DATASET: train_dataset(big mart).csv ===')
df = pd.read_csv('big mart dataset/train_dataset(big mart).csv')
print(f'Shape: {df.shape}')
print('Columns:')
for c in df.columns:
    print(f'  - {c}  [{df[c].dtype}]')

print()
print('First 3 rows:')
print(df.head(3).to_string())

print()
print('=== NUMERIC COLUMN STATS ===')
print(df.describe().to_string())

print()
if 'Profit' in df.columns:
    print('Profit column found!')
    print(f'  Negative profit rows: {(df["Profit"] < 0).sum()}')
    print(f'  Zero profit rows    : {(df["Profit"] == 0).sum()}')
    print(f'  Positive profit rows: {(df["Profit"] > 0).sum()}')
    print(f'  Min : {df["Profit"].min():.2f}')
    print(f'  Max : {df["Profit"].max():.2f}')
    print(f'  Mean: {df["Profit"].mean():.2f}')
else:
    print('No Profit column found.')

print()
print('=== NULL VALUES ===')
nulls = df.isnull().sum()
print(nulls[nulls > 0].to_string() if nulls.any() else 'No nulls found.')
