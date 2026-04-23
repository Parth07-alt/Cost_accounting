import pandas as pd
import numpy as np

df = pd.read_csv('output/enriched_bigmart.csv')
loss = df[df['Profit'] < 0]
prof = df[df['Profit'] >= 0]

print('=== WHY IS LOSS PREDICTION IMPOSSIBLE? ===')
print()
print(f'Loss rows: {len(loss)} | Profit rows: {len(prof)}')

features = ['Item_MRP', 'Item_Outlet_Sales', 'Item_Visibility', 'Item_Weight', 'Outlet_Age']
print()
print('Feature comparison - Loss rows vs Profitable rows:')
for f in features:
    lm = loss[f].mean()
    pm = prof[f].mean()
    print(f'  {f}: Loss={lm:.2f}  Profit={pm:.2f}  diff={lm-pm:+.2f}')

df['eff_cost_rate'] = df['Total_Cost'] / df['Item_Outlet_Sales']
print()
print('Effective cost rate (Total_Cost / Sales):')
print(f'  Loss rows  : {df.loc[df.Profit < 0, "eff_cost_rate"].mean():.4f}  (>1.0 means loss)')
print(f'  Profit rows: {df.loc[df.Profit >= 0, "eff_cost_rate"].mean():.4f}')
print(f'  Min: {df["eff_cost_rate"].min():.4f}  Max: {df["eff_cost_rate"].max():.4f}')

print()
# Check correlation of features with Is_Loss
from scipy.stats import pointbiserialr
print('Correlation of each feature with Is_Loss (1=loss, 0=profit):')
df['Is_Loss_bin'] = (df['Profit'] < 0).astype(int)
for f in features:
    corr, pval = pointbiserialr(df['Is_Loss_bin'], df[f])
    print(f'  {f}: r={corr:.4f}  p={pval:.4f}')

print()
print('CONCLUSION: If all feature correlations with Is_Loss are near 0,')
print('loss is determined by RANDOM cost rates in the Excel formula,')
print('not by any observable product/outlet characteristic.')
print('No ML model can learn to predict random noise.')
