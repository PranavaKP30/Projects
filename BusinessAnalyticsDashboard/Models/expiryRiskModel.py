"""
Expiry Risk Prediction Model

Usage:
    python expiryRiskModel.py [--model MODEL] [--window_days N] [--test_split_days N] [--output_dir DIR]

Options:
    --model MODEL          Model to use: xgboost (default), randomforest, logistic
    --window_days N        Number of days for feature window (default: 90)
    --test_split_days N    Number of most recent days to use for test/validation (default: 30)
    --output_dir DIR       Output directory (default: expiryRiskOutput/)

Outputs:
    - Trained model file (.joblib) in output_dir
    - CSV with per-product, per-date expiry risk predictions and true flags
    - Model metrics (accuracy, precision, recall, F1)

"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '../Data'))
sales_path = os.path.join(DATA_DIR, 'sales.csv')
products_path = os.path.join(DATA_DIR, 'products.csv')
inventory_path = os.path.join(DATA_DIR, 'inventory.csv')

# CLI
parser = argparse.ArgumentParser(description='Expiry Risk Prediction')
parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'randomforest', 'logistic'], help='Model to use')
parser.add_argument('--window_days', type=int, default=90, help='Feature window in days (default: 90)')
parser.add_argument('--test_split_days', type=int, default=30, help='Days for test split (default: 30)')
parser.add_argument('--output_dir', type=str, default='expiryRiskOutput', help='Output directory')
args = parser.parse_args()
model_name = args.model
window_days = args.window_days
test_split_days = args.test_split_days

# Only the CSV goes to /Users/pranav/Coding/Projects/ThataRetail/Output
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'Output')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
csv_output_dir = OUTPUT_ROOT

# Other files (models, metrics) go in expiryRiskOutput as before
model_metrics_dir = os.path.join(BASE_DIR, args.output_dir)
os.makedirs(model_metrics_dir, exist_ok=True)

# Load data
sales = pd.read_csv(sales_path)
products = pd.read_csv(products_path, header=None)
inventory = pd.read_csv(inventory_path)

# Standardize columns
inv_rename_dict = {}
if 'product_id' in inventory.columns:
    inv_rename_dict['product_id'] = 'ProductID'
if 'inventory_date' in inventory.columns:
    inv_rename_dict['inventory_date'] = 'Date'
inventory = inventory.rename(columns=inv_rename_dict)
if 'stock_on_hand_eod' in inventory.columns:
    inventory = inventory.rename(columns={'stock_on_hand_eod': 'Quantity'})
# Products
prod_rename_dict = {}
if 'product_id' in products.columns:
    prod_rename_dict['product_id'] = 'ProductID'
if 'product_name' in products.columns:
    prod_rename_dict['product_name'] = 'ProductName'
if 'shelf_life' in products.columns:
    prod_rename_dict['shelf_life'] = 'ShelfLife'
products = products.rename(columns=prod_rename_dict)
if products.shape[1] > 1 and products.columns[0] != 'ProductID':
    product_colnames = ['ProductID', 'ProductName', 'ShelfLife', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Col12', 'Col13', 'Col14', 'Col15', 'Col16', 'Col17', 'Col18', 'Col19', 'Col20']
    if products.shape[1] > len(product_colnames):
        product_colnames += [f'Col{i}' for i in range(21, products.shape[1]+1)]
    products.columns = product_colnames[:products.shape[1]]
if 'ShelfLife' in products.columns:
    products['ShelfLife'] = pd.to_numeric(products['ShelfLife'], errors='coerce').fillna(30)
# Sales
rename_dict = {}
if 'sale_date' in sales.columns:
    rename_dict['sale_date'] = 'OrderDate'
if 'product_id' in sales.columns:
    rename_dict['product_id'] = 'ProductID'
if 'units_sold' in sales.columns:
    rename_dict['units_sold'] = 'Quantity'
sales = sales.rename(columns=rename_dict)
if 'OrderDate' in sales.columns:
    sales['OrderDate'] = pd.to_datetime(sales['OrderDate'], errors='coerce')
else:
    raise ValueError("sales.csv must contain a 'sale_date' or 'OrderDate' column. Please check your data file.")
inventory['Date'] = pd.to_datetime(inventory['Date'], errors='coerce')

# Filter to window_days
max_date = inventory['Date'].max()
min_date = max_date - pd.Timedelta(days=window_days)
inventory = inventory[(inventory['Date'] >= min_date) & (inventory['Date'] <= max_date)]
sales = sales[(sales['OrderDate'] >= min_date) & (sales['OrderDate'] <= max_date)]

# Feature engineering
# Recent sales rate (7-day rolling avg)
sales['OrderDate'] = pd.to_datetime(sales['OrderDate'], errors='coerce')
sales = sales.sort_values(['ProductID', 'OrderDate'])
sales['rolling_7d_sales'] = sales.groupby('ProductID')['Quantity'].rolling(7, min_periods=1).mean().reset_index(0, drop=True)
sales_7d = sales.groupby(['ProductID', 'OrderDate'])['rolling_7d_sales'].mean().reset_index()

# Merge recent sales rate into inventory
data = inventory.merge(products, on='ProductID', how='left')
data = data.merge(sales_7d.rename(columns={'OrderDate': 'Date'}), on=['ProductID', 'Date'], how='left')
data['rolling_7d_sales'] = data['rolling_7d_sales'].fillna(0)

# Inventory age: days since received (assume Date is daily snapshot, so age = days since min(Date) for that product)
data['InventoryAge'] = data.groupby('ProductID')['Date'].transform(lambda x: (x - x.min()).dt.days)

# Date/time features
if 'Date' in data.columns:
    data['Month'] = data['Date'].dt.month
    data['Quarter'] = data['Date'].dt.quarter
    data['DayOfWeek'] = data['Date'].dt.dayofweek

# Target: expiry risk flag
def compute_expiry_risk(row):
    # If no stock, no risk
    if row['Quantity'] <= 0:
        return 0
    # If no sales, all stock at risk
    if row['rolling_7d_sales'] <= 0:
        return 1
    # Projected days to sell all stock
    days_to_sell = row['Quantity'] / row['rolling_7d_sales']
    # If inventory age + days to sell > shelf life, at risk
    if (row['InventoryAge'] + days_to_sell) > row['ShelfLife']:
        return 1
    return 0

data['ExpiryRisk'] = data.apply(compute_expiry_risk, axis=1)

# Features for model
feature_cols = ['Quantity', 'InventoryAge', 'ShelfLife', 'rolling_7d_sales', 'Month', 'Quarter', 'DayOfWeek']
X = data[feature_cols].fillna(0)
y = data['ExpiryRisk']

# Chronological split: train on older, test on most recent test_split_days
test_cutoff = max_date - pd.Timedelta(days=test_split_days)
train_idx = data['Date'] < test_cutoff
test_idx = data['Date'] >= test_cutoff
train_X, test_X = X[train_idx], X[test_idx]
train_y, test_y = y[train_idx], y[test_idx]

# Model selection
if model_name == 'xgboost':
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
elif model_name == 'randomforest':
    model = RandomForestClassifier(n_estimators=100, random_state=42)
else:
    model = LogisticRegression(max_iter=1000)
model.fit(train_X, train_y)
pred = model.predict(test_X)
acc = accuracy_score(test_y, pred)
prec = precision_score(test_y, pred, zero_division=0)
rec = recall_score(test_y, pred, zero_division=0)
f1 = f1_score(test_y, pred, zero_division=0)
metrics = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
print(f"Test metrics: Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

# Save model

model_path = os.path.join(model_metrics_dir, f'expiry_risk_{model_name}.joblib')
joblib.dump(model, model_path)
print(f"Saved model to {model_path}")

# Save results
results = data.loc[test_idx, ['ProductID', 'ProductName', 'Date', 'Quantity', 'InventoryAge', 'ShelfLife', 'rolling_7d_sales', 'ExpiryRisk']].copy()
results['PredictedExpiryRisk'] = pred

script_base = os.path.splitext(os.path.basename(__file__))[0]
results_path = os.path.join(csv_output_dir, f'{script_base}.csv')
results.to_csv(results_path, index=False)
print(f"Saved results to {results_path}")

# Save metrics

metrics_path = os.path.join(model_metrics_dir, 'expiry_risk_metrics.csv')
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"Saved metrics to {metrics_path}")
