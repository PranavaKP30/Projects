"""
Inventory Turnover Analysis Model

Usage:
    python inventoryTurnoverModel.py [--window_days N] [--model_type regression|classification] [--model MODEL] [--output_dir DIR]

Options:
    --window_days N        Number of days for aggregation window (default: 90)
    --model_type TYPE      regression (predict ratio) or classification (predict class) [default: classification]
    --model MODEL          Model to use: xgboost (default), randomforest, logistic
    --output_dir DIR       Output directory (default: inventoryTurnoverOutput/)

Outputs:
    - Trained model file (.joblib) in output_dir
    - CSV with per-product turnover ratio, class, predicted class, and features
    - Model metrics (accuracy/F1 for classification, MAE/RMSE for regression)

"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '../Data'))
sales_path = os.path.join(DATA_DIR, 'sales.csv')
products_path = os.path.join(DATA_DIR, 'products.csv')
inventory_path = os.path.join(DATA_DIR, 'inventory.csv')

# CLI
parser = argparse.ArgumentParser(description='Inventory Turnover Analysis')
parser.add_argument('--window_days', type=int, default=90, help='Aggregation window in days (default: 90)')
parser.add_argument('--model_type', type=str, default='classification', choices=['classification', 'regression'], help='Model type: classification or regression')
parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'randomforest', 'logistic'], help='Model to use')
parser.add_argument('--output_dir', type=str, default='inventoryTurnoverOutput', help='Output directory')
args = parser.parse_args()
window_days = args.window_days
model_type = args.model_type
model_name = args.model

# Only the CSV goes to /Users/pranav/Coding/Projects/ThataRetail/Output
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'Output')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
csv_output_dir = OUTPUT_ROOT

# Other files (models, metrics) go in inventoryTurnoverOutput as before
model_metrics_dir = os.path.join(BASE_DIR, args.output_dir)
os.makedirs(model_metrics_dir, exist_ok=True)

# Load data
sales = pd.read_csv(sales_path)
products = pd.read_csv(products_path, header=None)
inventory = pd.read_csv(inventory_path)

# Standardize columns (reuse conventions from other scripts)
# Inventory
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
if 'category' in products.columns:
    prod_rename_dict['category'] = 'Category'
if 'shelf_life' in products.columns:
    prod_rename_dict['shelf_life'] = 'ShelfLife'
products = products.rename(columns=prod_rename_dict)
if products.shape[1] > 1 and products.columns[0] != 'ProductID':
    product_colnames = ['ProductID', 'ProductName', 'Category', 'ShelfLife', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Col12', 'Col13', 'Col14', 'Col15', 'Col16', 'Col17', 'Col18', 'Col19', 'Col20']
    if products.shape[1] > len(product_colnames):
        product_colnames += [f'Col{i}' for i in range(21, products.shape[1]+1)]
    products.columns = product_colnames[:products.shape[1]]
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

# Filter to window_days
max_date = sales['OrderDate'].max()
min_date = max_date - pd.Timedelta(days=window_days)
sales_window = sales[(sales['OrderDate'] >= min_date) & (sales['OrderDate'] <= max_date)]
inventory['Date'] = pd.to_datetime(inventory['Date'], errors='coerce')
inventory_window = inventory[(inventory['Date'] >= min_date) & (inventory['Date'] <= max_date)]

# Aggregate features
sales_agg = sales_window.groupby('ProductID')['Quantity'].sum().reset_index().rename(columns={'Quantity': 'TotalUnitsSold'})
inv_agg = inventory_window.groupby('ProductID')['Quantity'].mean().reset_index().rename(columns={'Quantity': 'AvgInventory'})

# Merge all features
data = sales_agg.merge(inv_agg, on='ProductID', how='outer').merge(products, on='ProductID', how='left')

# Fill missing
data['TotalUnitsSold'] = data['TotalUnitsSold'].fillna(0)
data['AvgInventory'] = data['AvgInventory'].fillna(1)  # avoid div by zero

# Turnover ratio
data['TurnoverRatio'] = data['TotalUnitsSold'] / data['AvgInventory']

# Date/time features (use max_date for seasonality)
data['Month'] = max_date.month
data['Quarter'] = max_date.quarter

# Turnover class (fast/medium/slow)

quantiles = data['TurnoverRatio'].quantile([0.33, 0.66]).values
bins = [-np.inf, quantiles[0], quantiles[1], np.inf]
labels = ['slow', 'medium', 'fast']
data['TurnoverClass'] = pd.cut(data['TurnoverRatio'], bins=bins, labels=labels)
# Encode class as integer for model
class_map = {'slow': 0, 'medium': 1, 'fast': 2}
data['TurnoverClassCode'] = data['TurnoverClass'].map(class_map)

# Features for model

# Ensure ShelfLife is numeric
if 'ShelfLife' in data.columns:
    data['ShelfLife'] = pd.to_numeric(data['ShelfLife'], errors='coerce').fillna(0)

feature_cols = ['TotalUnitsSold', 'AvgInventory', 'ShelfLife', 'Month', 'Quarter']
if 'Category' in data.columns:
    data = pd.get_dummies(data, columns=['Category'], drop_first=True)
    feature_cols += [col for col in data.columns if col.startswith('Category_')]
X = data[feature_cols].fillna(0)

# Target
y_reg = data['TurnoverRatio']
y_class = data['TurnoverClassCode']

# Chronological split (train/test)
# Here, use all for train; for real use, split by time or hold out some products
train_X, test_X = X, X
train_y_reg, test_y_reg = y_reg, y_reg
train_y_class, test_y_class = y_class, y_class

# Model selection
if model_type == 'regression':
    if model_name == 'xgboost':
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    elif model_name == 'randomforest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    model.fit(train_X, train_y_reg)
    pred = model.predict(test_X)
    mae = mean_absolute_error(test_y_reg, pred)
    rmse = mean_squared_error(test_y_reg, pred, squared=False)
    metrics = {'MAE': mae, 'RMSE': rmse}
    data['PredictedTurnoverRatio'] = pred
    print(f"Regression metrics: MAE={mae:.3f}, RMSE={rmse:.3f}")
else:
    if model_name == 'xgboost':
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    elif model_name == 'randomforest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
    model.fit(train_X, train_y_class)
    pred = model.predict(test_X)
    acc = accuracy_score(test_y_class, pred)
    f1 = f1_score(test_y_class, pred, average='weighted')
    metrics = {'Accuracy': acc, 'F1': f1}
    # Decode predictions back to string labels for output
    inv_class_map = {v: k for k, v in class_map.items()}
    data['PredictedTurnoverClass'] = [inv_class_map.get(p, 'unknown') for p in pred]
    print(f"Classification metrics: Accuracy={acc:.3f}, F1={f1:.3f}")

# Save model

model_path = os.path.join(model_metrics_dir, f'inventory_turnover_{model_type}_{model_name}.joblib')
joblib.dump(model, model_path)
print(f"Saved model to {model_path}")

# Save results

out_cols = ['ProductID', 'ProductName', 'TotalUnitsSold', 'AvgInventory', 'TurnoverRatio', 'TurnoverClass', 'Month', 'Quarter']
if 'Category' in data.columns:
    out_cols.insert(2, 'Category')
if model_type == 'regression':
    out_cols.append('PredictedTurnoverRatio')
else:
    out_cols.append('PredictedTurnoverClass')
# Add any category dummies
out_cols += [col for col in data.columns if col.startswith('Category_')]

script_base = os.path.splitext(os.path.basename(__file__))[0]
results_path = os.path.join(csv_output_dir, f'{script_base}.csv')
data[out_cols].to_csv(results_path, index=False)
print(f"Saved results to {results_path}")

# Save metrics

metrics_path = os.path.join(model_metrics_dir, 'inventory_turnover_metrics.csv')
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"Saved metrics to {metrics_path}")
