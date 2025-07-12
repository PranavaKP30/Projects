"""
ABC Analysis (Inventory Classification)

Usage:
    python abcAnalysisModel.py [--window_days N] [--model MODEL] [--output_dir DIR] [--ml]

Options:
    --window_days N        Number of days for aggregation window (default: 90)
    --model MODEL          Model to use: xgboost (default), randomforest, logistic
    --output_dir DIR       Output directory (default: abcAnalysisOutput/)
    --ml                   Use ML model for ABC prediction (default: rule-based)

Outputs:
    - CSV with per-product ABC class, sales, and features
    - (If --ml) Trained model file (.joblib) and model metrics

"""

import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import accuracy_score, f1_score
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
parser = argparse.ArgumentParser(description='ABC Analysis (Inventory Classification)')
parser.add_argument('--window_days', type=int, default=90, help='Aggregation window in days (default: 90)')
parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'randomforest', 'logistic'], help='Model to use (if --ml)')
parser.add_argument('--output_dir', type=str, default='abcAnalysisOutput', help='Output directory')
parser.add_argument('--ml', action='store_true', help='Use ML model for ABC prediction (default: rule-based)')
args = parser.parse_args()
window_days = args.window_days
model_name = args.model


# Always output to /Users/pranav/Coding/Projects/ThataRetail/Output
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'Output')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
output_dir = OUTPUT_ROOT
use_ml = args.ml

# Load data
sales = pd.read_csv(sales_path)
products = pd.read_csv(products_path, header=None)
inventory = pd.read_csv(inventory_path)

# Standardize columns
rename_dict = {}
if 'sale_date' in sales.columns:
    rename_dict['sale_date'] = 'OrderDate'
if 'product_id' in sales.columns:
    rename_dict['product_id'] = 'ProductID'
if 'units_sold' in sales.columns:
    rename_dict['units_sold'] = 'Quantity'
if 'price' in sales.columns:
    rename_dict['price'] = 'Price'
sales = sales.rename(columns=rename_dict)
if 'OrderDate' in sales.columns:
    sales['OrderDate'] = pd.to_datetime(sales['OrderDate'], errors='coerce')
else:
    raise ValueError("sales.csv must contain a 'sale_date' or 'OrderDate' column. Please check your data file.")
# Products
prod_rename_dict = {}
if 'product_id' in products.columns:
    prod_rename_dict['product_id'] = 'ProductID'
if 'product_name' in products.columns:
    prod_rename_dict['product_name'] = 'ProductName'
if 'category' in products.columns:
    prod_rename_dict['category'] = 'Category'
products = products.rename(columns=prod_rename_dict)
if products.shape[1] > 1 and products.columns[0] != 'ProductID':
    product_colnames = ['ProductID', 'ProductName', 'Category', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Col12', 'Col13', 'Col14', 'Col15', 'Col16', 'Col17', 'Col18', 'Col19', 'Col20']
    if products.shape[1] > len(product_colnames):
        product_colnames += [f'Col{i}' for i in range(21, products.shape[1]+1)]
    products.columns = product_colnames[:products.shape[1]]
# Inventory (optional)
inv_rename_dict = {}
if 'product_id' in inventory.columns:
    inv_rename_dict['product_id'] = 'ProductID'
if 'inventory_date' in inventory.columns:
    inv_rename_dict['inventory_date'] = 'Date'
inventory = inventory.rename(columns=inv_rename_dict)
if 'stock_on_hand_eod' in inventory.columns:
    inventory = inventory.rename(columns={'stock_on_hand_eod': 'Quantity'})

# Filter to window_days
max_date = sales['OrderDate'].max()
min_date = max_date - pd.Timedelta(days=window_days)
sales_window = sales[(sales['OrderDate'] >= min_date) & (sales['OrderDate'] <= max_date)]

# Aggregate sales value and volume
if 'Price' in sales_window.columns:
    sales_window['SalesValue'] = sales_window['Quantity'] * sales_window['Price']
    sales_agg = sales_window.groupby('ProductID').agg({'Quantity': 'sum', 'SalesValue': 'sum'}).reset_index()
else:
    sales_agg = sales_window.groupby('ProductID').agg({'Quantity': 'sum'}).reset_index()
    sales_agg['SalesValue'] = sales_agg['Quantity']  # fallback: use volume as value

# Merge with product features
abc_data = sales_agg.merge(products, on='ProductID', how='left')

# Optionally add inventory features
if 'Quantity' in inventory.columns:
    inv_agg = inventory.groupby('ProductID')['Quantity'].mean().reset_index().rename(columns={'Quantity': 'AvgInventory'})
    abc_data = abc_data.merge(inv_agg, on='ProductID', how='left')

# ABC assignment (rule-based)
abc_data = abc_data.sort_values('SalesValue', ascending=False)
abc_data['CumulativeSales'] = abc_data['SalesValue'].cumsum()
total_sales = abc_data['SalesValue'].sum()
abc_data['CumulativeSalesPct'] = abc_data['CumulativeSales'] / total_sales

def assign_abc(pct):
    if pct <= 0.7:
        return 'A'
    elif pct <= 0.9:
        return 'B'
    else:
        return 'C'
abc_data['ABC_Class'] = abc_data['CumulativeSalesPct'].apply(assign_abc)

# Save rule-based results
out_cols = ['ProductID', 'ProductName', 'Category', 'Quantity', 'SalesValue', 'AvgInventory', 'ABC_Class']
out_cols = [col for col in out_cols if col in abc_data.columns]
# Output CSV name matches this script's filename
script_base = os.path.splitext(os.path.basename(__file__))[0]
results_path = os.path.join(output_dir, f'{script_base}.csv')
abc_data[out_cols].to_csv(results_path, index=False)
print(f"Saved ABC analysis results to {results_path}")

# If ML, train classifier to predict ABC class from features
if use_ml:
    # Encode ABC class
    class_map = {'A': 0, 'B': 1, 'C': 2}
    abc_data['ABC_Code'] = abc_data['ABC_Class'].map(class_map)
    feature_cols = ['Quantity', 'SalesValue', 'AvgInventory']
    if 'Category' in abc_data.columns:
        abc_data = pd.get_dummies(abc_data, columns=['Category'], drop_first=True)
        feature_cols += [col for col in abc_data.columns if col.startswith('Category_')]
    X = abc_data[feature_cols].fillna(0)
    y = abc_data['ABC_Code']
    # Chronological split not needed; use all for train/test
    train_X, test_X = X, X
    train_y, test_y = y, y
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
    f1 = f1_score(test_y, pred, average='weighted')
    metrics = {'Accuracy': acc, 'F1': f1}
    print(f"ML Classification metrics: Accuracy={acc:.3f}, F1={f1:.3f}")
    # Save model
    model_path = os.path.join(output_dir, f'abc_analysis_{model_name}.joblib')
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    # Save ML results
    inv_class_map = {v: k for k, v in class_map.items()}
    abc_data['Predicted_ABC_Class'] = [inv_class_map.get(p, 'unknown') for p in pred]
    ml_results_path = os.path.join(output_dir, 'abc_analysis_ml_results.csv')
    abc_data[out_cols + ['Predicted_ABC_Class']].to_csv(ml_results_path, index=False)
    print(f"Saved ML ABC analysis results to {ml_results_path}")
    # Save metrics
    metrics_path = os.path.join(output_dir, 'abc_analysis_metrics.csv')
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f"Saved metrics to {metrics_path}")