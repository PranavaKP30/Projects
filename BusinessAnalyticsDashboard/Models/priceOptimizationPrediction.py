"""
Price Optimization Model: Recommend optimal price per product to maximize revenue

Usage:
    python priceOptimizationPrediction.py [--model MODEL] [--visualize]

Options:
    --model MODEL      Regression model to use: xgboost (default), randomforest, linear
    --visualize        Generate and save PNG plots of price vs. predicted revenue for each product

Outputs:
    - Trained regression model files for each product (.joblib) in priceOptOutput/
    - Combined CSV of optimal prices, expected revenue, and model metrics in priceOptOutput/
    - (Optional) PNG plots for each product in priceOptOutput/ if --visualize is used

Example:
    python priceOptimizationPrediction.py --model xgboost --visualize
    # Recommends optimal prices for each product to maximize revenue and saves plots
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
import argparse

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '../Data'))
sales_path = os.path.join(DATA_DIR, 'sales.csv')
products_path = os.path.join(DATA_DIR, 'products.csv')
inventory_path = os.path.join(DATA_DIR, 'inventory.csv')

# Load data
sales = pd.read_csv(sales_path)
products = pd.read_csv(products_path, header=None)
inventory = pd.read_csv(inventory_path)

# Standardize inventory columns
inv_rename_dict = {}
if 'product_id' in inventory.columns:
    inv_rename_dict['product_id'] = 'ProductID'
if 'inventory_date' in inventory.columns:
    inv_rename_dict['inventory_date'] = 'Date'
inventory = inventory.rename(columns=inv_rename_dict)

# Standardize quantity column in inventory
if 'stock_on_hand_eod' in inventory.columns:
    inventory = inventory.rename(columns={'stock_on_hand_eod': 'Quantity'})

# Standardize columns
prod_rename_dict = {}
if 'product_id' in products.columns:
    prod_rename_dict['product_id'] = 'ProductID'
if 'product_name' in products.columns:
    prod_rename_dict['product_name'] = 'ProductName'
if 'category' in products.columns:
    prod_rename_dict['category'] = 'Category'
if 'supplier' in products.columns:
    prod_rename_dict['supplier'] = 'Supplier'
if 'shelf_life' in products.columns:
    prod_rename_dict['shelf_life'] = 'ShelfLife'
if 'min_order_qty' in products.columns:
    prod_rename_dict['min_order_qty'] = 'MinOrderQty'
products = products.rename(columns=prod_rename_dict)
if products.shape[1] > 1 and products.columns[0] != 'ProductID':
    product_colnames = ['ProductID', 'ProductName', 'Category', 'Supplier', 'ShelfLife', 'MinOrderQty', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Col12', 'Col13', 'Col14', 'Col15', 'Col16', 'Col17', 'Col18', 'Col19', 'Col20']
    if products.shape[1] > len(product_colnames):
        product_colnames += [f'Col{i}' for i in range(21, products.shape[1]+1)]
    products.columns = product_colnames[:products.shape[1]]
if 'ShelfLife' in products.columns:
    products['ShelfLife'] = pd.to_numeric(products['ShelfLife'], errors='coerce').fillna(0)
if 'MinOrderQty' in products.columns:
    products['MinOrderQty'] = pd.to_numeric(products['MinOrderQty'], errors='coerce').fillna(1)

rename_dict = {}
if 'sale_date' in sales.columns:
    rename_dict['sale_date'] = 'OrderDate'
if 'product_id' in sales.columns:
    rename_dict['product_id'] = 'ProductID'
if 'units_sold' in sales.columns:
    rename_dict['units_sold'] = 'Quantity'
if 'price' in sales.columns:
    rename_dict['price'] = 'Price'
if 'price_per_unit_at_sale' in sales.columns:
    rename_dict['price_per_unit_at_sale'] = 'Price'
if 'promotion_flag' in sales.columns:
    rename_dict['promotion_flag'] = 'Promotion'
sales = sales.rename(columns=rename_dict)
if 'Promotion' in sales.columns:
    sales['Promotion'] = sales['Promotion'].map(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
if 'OrderDate' in sales.columns:
    sales['OrderDate'] = pd.to_datetime(sales['OrderDate'], errors='coerce')
else:
    raise ValueError("sales.csv must contain a 'sale_date' or 'OrderDate' column. Please check your data file.")

# Merge product features into sales
data = sales.merge(products, on='ProductID', how='left')

# Feature engineering
data['month'] = data['OrderDate'].dt.month
# Lagged/rolling sales
for lag in [1, 7]:
    data[f'lag_{lag}'] = data.groupby('ProductID')['Quantity'].shift(lag)
for window in [7, 30]:
    data[f'roll_{window}'] = data.groupby('ProductID')['Quantity'].shift(1).rolling(window).mean().reset_index(0,drop=True)
# Inventory (optional)
if 'Date' not in inventory.columns and 'inventory_date' in inventory.columns:
    inventory = inventory.rename(columns={'inventory_date': 'Date'})
if 'Date' in inventory.columns:
    inventory['Date'] = pd.to_datetime(inventory['Date'], errors='coerce')
    inventory_daily = inventory.groupby(['ProductID', 'Date'])['Quantity'].sum().reset_index()
    data = data.merge(inventory_daily.rename(columns={'Date':'OrderDate', 'Quantity':'stock_on_hand'}), on=['ProductID','OrderDate'], how='left')
else:
    data['stock_on_hand'] = np.nan
# Fill missing
if 'Promotion' not in data.columns:
    data['Promotion'] = 0
data = data.fillna(0)

# Target: revenue = units_sold * price
data['Revenue'] = data['Quantity'] * data['Price']

parser = argparse.ArgumentParser(description='Price optimization for each product to maximize revenue')
parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'randomforest', 'linear'], help='Regression model to use')
parser.add_argument('--visualize', action='store_true', help='Visualize price vs. predicted revenue for each product')
args = parser.parse_args()
model_type = args.model
visualize = args.visualize


# Only the CSV goes to /Users/pranav/Coding/Projects/ThataRetail/Output
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'Output')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
csv_output_dir = OUTPUT_ROOT

# Other files (models, plots) go in priceOptOutput as before
model_plot_dir = os.path.join(BASE_DIR, 'priceOptOutput')
os.makedirs(model_plot_dir, exist_ok=True)

results = []



for pid, group in data.groupby('ProductID'):
    group = group.sort_values('OrderDate')
    # Remove strict data filter, allow prediction for all products
    # If not enough data, fill output with NaN
    if group['Quantity'].sum() < 1 or group['Price'].nunique() < 1:
        product_name = group['ProductName'].iloc[0] if 'ProductName' in group.columns else ''
        product_category = ''
        prod_row = products[products['ProductID'] == pid]
        if not prod_row.empty and 'Category' in prod_row.columns:
            product_category = prod_row.iloc[0]['Category']
        results.append({'ProductID': pid, 'ProductName': product_name, 'Category': product_category, 'OptimalPrice': np.nan, 'ExpectedRevenue': np.nan, 'MAE': np.nan, 'RMSE': np.nan})
        continue
    # Features for model
    feature_cols = ['Price', 'Promotion', 'month', 'lag_1', 'lag_7', 'roll_7', 'roll_30', 'stock_on_hand', 'ShelfLife', 'MinOrderQty']
    # Add category/supplier if present and numeric/categorical
    for col in ['Category', 'Supplier']:
        if col in group.columns and group[col].dtype == object:
            group[col] = group[col].astype('category').cat.codes
            feature_cols.append(col)
    X = group[feature_cols]
    y = group['Revenue']
    # Chronological split
    train_X, valid_X = X.iloc[:-30], X.iloc[-30:]
    train_y, valid_y = y.iloc[:-30], y.iloc[-30:]
    # Robust check: only fit if enough samples
    if train_X.shape[0] >= 2 and train_y.shape[0] >= 2:
        # Model selection
        if model_type == 'xgboost':
            model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        elif model_type == 'randomforest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
        model.fit(train_X, train_y)
        pred = model.predict(valid_X)
        mae = mean_absolute_error(valid_y, pred)
        rmse = np.sqrt(mean_squared_error(valid_y, pred))
        # Price grid for optimization
        price_min, price_max = group['Price'].min(), group['Price'].max()
        price_grid = np.linspace(price_min, price_max, 30)
        # Use median values for other features
        median_vals = {col: group[col].median() for col in feature_cols if col != 'Price'}
        opt_revenue = -np.inf
        opt_price = price_min
        pred_curve = []
        for p in price_grid:
            feat = median_vals.copy()
            feat['Price'] = p
            feat_df = pd.DataFrame([feat])
            # Ensure column order matches training
            feat_df = feat_df.reindex(columns=feature_cols)
            rev_pred = model.predict(feat_df)[0]
            pred_curve.append((p, rev_pred))
            if rev_pred > opt_revenue:
                opt_revenue = rev_pred
                opt_price = p
        # Get ProductName for this product
        product_name = group['ProductName'].iloc[0] if 'ProductName' in group.columns else ''
        # Get Category from products.csv
        product_category = ''
        prod_row = products[products['ProductID'] == pid]
        if not prod_row.empty and 'Category' in prod_row.columns:
            product_category = prod_row.iloc[0]['Category']
        results.append({'ProductID': pid, 'ProductName': product_name, 'Category': product_category, 'OptimalPrice': round(opt_price,2), 'ExpectedRevenue': round(opt_revenue,2), 'MAE': mae, 'RMSE': rmse})
        # Save model
        joblib.dump(model, os.path.join(model_plot_dir, f'price_opt_model_{pid}.joblib'))
        # Visualization (optional)
        if visualize:
            curve = np.array(pred_curve)
            plt.figure(figsize=(10,6))
            plt.plot(curve[:,0], curve[:,1], label='Predicted Revenue')
            plt.axvline(opt_price, color='r', linestyle='--', label=f'Optimal Price: {opt_price:.2f}')
            plt.title(f'Product {pid} Price Optimization')
            plt.xlabel('Price')
            plt.ylabel('Predicted Revenue')
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(model_plot_dir, f'product_{pid}_price_opt.png')
            plt.savefig(plot_path)
            plt.close()
    else:
        # Not enough samples for model training
        product_name = group['ProductName'].iloc[0] if 'ProductName' in group.columns else ''
        product_category = ''
        prod_row = products[products['ProductID'] == pid]
        if not prod_row.empty and 'Category' in prod_row.columns:
            product_category = prod_row.iloc[0]['Category']
        results.append({'ProductID': pid, 'ProductName': product_name, 'Category': product_category, 'OptimalPrice': np.nan, 'ExpectedRevenue': np.nan, 'MAE': np.nan, 'RMSE': np.nan})

# Save all output into a single CSV file
results_df = pd.DataFrame(results)
# Output CSV name matches this script's filename
script_base = os.path.splitext(os.path.basename(__file__))[0]
combined_path = os.path.join(csv_output_dir, f'{script_base}.csv')
results_df.to_csv(combined_path, index=False)
print(f'Saved price optimization results to {combined_path}')