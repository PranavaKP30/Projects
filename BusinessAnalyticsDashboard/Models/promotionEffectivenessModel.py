"""
Promotion Effectiveness Model: Quantify the impact of promotions/discounts on sales

Usage:
    python promotionEffectivenessModel.py [--target TARGET] [--model MODEL] [--visualize]

Options:
    --target TARGET    Target variable: units_sold (default) or uplift
    --model MODEL      Regression model to use: xgboost (default), linear
    --visualize        Generate and save PNG plots of actual vs. predicted sales/uplift for each product

Outputs:
    - Trained regression model files for each product (.joblib) in promoEffectOutput/
    - Combined CSV of predictions, metrics, and (optionally) uplift in promoEffectOutput/
    - (Optional) PNG plots for each product in promoEffectOutput/ if --visualize is used

Example:
    python promotionEffectivenessModel.py --target units_sold --model xgboost --visualize
    # Quantifies promotion impact and saves plots
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
if 'stock_on_hand_eod' in inventory.columns:
    inventory = inventory.rename(columns={'stock_on_hand_eod': 'Quantity'})

# Standardize product columns
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
if 'ShelfLife' in products.columns:
    products['ShelfLife'] = pd.to_numeric(products['ShelfLife'], errors='coerce').fillna(0)

# Standardize sales columns
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
data['dayofweek'] = data['OrderDate'].dt.dayofweek
data['month'] = data['OrderDate'].dt.month
# Lagged/rolling sales
for lag in [1, 7]:
    data[f'lag_{lag}'] = data.groupby('ProductID')['Quantity'].shift(lag)
for window in [7, 30]:
    data[f'roll_{window}'] = data.groupby('ProductID')['Quantity'].shift(1).rolling(window).mean().reset_index(0,drop=True)
# Inventory (optional)
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

parser = argparse.ArgumentParser(description='Promotion effectiveness modeling')
parser.add_argument('--target', type=str, default='uplift', choices=['units_sold', 'uplift'], help='Target variable')
parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'linear'], help='Regression model to use')
parser.add_argument('--visualize', action='store_true', help='Visualize actual vs. predicted sales/uplift for each product')
args = parser.parse_args()
target = args.target
model_type = args.model
visualize = args.visualize


# Only the CSV goes to /Users/pranav/Coding/Projects/ThataRetail/Output
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'Output')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
csv_output_dir = OUTPUT_ROOT

# Other files (models, plots, per-product uplift) go in promoEffectOutput as before
model_plot_dir = os.path.join(BASE_DIR, 'promoEffectOutput')
os.makedirs(model_plot_dir, exist_ok=True)

results = []

for pid, group in data.groupby('ProductID'):
    group = group.sort_values('OrderDate')
    if group['Quantity'].sum() < 10 or group['Promotion'].sum() < 3:
        continue  # Not enough data or promotions
    # Features for model
    feature_cols = ['Price', 'Promotion', 'dayofweek', 'month', 'lag_1', 'lag_7', 'roll_7', 'roll_30', 'stock_on_hand', 'ShelfLife']
    if 'Category' in group.columns and group['Category'].dtype == object:
        group['Category'] = group['Category'].astype('category').cat.codes
        feature_cols.append('Category')
    X = group[feature_cols]
    if target == 'units_sold':
        y = group['Quantity']
        output_rows = []
    else:
        # Uplift: difference between actual sales and expected sales without promotion (fit on non-promo, predict on promo)
        non_promo = group[group['Promotion'] == 0]
        promo = group[group['Promotion'] == 1]
        if len(non_promo) < 10 or len(promo) < 3:
            continue
        base_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42) if model_type == 'xgboost' else LinearRegression()
        base_model.fit(non_promo[feature_cols], non_promo['Quantity'])
        promo = promo.copy()
        promo['expected_no_promo'] = base_model.predict(promo[feature_cols])
        promo['uplift'] = promo['Quantity'] - promo['expected_no_promo']
        # Add ProductName if available
        if 'ProductName' in promo.columns:
            output_rows = promo[['OrderDate', 'ProductID', 'ProductName', 'Quantity', 'expected_no_promo', 'uplift']].copy().to_dict('records')
        else:
            output_rows = promo[['OrderDate', 'ProductID', 'Quantity', 'expected_no_promo', 'uplift']].copy().to_dict('records')
        y = promo['uplift']
        X = promo[feature_cols]
    # Chronological split
    train_X, valid_X = X.iloc[:-10], X.iloc[-10:]
    train_y, valid_y = y.iloc[:-10], y.iloc[-10:]
    # Model selection
    if model_type == 'xgboost':
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    else:
        model = LinearRegression()
    model.fit(train_X, train_y)
    pred = model.predict(valid_X)
    mae = mean_absolute_error(valid_y, pred)
    rmse = np.sqrt(mean_squared_error(valid_y, pred))
    # Save per-product summary
    results.append({'ProductID': pid, 'MAE': mae, 'RMSE': rmse})
    # Save per-row uplift if target is uplift
    if target == 'uplift' and output_rows:
        uplift_path = os.path.join(model_plot_dir, f'{pid}_uplift_details.csv')
        pd.DataFrame(output_rows).to_csv(uplift_path, index=False)
    # Visualization (optional)
    if visualize:
        plt.figure(figsize=(10,6))
        plt.plot(valid_X.index, valid_y, label='Actual', marker='o')
        plt.plot(valid_X.index, pred, label='Predicted', marker='x')
        plt.title(f'Product {pid} Promotion Effectiveness')
        plt.xlabel('Index')
        plt.ylabel('Sales' if target=='units_sold' else 'Uplift')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(model_plot_dir, f'product_{pid}_promo_effect.png')
        plt.savefig(plot_path)
        plt.close()
    # Save model
    joblib.dump(model, os.path.join(model_plot_dir, f'promo_effect_model_{pid}.joblib'))

# Save all output into a single CSV file

# Combine all per-row uplift details into a single DataFrame if target is uplift
if target == 'uplift':
    all_uplift_rows = []
    for pid in [r['ProductID'] for r in results]:
        uplift_path = os.path.join(model_plot_dir, f'{pid}_uplift_details.csv')
        if os.path.exists(uplift_path):
            df = pd.read_csv(uplift_path)
            # Add Category from products.csv
            if 'ProductID' in df.columns:
                cat_map = products.set_index('ProductID')['Category'] if 'Category' in products.columns else None
                if cat_map is not None:
                    df['Category'] = df['ProductID'].map(cat_map)
            all_uplift_rows.append(df)
    if all_uplift_rows:
        uplift_df = pd.concat(all_uplift_rows, ignore_index=True)
        # Merge with summary metrics
        results_df = pd.DataFrame(results)
        merged = pd.merge(uplift_df, results_df, on='ProductID', how='left')
        script_base = os.path.splitext(os.path.basename(__file__))[0]
        combined_path = os.path.join(csv_output_dir, f'{script_base}.csv')
        merged.to_csv(combined_path, index=False)
        print(f'Saved promotion effectiveness results with uplift details to {combined_path}')
    else:
        results_df = pd.DataFrame(results)
        script_base = os.path.splitext(os.path.basename(__file__))[0]
        combined_path = os.path.join(csv_output_dir, f'{script_base}.csv')
        results_df.to_csv(combined_path, index=False)
        print(f'Saved promotion effectiveness results to {combined_path}')
else:
    results_df = pd.DataFrame(results)
    script_base = os.path.splitext(os.path.basename(__file__))[0]
    combined_path = os.path.join(csv_output_dir, f'{script_base}.csv')
    results_df.to_csv(combined_path, index=False)
    print(f'Saved promotion effectiveness results to {combined_path}')