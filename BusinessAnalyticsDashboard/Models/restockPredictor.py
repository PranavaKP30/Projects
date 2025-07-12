"""
Restock Predictor for Each Product for the Next n Days

Usage:
    python restockPredictor.py [--predict_days N] [--visualize]

Options:
    --predict_days N   Number of days to predict in advance (default: 7)
    --visualize        Generate and save PNG plots of restock alerts and quantities for each product

Outputs:
    - Trained classification and regression model files for each product (.joblib) in restockOutput/
    - Combined CSV of model metrics and predictions in restockOutput/
    - (Optional) PNG plots for each product in restockOutput/ if --visualize is used

Example:
    python restockPredictor.py --predict_days 7 --visualize
    # Predicts restock alerts and quantities for the next 7 days for each product and saves plots
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
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
products = pd.read_csv(products_path)
inventory = pd.read_csv(inventory_path)

# Standardize columns
prod_rename_dict = {}
if 'product_id' in products.columns:
    prod_rename_dict['product_id'] = 'ProductID'
if 'product_name' in products.columns:
    prod_rename_dict['product_name'] = 'ProductName'
if 'category' in products.columns:
    prod_rename_dict['category'] = 'Category'
if 'shelf_life' in products.columns:
    prod_rename_dict['shelf_life'] = 'ShelfLife'
if 'min_order_qty' in products.columns:
    prod_rename_dict['min_order_qty'] = 'MinOrderQty'
products = products.rename(columns=prod_rename_dict)
# Dynamically set column names to preserve all columns, but ensure required ones are present
required_cols = ['ProductID', 'ProductName', 'Category', 'ShelfLife', 'MinOrderQty']
for col in required_cols:
    if col not in products.columns:
        products[col] = ''
if 'MinOrderQty' in products.columns:
    products['MinOrderQty'] = pd.to_numeric(products['MinOrderQty'], errors='coerce').fillna(1)
rename_dict = {}
if 'sale_date' in sales.columns:
    rename_dict['sale_date'] = 'OrderDate'
if 'product_id' in sales.columns:
    rename_dict['product_id'] = 'ProductID'
if 'units_sold' in sales.columns:
    rename_dict['units_sold'] = 'Quantity'
if 'promotion_flag' in sales.columns:
    rename_dict['promotion_flag'] = 'Promotion'
sales = sales.rename(columns=rename_dict)
if 'Promotion' in sales.columns:
    sales['Promotion'] = sales['Promotion'].map(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
if 'OrderDate' in sales.columns:
    sales['OrderDate'] = pd.to_datetime(sales['OrderDate'], errors='coerce')
else:
    raise ValueError("sales.csv must contain a 'sale_date' or 'OrderDate' column. Please check your data file.")
inv_rename_dict = {}
if 'product_id' in inventory.columns:
    inv_rename_dict['product_id'] = 'ProductID'
if 'inventory_date' in inventory.columns:
    inv_rename_dict['inventory_date'] = 'Date'
inventory = inventory.rename(columns=inv_rename_dict)
if 'stock_on_hand_eod' in inventory.columns:
    inventory = inventory.rename(columns={'stock_on_hand_eod': 'Quantity'})
if 'Date' in inventory.columns:
    inventory['Date'] = pd.to_datetime(inventory['Date'], errors='coerce')
elif 'AdjustmentDate' in inventory.columns:
    inventory['Date'] = pd.to_datetime(inventory['AdjustmentDate'], errors='coerce')
else:
    raise ValueError(f"inventory.csv must contain a 'Date' or 'AdjustmentDate' column. Columns found: {list(inventory.columns)}")

def create_features(df, inventory, products, predict_days):
    df = df.copy()
    df['dayofweek'] = df['OrderDate'].dt.dayofweek
    df['month'] = df['OrderDate'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    df = df.sort_values(['ProductID', 'OrderDate'])
    df['lag_1'] = df.groupby('ProductID')['Quantity'].shift(1)
    df['lag_2'] = df.groupby('ProductID')['Quantity'].shift(2)
    df['lag_7'] = df.groupby('ProductID')['Quantity'].shift(7)
    df['roll_7'] = df.groupby('ProductID')['Quantity'].shift(1).rolling(7).mean().reset_index(0,drop=True)
    df['roll_30'] = df.groupby('ProductID')['Quantity'].shift(1).rolling(30).mean().reset_index(0,drop=True)
    # Merge product features
    # Only merge columns that exist in products
    prod_cols = ['ProductID', 'ProductName']
    if 'ShelfLife' in products.columns:
        prod_cols.append('ShelfLife')
    if 'MinOrderQty' in products.columns:
        prod_cols.append('MinOrderQty')
    df = df.merge(products[prod_cols], on='ProductID', how='left')
    # Ensure MinOrderQty is numeric and replace missing/invalid with 1
    if 'MinOrderQty' in df.columns:
        df['MinOrderQty'] = pd.to_numeric(df['MinOrderQty'], errors='coerce').fillna(1)
    # Merge inventory (stock_on_hand)
    inventory_daily = inventory.groupby(['ProductID', 'Date'])['Quantity'].sum().reset_index()
    df = df.merge(inventory_daily.rename(columns={'Date':'OrderDate', 'Quantity':'stock_on_hand'}), on=['ProductID','OrderDate'], how='left')
    if 'Promotion' in df.columns:
        df['promotion_flag'] = df['Promotion'].fillna(0)
    else:
        df['promotion_flag'] = 0
    df = df.fillna(0)
    # Predicted demand for next n days (rolling sum of Quantity)
    df['future_demand'] = df.groupby('ProductID')['Quantity'].shift(-1).rolling(predict_days).sum().reset_index(0,drop=True)
    # Restock threshold: if stock_on_hand < future_demand * 1.1 (10% buffer), needs restock
    df['restock_alert'] = (df['stock_on_hand'] < df['future_demand'] * 1.1).astype(int)
    # Restock quantity: if alert, order enough to cover future demand + buffer - current stock, else 0
    df['restock_qty'] = np.where(df['restock_alert'] == 1, np.maximum(df['future_demand'] * 1.1 - df['stock_on_hand'], 0), 0)
    # Optionally, round up to min order qty if available
    if 'MinOrderQty' in df.columns:
        # Ensure MinOrderQty is numeric and replace 0 or NaN with 1 to avoid division errors
        df['MinOrderQty'] = pd.to_numeric(df['MinOrderQty'], errors='coerce').fillna(1).replace(0, 1)
        df['restock_qty'] = np.where(
            df['restock_qty'] > 0,
            np.ceil(df['restock_qty'] / df['MinOrderQty']).astype(int) * df['MinOrderQty'],
            0
        )
    return df

sales['ProductID'] = sales['ProductID'].astype(str)
products['ProductID'] = products['ProductID'].astype(str)
inventory['ProductID'] = inventory['ProductID'].astype(str)
sales = sales[sales['OrderDate'].notna()]
sales = sales.sort_values(['ProductID', 'OrderDate'])

parser = argparse.ArgumentParser(description='Restock prediction for next n days')
parser.add_argument('--predict_days', type=int, default=7, help='Number of days to predict in advance (default: 7)')
parser.add_argument('--visualize', action='store_true', help='Visualize restock alerts and quantities for each product')
args = parser.parse_args()
predict_days = args.predict_days
visualize = args.visualize


# Only the CSV goes to /Users/pranav/Coding/Projects/ThataRetail/Output
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'Output')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
csv_output_dir = OUTPUT_ROOT

# Other files (models, plots) go in restockOutput as before
model_plot_dir = os.path.join(BASE_DIR, 'restockOutput')
os.makedirs(model_plot_dir, exist_ok=True)

results = []
restock_predictions = []


for pid, group in sales.groupby('ProductID'):
    group = group.sort_values('OrderDate')
    # Lower threshold to 1 to allow prediction for all products
    if len(group) < 1:
        continue
    # Get ProductName for this ProductID
    product_name = ''
    product_category = ''
    prod_row = products[products['ProductID'] == pid]
    if not prod_row.empty:
        product_name = prod_row.iloc[0]['ProductName'] if 'ProductName' in prod_row.columns else ''
        product_category = prod_row.iloc[0]['Category'] if 'Category' in prod_row.columns else ''
    df_feat = create_features(group, inventory, products, predict_days)
    feature_cols = [
        'dayofweek', 'month', 'is_weekend', 'lag_1', 'lag_2', 'lag_7',
        'roll_7', 'roll_30', 'stock_on_hand', 'promotion_flag',
        'future_demand'
    ]
    if 'ShelfLife' in df_feat.columns:
        feature_cols.append('ShelfLife')
    if 'MinOrderQty' in df_feat.columns:
        feature_cols.append('MinOrderQty')
    # Remove rows with missing targets
    mask = (~df_feat['restock_alert'].isna()) & (~df_feat['restock_qty'].isna())
    X = df_feat.loc[mask, feature_cols]
    # Ensure all features are numeric; drop non-numeric columns if any
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"[Warning] Dropping non-numeric columns for ProductID {pid}: {non_numeric}")
        X = X.select_dtypes(include=[np.number])
    y_alert = df_feat.loc[mask, 'restock_alert']
    y_qty = df_feat.loc[mask, 'restock_qty']
    # Chronological split
    train_X, valid_X = X.iloc[:-30], X.iloc[-30:]
    train_y_alert, valid_y_alert = y_alert.iloc[:-30], y_alert.iloc[-30:]
    train_y_qty, valid_y_qty = y_qty.iloc[:-30], y_qty.iloc[-30:]
    # Robust check: only fit if enough samples
    if train_X.shape[0] >= 2 and train_y_alert.shape[0] >= 2 and train_y_qty.shape[0] >= 2:
        # Classification model for restock alert
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(train_X, train_y_alert)
        alert_pred = clf.predict(valid_X)
        acc = accuracy_score(valid_y_alert, alert_pred)
        prec = precision_score(valid_y_alert, alert_pred, zero_division=0)
        rec = recall_score(valid_y_alert, alert_pred, zero_division=0)
        f1 = f1_score(valid_y_alert, alert_pred, zero_division=0)
        # Regression model for restock quantity
        reg = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        reg.fit(train_X, train_y_qty)
        qty_pred = reg.predict(valid_X)
        mae = mean_absolute_error(valid_y_qty, qty_pred)
        rmse = np.sqrt(mean_squared_error(valid_y_qty, qty_pred))
        results.append({'ProductID': pid, 'ProductName': product_name, 'Category': product_category, 'Alert_Accuracy': acc, 'Alert_Precision': prec, 'Alert_Recall': rec, 'Alert_F1': f1, 'Qty_MAE': mae, 'Qty_RMSE': rmse})
        # Predict for next day
        last_feat = df_feat.iloc[[-1]][feature_cols]
        # Ensure last_feat matches the columns used for training (numeric only)
        last_feat = last_feat.select_dtypes(include=[np.number])
        next_qty = reg.predict(last_feat)[0]
        # If predicted restock quantity > 0, force alert to 1
        if next_qty > 0:
            next_alert = 1
        else:
            next_alert = clf.predict(last_feat)[0]
        restock_predictions.append({'ProductID': pid, 'ProductName': product_name, 'Category': product_category, 'NextDayAlert': int(next_alert), 'NextDayRestockQty': max(0, int(round(next_qty)))})
        # Save models
        joblib.dump(clf, os.path.join(model_plot_dir, f'restock_alert_model_{pid}.joblib'))
        joblib.dump(reg, os.path.join(model_plot_dir, f'restock_qty_model_{pid}.joblib'))
        # Visualization (optional)
        if visualize:
            plt.figure(figsize=(12,6))
            plt.plot(df_feat['OrderDate'], df_feat['stock_on_hand'], label='Stock On Hand', marker='o')
            plt.plot(df_feat['OrderDate'], df_feat['future_demand'], label='Future Demand', marker='x')
            plt.plot(df_feat['OrderDate'], df_feat['restock_qty'], label='Restock Qty', marker='s')
            plt.title(f'Product {pid} Restock Prediction')
            plt.xlabel('Date')
            plt.ylabel('Units')
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(model_plot_dir, f'product_{pid}_restock.png')
            plt.savefig(plot_path)
            plt.close()
    else:
        # Not enough samples for model training
        results.append({'ProductID': pid, 'ProductName': product_name, 'Category': product_category, 'Alert_Accuracy': np.nan, 'Alert_Precision': np.nan, 'Alert_Recall': np.nan, 'Alert_F1': np.nan, 'Qty_MAE': np.nan, 'Qty_RMSE': np.nan})
        restock_predictions.append({'ProductID': pid, 'ProductName': product_name, 'Category': product_category, 'NextDayAlert': np.nan, 'NextDayRestockQty': np.nan})

# Save all output into a single CSV file

# Add Category column from products.csv if available
results_df = pd.DataFrame(results)
pred_df = pd.DataFrame(restock_predictions)
merge_cols = ['ProductID', 'ProductName', 'Category']
# Only merge if all merge columns exist in both DataFrames
missing_cols = [col for col in merge_cols if col not in results_df.columns or col not in pred_df.columns]
if missing_cols:
    print(f"Warning: Missing columns for merge: {missing_cols}. Merging on intersection only.")
    merge_cols = [col for col in merge_cols if col in results_df.columns and col in pred_df.columns]
    if merge_cols:
        combined_df = pd.merge(results_df, pred_df, on=merge_cols, how='outer')
    else:
        print("Warning: No common columns to merge on. Concatenating results.")
        combined_df = pd.concat([results_df, pred_df], axis=1)
else:
    combined_df = pd.merge(results_df, pred_df, on=merge_cols, how='outer')
script_base = os.path.splitext(os.path.basename(__file__))[0]
combined_path = os.path.join(csv_output_dir, f'{script_base}.csv')
combined_df.to_csv(combined_path, index=False)
print(f'Saved combined restock model metrics and predictions to {combined_path}')