"""
Stockout Prediction Model: Predict which products are likely to go out of stock in the near future

Usage:
    python stockoutPredictionModel.py [--predict_days N] [--model MODEL] [--visualize]

Options:
    --predict_days N   Number of days to look ahead for stockout event (default: 7)
    --model MODEL      Classification model to use: xgboost (default), randomforest, logistic
    --visualize        Generate and save PNG plots of actual vs. predicted stockouts for each product

Outputs:
    - Trained classification model files for each product (.joblib) in stockoutOutput/
    - Combined CSV of predictions and metrics in stockoutOutput/
    - (Optional) PNG plots for each product in stockoutOutput/ if --visualize is used

Example:
    python stockoutPredictionModel.py --predict_days 7 --model xgboost --visualize
    # Predicts stockouts for each product and saves plots
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
if 'shelf_life' in products.columns:
    prod_rename_dict['shelf_life'] = 'ShelfLife'
if 'shelf_life_days' in products.columns:
    prod_rename_dict['shelf_life_days'] = 'ShelfLife'
if 'min_order_qty' in products.columns:
    prod_rename_dict['min_order_qty'] = 'MinOrderQty'
if 'category' in products.columns:
    prod_rename_dict['category'] = 'Category'
products = products.rename(columns=prod_rename_dict)
if products.shape[1] > 1 and products.columns[0] != 'ProductID':
    product_colnames = ['ProductID', 'ProductName', 'ShelfLife', 'MinOrderQty', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Col12', 'Col13', 'Col14', 'Col15', 'Col16', 'Col17', 'Col18', 'Col19', 'Col20']
    if products.shape[1] > len(product_colnames):
        product_colnames += [f'Col{i}' for i in range(21, products.shape[1]+1)]
    products.columns = product_colnames[:products.shape[1]]
if 'ShelfLife' in products.columns:
    products['ShelfLife'] = pd.to_numeric(products['ShelfLife'], errors='coerce').fillna(0)
if 'MinOrderQty' in products.columns:
    products['MinOrderQty'] = pd.to_numeric(products['MinOrderQty'], errors='coerce').fillna(1)

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
sales = sales.rename(columns=rename_dict)
if 'OrderDate' in sales.columns:
    sales['OrderDate'] = pd.to_datetime(sales['OrderDate'], errors='coerce')
else:
    raise ValueError("sales.csv must contain a 'sale_date' or 'OrderDate' column. Please check your data file.")

# Merge product features into inventory


# Merge product features into inventory
data = inventory.merge(products, on='ProductID', how='left')

# Ensure Date columns are datetime before merging sales
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
sales['OrderDate'] = pd.to_datetime(sales['OrderDate'], errors='coerce')


# --- Compute running stock for each product ---
# Prepare sales per product per date

sales_daily = sales.groupby(['ProductID', 'OrderDate'])['Quantity'].sum().reset_index()
sales_daily = sales_daily.rename(columns={'OrderDate': 'Date', 'Quantity': 'SalesQty'})
sales_daily['Date'] = pd.to_datetime(sales_daily['Date'], errors='coerce')

# Merge sales into inventory data (so each inventory row has sales for that day)
data = data.merge(sales_daily, on=['ProductID', 'Date'], how='left')
data['SalesQty'] = data['SalesQty'].fillna(0)

# Compute running stock for each product
data = data.sort_values(['ProductID', 'Date'])
data['RunningStock'] = np.nan
for pid, group in data.groupby('ProductID'):
    group = group.sort_values('Date').copy()
    # Use the first available Quantity as initial stock
    initial_stock = group['Quantity'].iloc[0]
    running_stock = [initial_stock]
    for i in range(1, len(group)):
        prev_stock = running_stock[-1]
        sales_today = group['SalesQty'].iloc[i]
        # If you have restock/adjustment columns, add them here
        restock = 0
        running_stock.append(prev_stock - sales_today + restock)
    data.loc[group.index, 'RunningStock'] = running_stock


# Ensure Date is datetime

# Ensure Date is datetime and drop rows with invalid dates
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    invalid_dates = data['Date'].isna().sum()
    if invalid_dates > 0:
        print(f"Warning: Dropping {invalid_dates} rows with invalid Date values from inventory data.")
        data = data.dropna(subset=['Date'])

# Feature engineering
data['dayofweek'] = data['Date'].dt.dayofweek
data['month'] = data['Date'].dt.month
# Lagged/rolling sales
sales['OrderDate'] = pd.to_datetime(sales['OrderDate'], errors='coerce')
for lag in [1, 7]:
    sales[f'lag_{lag}'] = sales.groupby('ProductID')['Quantity'].shift(lag)
for window in [7, 30]:
    sales[f'roll_{window}'] = sales.groupby('ProductID')['Quantity'].shift(1).rolling(window).mean().reset_index(0,drop=True)
# Merge recent sales features into inventory data
sales_daily = sales.groupby(['ProductID', 'OrderDate'])[['Quantity', 'lag_1', 'lag_7', 'roll_7', 'roll_30']].sum().reset_index()
data = data.merge(sales_daily.rename(columns={'OrderDate':'Date'}), on=['ProductID','Date'], how='left', suffixes=('', '_sales'))
# Fill missing
for col in ['Quantity', 'lag_1', 'lag_7', 'roll_7', 'roll_30']:
    if col in data.columns:
        data[col] = data[col].fillna(0)

parser = argparse.ArgumentParser(description='Stockout prediction for each product')
parser.add_argument('--predict_days', type=int, default=7, help='Number of days to look ahead for stockout event (default: 7)')
parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'randomforest', 'logistic'], help='Classification model to use')
parser.add_argument('--visualize', action='store_true', help='Visualize actual vs. predicted stockouts for each product')
args = parser.parse_args()
predict_days = args.predict_days
model_type = args.model
visualize = args.visualize


# Only the CSV goes to /Users/pranav/Coding/Projects/ThataRetail/Output
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'Output')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
csv_output_dir = OUTPUT_ROOT

# Other files (models, plots) go in stockoutOutput as before
model_plot_dir = os.path.join(BASE_DIR, 'stockoutOutput')
os.makedirs(model_plot_dir, exist_ok=True)

results = []
all_predictions = []

for pid, group in data.groupby('ProductID'):
    group = group.sort_values('Date')
    if len(group) < 30:
        print(f"Skipping ProductID {pid}: not enough data points ({len(group)})")
        continue
    # Target: stockout in next n days, using RunningStock
    group = group.copy()
    group['future_stock'] = group['RunningStock'].shift(-predict_days)
    group['stockout'] = (group['future_stock'] <= 0).astype(int)
    # Ensure MinOrderQty exists
    if 'MinOrderQty' not in group.columns:
        group['MinOrderQty'] = 1
    feature_cols = ['RunningStock', 'lag_1', 'lag_7', 'roll_7', 'roll_30', 'ShelfLife', 'MinOrderQty', 'dayofweek', 'month']
    X = group[feature_cols]
    y = group['stockout']
    # Chronological split
    train_X, valid_X = X.iloc[:-10], X.iloc[-10:]
    train_y, valid_y = y.iloc[:-10], y.iloc[-10:]
    # Debug info
    print(f"ProductID {pid}: n_rows={len(group)}, train_y unique={train_y.unique()}, valid_y unique={valid_y.unique()}, future_stock min={group['future_stock'].min()}, max={group['future_stock'].max()}, stockout unique={group['stockout'].unique()}")
    # Ensure binary target has both classes
    if train_y.nunique() < 2:
        print(f"Skipping ProductID {pid}: training target does not contain both classes (0 and 1).")
        continue
    # Model selection
    if model_type == 'xgboost':
        model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    elif model_type == 'randomforest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
    model.fit(train_X, train_y)
    pred = model.predict(valid_X)
    acc = accuracy_score(valid_y, pred)
    prec = precision_score(valid_y, pred, zero_division=0)
    rec = recall_score(valid_y, pred, zero_division=0)
    f1 = f1_score(valid_y, pred, zero_division=0)
    results.append({'ProductID': pid, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1})
    # Save detailed predictions for this product
    pred_product_ids = [pid] * 10
    pred_df = pd.DataFrame({
        'ProductID': pred_product_ids,
        'ProductName': group['ProductName'].iloc[-10:].values if 'ProductName' in group.columns else [np.nan]*10,
        'Date': group['Date'].iloc[-10:].values,
        'ActualStockout': valid_y.values,
        'PredictedStockout': pred,
        'RunningStock': group['RunningStock'].iloc[-10:].values
    })
    all_predictions.append(pred_df)
    # Save model
    joblib.dump(model, os.path.join(model_plot_dir, f'stockout_model_{pid}.joblib'))
    # Visualization (optional)
    if visualize:
        plt.figure(figsize=(10,6))
        plt.plot(group['Date'].iloc[-10:], valid_y, label='Actual', marker='o')
        plt.plot(group['Date'].iloc[-10:], pred, label='Predicted', marker='x')
        plt.title(f'Product {pid} Stockout Prediction')
        plt.xlabel('Date')
        plt.ylabel('Stockout (1=Yes, 0=No)')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(model_plot_dir, f'product_{pid}_stockout.png')
        plt.savefig(plot_path)
        plt.close()



# Save all output into a single CSV file (main output to Output folder)

# Save detailed predictions for all products (to Output folder as main CSV)
if all_predictions:
    all_pred_df = pd.concat(all_predictions, ignore_index=True)
    # Add Category from products.csv if available, ensuring type consistency
    if 'ProductID' in all_pred_df.columns and 'Category' in products.columns:
        products['ProductID'] = products['ProductID'].astype(str)
        all_pred_df['ProductID'] = all_pred_df['ProductID'].astype(str)
        cat_map = products.set_index('ProductID')['Category']
        all_pred_df['Category'] = all_pred_df['ProductID'].map(lambda x: cat_map.get(x, np.nan))
        print('DEBUG: First 10 mapped categories:', all_pred_df['Category'].head(10).tolist())
    else:
        print('DEBUG: Category column missing in products.csv or ProductID missing in output.')
    # Reorder columns: ProductID, ProductName, Category, Date, ActualStockout, PredictedStockout, RunningStock
    cols = [c for c in ['ProductID', 'ProductName', 'Category', 'Date', 'ActualStockout', 'PredictedStockout', 'RunningStock'] if c in all_pred_df.columns]
    other_cols = [c for c in all_pred_df.columns if c not in cols]
    all_pred_df = all_pred_df[cols + other_cols]
    script_base = os.path.splitext(os.path.basename(__file__))[0]
    combined_path = os.path.join(csv_output_dir, f'{script_base}.csv')
    all_pred_df.to_csv(combined_path, index=False)
    print(f'Saved stockout prediction results to {combined_path}')
    # Save detailed predictions for all products (to per-model folder)
    detailed_path = os.path.join(model_plot_dir, 'stockout_predictions_detailed.csv')
    all_pred_df.to_csv(detailed_path, index=False)
    print(f'Saved detailed stockout predictions to {detailed_path}')
else:
    # Fallback: save summary metrics only
    results_df = pd.DataFrame(results)
    if 'ProductName' not in results_df.columns and 'ProductID' in results_df.columns:
        results_df = results_df.merge(products[['ProductID', 'ProductName']], on='ProductID', how='left')
    cols = list(results_df.columns)
    if 'ProductID' in cols and 'ProductName' in cols:
        pid_idx = cols.index('ProductID')
        pname_idx = cols.index('ProductName')
        if pname_idx != pid_idx + 1:
            cols.remove('ProductName')
            cols.insert(pid_idx + 1, 'ProductName')
        results_df = results_df[cols]
    script_base = os.path.splitext(os.path.basename(__file__))[0]
    combined_path = os.path.join(csv_output_dir, f'{script_base}.csv')
    results_df.to_csv(combined_path, index=False)
    print(f'Saved stockout prediction results to {combined_path}')

# Save detailed predictions for all products (to per-model folder)
if all_predictions:
    all_pred_df = pd.concat(all_predictions, ignore_index=True)
    detailed_path = os.path.join(model_plot_dir, 'stockout_predictions_detailed.csv')
    all_pred_df.to_csv(detailed_path, index=False)
    print(f'Saved detailed stockout predictions to {detailed_path}')
