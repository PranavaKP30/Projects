"""
Expected Demand Predictor for Each Product for the Next n Days

Usage:
    python expectedDemandPredictor.py [--predict_days N] [--sequence] [--visualize]

Options:
    --predict_days N   Number of days to predict in advance (default: 7)
    --sequence         Predict daily demand as a sequence (multi-output regression)
    --visualize        Generate and save PNG plots of historical and predicted demand for each product

Outputs:
    - Trained model files for each product (.joblib) in dataSalesOutput/
    - Combined CSV of model metrics and predictions in dataSalesOutput/
    - (Optional) PNG plots for each product in dataSalesOutput/ if --visualize is used

Example:
    python expectedDemandPredictor.py --predict_days 7 --sequence --visualize
    # Predicts daily demand for the next 7 days for each product and saves plots
"""

import pandas as pd
import numpy as np
import os
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
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
products = pd.read_csv(products_path, header=None)
inventory = pd.read_csv(inventory_path)

# Standardize columns (same as previous model)
prod_rename_dict = {}
if 'product_id' in products.columns:
    prod_rename_dict['product_id'] = 'ProductID'
if 'product_name' in products.columns:
    prod_rename_dict['product_name'] = 'ProductName'
products = products.rename(columns=prod_rename_dict)
if products.shape[1] > 1 and products.columns[0] != 'ProductID':
    product_colnames = ['ProductID', 'ProductName', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Col12', 'Col13', 'Col14', 'Col15', 'Col16', 'Col17', 'Col18', 'Col19', 'Col20']
    if products.shape[1] > len(product_colnames):
        product_colnames += [f'Col{i}' for i in range(21, products.shape[1]+1)]
    products.columns = product_colnames[:products.shape[1]]
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

def create_features(df, inventory, products):
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
    df = df.merge(products[['ProductID', 'ProductName', 'Col3', 'Col4', 'Col5']], on='ProductID', how='left')
    inventory_daily = inventory.groupby(['ProductID', 'Date'])['Quantity'].sum().reset_index()
    df = df.merge(inventory_daily.rename(columns={'Date':'OrderDate', 'Quantity':'stock_on_hand'}), on=['ProductID','OrderDate'], how='left')
    if 'Promotion' in df.columns:
        df['promotion_flag'] = df['Promotion'].fillna(0)
    else:
        df['promotion_flag'] = 0
    df = df.fillna(0)
    return df

sales['ProductID'] = sales['ProductID'].astype(str)
products['ProductID'] = products['ProductID'].astype(str)
inventory['ProductID'] = inventory['ProductID'].astype(str)
sales = sales[sales['OrderDate'].notna()]
sales = sales.sort_values(['ProductID', 'OrderDate'])

# Argument parsing
parser = argparse.ArgumentParser(description='Expected demand prediction for next n days')
parser.add_argument('--predict_days', type=int, default=7, help='Number of days to predict in advance (default: 7)')
parser.add_argument('--sequence', action='store_true', help='Predict daily demand as a sequence (multi-output regression)')
parser.add_argument('--visualize', action='store_true', help='Visualize sales and predictions for each product')
args = parser.parse_args()
predict_days = args.predict_days
sequence = args.sequence
visualize = args.visualize



# Only the CSV goes to /Users/pranav/Coding/Projects/ThataRetail/Output
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'Output')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
csv_output_dir = OUTPUT_ROOT

# Other files (models, plots) go in expectedDemandOutput as before
model_plot_dir = os.path.join(BASE_DIR, 'expectedDemandOutput')
os.makedirs(model_plot_dir, exist_ok=True)


results = []
future_predictions = []

for pid, group in sales.groupby('ProductID'):
    group = group.sort_values('OrderDate')
    if len(group) < 60:
        continue
    df_feat = create_features(group, inventory, products)
    feature_cols = ['dayofweek','month','is_weekend','lag_1','lag_2','lag_7','roll_7','roll_30','stock_on_hand','promotion_flag']
    # Build targets
    if sequence:
        # Multi-output: predict sales for each of the next n days
        targets = []
        for i in range(predict_days):
            targets.append(df_feat.groupby('ProductID')['Quantity'].shift(-i-1))
        y = np.stack([t.values for t in targets], axis=1)
        mask = ~np.any(np.isnan(y), axis=1)
        X = df_feat.loc[mask, feature_cols]
        y = y[mask]
        # Chronological split
        train_X, valid_X = X.iloc[:-30], X.iloc[-30:]
        train_y, valid_y = y[:-30], y[-30:]
        model = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42))
        model.fit(train_X, train_y)
        y_pred = model.predict(valid_X)
        mae = mean_absolute_error(valid_y, y_pred)
        rmse = np.sqrt(mean_squared_error(valid_y, y_pred))
        results.append({'ProductID': pid, 'MAE': mae, 'RMSE': rmse})
        # Predict next n days
        last_feat = df_feat.iloc[[-1]][feature_cols]
        next_pred = model.predict(last_feat)[0]
        for i, val in enumerate(next_pred):
            future_predictions.append({'ProductID': pid, 'Day': i+1, 'PredictedSales': max(0, val)})
        # Save model
        joblib.dump(model, os.path.join(model_plot_dir, f'model_{pid}_seq.joblib'))
    else:
        # Single-output: predict sum of next n days
        y = df_feat.groupby('ProductID')['Quantity'].shift(-1).rolling(predict_days).sum().reset_index(0,drop=True)
        mask = ~y.isna()
        X = df_feat.loc[mask, feature_cols]
        y = y[mask]
        train_X, valid_X = X.iloc[:-30], X.iloc[-30:]
        train_y, valid_y = y.iloc[:-30], y.iloc[-30:]
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
        model.fit(train_X, train_y)
        y_pred = model.predict(valid_X)
        mae = mean_absolute_error(valid_y, y_pred)
        rmse = np.sqrt(mean_squared_error(valid_y, y_pred))
        results.append({'ProductID': pid, 'MAE': mae, 'RMSE': rmse})
        # Predict next n days (sum)
        last_feat = df_feat.iloc[[-1]][feature_cols]
        next_pred = model.predict(last_feat)[0]
        future_predictions.append({'ProductID': pid, 'Days': predict_days, 'PredictedTotalSales': max(0, next_pred)})
        # Save model
        joblib.dump(model, os.path.join(model_plot_dir, f'model_{pid}_sum.joblib'))
    # Visualization (optional)
    if visualize:
        plt.figure(figsize=(12,6))
        plt.plot(group['OrderDate'], group['Quantity'], label='Historical Sales', marker='o')
        plt.title(f'Product {pid}')
        plt.xlabel('Date')
        plt.ylabel('Units Sold')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(model_plot_dir, f'product_{pid}_expected_demand.png')
        plt.savefig(plot_path)
        plt.close()


# Save all output into a single CSV file
results_df = pd.DataFrame(results)
pred_df = pd.DataFrame(future_predictions)
if not pred_df.empty:
    combined_df = pd.merge(results_df, pred_df, on='ProductID', how='outer')
else:
    combined_df = results_df.copy()


# Add ProductName to combined_df if available
if 'ProductName' in products.columns:
    # Robustly detect category column
    category_col = None
    for col in products.columns:
        col_lower = str(col).lower()
        if ('category' in col_lower) or (col_lower in ['cat', 'type']):
            category_col = col
            break
    # If not found, try to infer from Col3/Col4/etc. if they look categorical
    if not category_col:
        possible_cats = [c for c in products.columns if c.startswith('Col')]
        for c in possible_cats:
            # Heuristic: if >1 unique value and not numeric, treat as category
            vals = products[c].dropna().unique()
            if len(vals) > 1 and not pd.api.types.is_numeric_dtype(products[c]):
                category_col = c
                break
    merge_cols = ['ProductID', 'ProductName']
    if category_col:
        merge_cols.append(category_col)
        combined_df = combined_df.merge(products[merge_cols], on='ProductID', how='left')
        # Move ProductName and Category to be right after ProductID for readability
        cols = list(combined_df.columns)
        if 'ProductName' in cols:
            cols.insert(1, cols.pop(cols.index('ProductName')))
        if category_col and category_col in cols:
            # Standardize output column name to 'Category'
            combined_df = combined_df.rename(columns={category_col: 'Category'})
            cols = [c if c != category_col else 'Category' for c in cols]
            cols.insert(2, cols.pop(cols.index('Category')))
        combined_df = combined_df[cols]
    else:
        # No category found, add blank 'Category' column after ProductName
        combined_df = combined_df.merge(products[['ProductID', 'ProductName']], on='ProductID', how='left')
        cols = list(combined_df.columns)
        if 'ProductName' in cols:
            cols.insert(1, cols.pop(cols.index('ProductName')))
        # Insert blank Category column after ProductName
        idx = cols.index('ProductName') + 1
        cols.insert(idx, 'Category')
        combined_df = combined_df.reindex(columns=cols)
        combined_df['Category'] = ''
        combined_df = combined_df[cols]

# Save only the CSV to /Users/pranav/Coding/Projects/ThataRetail/Output
script_base = os.path.splitext(os.path.basename(__file__))[0]
combined_path = os.path.join(csv_output_dir, f'{script_base}.csv')
combined_df.to_csv(combined_path, index=False)
print(f'Saved combined model metrics and predictions to {combined_path}')
