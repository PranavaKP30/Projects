"""
Daily Sales Prediction Script

Usage:
    python dailySalesPrediction.py [--predict_days N] [--visualize]

Options:
    --predict_days N   Number of days to predict in advance (default: 1)
    --visualize        Generate and save PNG plots of historical and predicted sales for each product

Outputs:
    - Trained model files for each product (.joblib) in dataSalesOutput/
    - Combined CSV of model metrics and predictions in dataSalesOutput/
    - (Optional) PNG plots for each product in dataSalesOutput/ if --visualize is used

Example:
    python dailySalesPrediction.py --predict_days 7 --visualize
    # Predicts 7 days ahead and saves plots for each product
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import joblib
# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns


# Always use canonical Data folder for training/prediction files
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'Data')
sales_path = os.path.join(DATA_DIR, 'sales.csv')
products_path = os.path.join(DATA_DIR, 'products.csv')
inventory_path = os.path.join(DATA_DIR, 'inventory.csv')

# Load data
sales = pd.read_csv(sales_path)
products = pd.read_csv(products_path, header=None)
inventory = pd.read_csv(inventory_path)


# Standardize products column names to match code expectations
prod_rename_dict = {}
if 'product_id' in products.columns:
    prod_rename_dict['product_id'] = 'ProductID'
if 'product_name' in products.columns:
    prod_rename_dict['product_name'] = 'ProductName'
products = products.rename(columns=prod_rename_dict)

# If products file has no header, assign default column names
if products.shape[1] > 1 and products.columns[0] != 'ProductID':
    product_colnames = ['ProductID', 'ProductName', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9', 'Col10', 'Col11', 'Col12', 'Col13', 'Col14', 'Col15', 'Col16', 'Col17', 'Col18', 'Col19', 'Col20']
    if products.shape[1] > len(product_colnames):
        product_colnames += [f'Col{i}' for i in range(21, products.shape[1]+1)]
    products.columns = product_colnames[:products.shape[1]]


# Standardize sales column names to match code expectations
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

# Convert Promotion column to numeric (1 for Yes/True, 0 for No/False/NA)
if 'Promotion' in sales.columns:
    sales['Promotion'] = sales['Promotion'].map(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)

if 'OrderDate' in sales.columns:
    sales['OrderDate'] = pd.to_datetime(sales['OrderDate'], errors='coerce')
else:
    raise ValueError("sales.csv must contain a 'sale_date' or 'OrderDate' column. Please check your data file.")


# Standardize inventory column names to match code expectations
inv_rename_dict = {}
if 'product_id' in inventory.columns:
    inv_rename_dict['product_id'] = 'ProductID'
if 'inventory_date' in inventory.columns:
    inv_rename_dict['inventory_date'] = 'Date'
inventory = inventory.rename(columns=inv_rename_dict)

# Use stock_on_hand_eod as Quantity for inventory merging
if 'stock_on_hand_eod' in inventory.columns:
    inventory = inventory.rename(columns={'stock_on_hand_eod': 'Quantity'})

# Inventory date
if 'Date' in inventory.columns:
    inventory['Date'] = pd.to_datetime(inventory['Date'], errors='coerce')
elif 'AdjustmentDate' in inventory.columns:
    inventory['Date'] = pd.to_datetime(inventory['AdjustmentDate'], errors='coerce')
else:
    raise ValueError(f"inventory.csv must contain a 'Date' or 'AdjustmentDate' column. Columns found: {list(inventory.columns)}")

# Feature engineering
def create_features(df, inventory, products):
    df = df.copy()
    df['dayofweek'] = df['OrderDate'].dt.dayofweek
    df['month'] = df['OrderDate'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    # Lag features
    df = df.sort_values(['ProductID', 'OrderDate'])
    df['lag_1'] = df.groupby('ProductID')['Quantity'].shift(1)
    df['lag_2'] = df.groupby('ProductID')['Quantity'].shift(2)
    df['lag_7'] = df.groupby('ProductID')['Quantity'].shift(7)
    # Rolling mean
    df['roll_7'] = df.groupby('ProductID')['Quantity'].shift(1).rolling(7).mean().reset_index(0,drop=True)
    df['roll_30'] = df.groupby('ProductID')['Quantity'].shift(1).rolling(30).mean().reset_index(0,drop=True)
    # Merge product features
    df = df.merge(products[['ProductID', 'ProductName', 'Col3', 'Col4', 'Col5']], on='ProductID', how='left')
    # Merge inventory (stock_on_hand)
    inventory_daily = inventory.groupby(['ProductID', 'Date'])['Quantity'].sum().reset_index()
    df = df.merge(inventory_daily.rename(columns={'Date':'OrderDate', 'Quantity':'stock_on_hand'}), on=['ProductID','OrderDate'], how='left')
    # Promotion flag (if present)
    if 'Promotion' in df.columns:
        df['promotion_flag'] = df['Promotion'].fillna(0)
    else:
        df['promotion_flag'] = 0
    # Fill missing
    df = df.fillna(0)
    return df

# Prepare data
sales['ProductID'] = sales['ProductID'].astype(str)
products['ProductID'] = products['ProductID'].astype(str)
inventory['ProductID'] = inventory['ProductID'].astype(str)
sales = sales[sales['OrderDate'].notna()]
sales = sales.sort_values(['ProductID', 'OrderDate'])

# For each product, train/test split and model
results = []
models = {}



# Allow user to specify number of days to predict in advance and whether to visualize
import argparse
parser = argparse.ArgumentParser(description='Daily sales prediction for N days ahead')

parser.add_argument('--predict_days', type=int, default=1, help='Number of days to predict in advance (default: 1)')


parser.add_argument('--visualize', action='store_true', help='Visualize sales and predictions for each product')
args = parser.parse_args()
predict_days = args.predict_days
visualize = args.visualize



## Output files are saved to canonical Data folder
## Remove undefined BASE_DIR usage

# Other files (models, plots) go in dataSalesOutput as before
model_plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataSalesOutput')
os.makedirs(model_plot_dir, exist_ok=True)

future_day_predictions = []



for pid, group in sales.groupby('ProductID'):
    group = group.sort_values('OrderDate')
    # Only use products with enough data (lower threshold to 1 for prediction)
    if len(group) < 1:
        continue
    df_feat = create_features(group, inventory, products)
    # Target: next day's sales
    df_feat['target'] = df_feat.groupby('ProductID')['Quantity'].shift(-1)
    df_feat = df_feat.dropna(subset=['target'])
    # Chronological split: last 30 days for validation
    train = df_feat.iloc[:-30]
    valid = df_feat.iloc[-30:]
    feature_cols = ['dayofweek','month','is_weekend','lag_1','lag_2','lag_7','roll_7','roll_30','stock_on_hand','promotion_flag']
    X_train = train[feature_cols]
    y_train = train['target']
    X_valid = valid[feature_cols]
    y_valid = valid['target']
    # Train model
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    # Validation
    y_pred = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, y_pred)
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    results.append({'ProductID': pid, 'MAE': mae, 'RMSE': rmse})

    # Predict sales for today and for all days in the current month
    today = datetime.today().date()
    first_of_month = today.replace(day=1)
    last_group = group.copy()
    preds = []
    # Predict for each day in current month (from 1st to today)
    for i, pred_date in enumerate(pd.date_range(first_of_month, today)):
        next_row = pd.DataFrame({
            'ProductID': [pid],
            'OrderDate': [pd.Timestamp(pred_date)],
            'Quantity': [np.nan],
            'Promotion': [0]
        })
        feat_input = pd.concat([last_group, next_row], ignore_index=True)
        next_feat = create_features(feat_input, inventory, products)
        next_feat = next_feat.iloc[[-1]]
        X_next = next_feat[feature_cols]
        next_pred = model.predict(X_next)[0]
        next_pred = max(0, next_pred)
        preds.append({'ProductID': pid, 'Day': i+1, 'Date': pd.Timestamp(pred_date).strftime('%Y-%m-%d'), 'PredictedSales': next_pred})
        # For next iteration, append the predicted value as if it was observed
        next_row['Quantity'] = next_pred
        last_group = pd.concat([last_group, next_row], ignore_index=True)

    # --- Predict three sales values for each month prior to the current month ---
    last_sale_date = group['OrderDate'].max().date()
    # Get all months between last_sale_date (exclusive) and the start of the current month (exclusive)
    first_of_current_month = today.replace(day=1)
    month_iter = pd.date_range(last_sale_date + pd.offsets.MonthBegin(1), first_of_current_month, freq='MS')
    for month_start in month_iter:
        # Predict for three dates in this month: 1st, 10th, 20th (or closest valid day)
        for day in [1, 10, 20]:
            try:
                pred_date = month_start.replace(day=day).date()
            except ValueError:
                # If day is out of range for month, use last day of month
                pred_date = (month_start + pd.offsets.MonthEnd(0)).date()
            next_row = pd.DataFrame({
                'ProductID': [pid],
                'OrderDate': [pd.Timestamp(pred_date)],
                'Quantity': [np.nan],
                'Promotion': [0]
            })
            feat_input = pd.concat([last_group, next_row], ignore_index=True)
            next_feat = create_features(feat_input, inventory, products)
            next_feat = next_feat.iloc[[-1]]
            X_next = next_feat[feature_cols]
            next_pred = model.predict(X_next)[0]
            next_pred = max(0, next_pred)
            preds.append({'ProductID': pid, 'Day': None, 'Date': pd.Timestamp(pred_date).strftime('%Y-%m-%d'), 'PredictedSales': next_pred})
            # For next iteration, append the predicted value as if it was observed
            next_row['Quantity'] = next_pred
            last_group = pd.concat([last_group, next_row], ignore_index=True)

    # Save all predictions for this product
    for p in preds:
        future_day_predictions.append(p)

    # Save model
    models[pid] = model
    # Save model in dataSalesOutput as before
    model_path = os.path.join(model_plot_dir, f'model_{pid}.joblib')
    joblib.dump(model, model_path)
    print(f'Product {pid}: MAE={mae:.2f}, RMSE={rmse:.2f}, PredictedSales={next_pred}')

    # Visualization (if requested)
    if visualize:
        plt.figure(figsize=(12,6))
        # Plot historical sales
        plt.plot(group['OrderDate'], group['Quantity'], label='Historical Sales', marker='o')
        # Plot predictions for each day in current month
        if preds:
            pred_dates = [pd.to_datetime(p['Date']) for p in preds]
            pred_sales = [p['PredictedSales'] for p in preds]
            plt.plot(pred_dates, pred_sales, label='Predicted Sales (Current Month)', marker='x', linestyle='--', color='red')
        # Title and labels
        pname = products[products['ProductID'] == pid]['ProductName'].values[0] if 'ProductName' in products.columns else pid
        plt.title(f'Product {pid} - {pname}')
        plt.xlabel('Date')
        plt.ylabel('Units Sold')
        plt.legend()
        plt.tight_layout()
        # Save plot to dataSalesOutput as before
        plot_path = os.path.join(model_plot_dir, f'product_{pid}_forecast.png')
        plt.savefig(plot_path)
        print(f'Visualization saved to {plot_path}')
        plt.close()


# Save combined results
results_df = pd.DataFrame(results)
future_pred_df = pd.DataFrame(future_day_predictions)

# Pivot predictions so each row is ProductID, ProductName, MAE, RMSE, and then columns for each day ahead
if not future_pred_df.empty:
    # Remove duplicate ProductID-Date pairs to avoid pivot error
    future_pred_df = future_pred_df.drop_duplicates(subset=['ProductID', 'Date'], keep='last')
    # Pivot so columns are prediction dates
    future_pred_df['Date'] = pd.to_datetime(future_pred_df['Date'])
    pivot_df = future_pred_df.pivot(index='ProductID', columns='Date', values='PredictedSales')
    # Convert date columns to string for CSV header
    pivot_df.columns = [d.strftime('%Y-%m-%d') for d in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    combined_df = results_df.merge(pivot_df, on='ProductID', how='inner')
else:
    combined_df = results_df.copy()


# Add ProductName from products DataFrame, robust to missing ProductID column
if 'ProductName' in products.columns:
    # Ensure 'ProductID' exists in both DataFrames
    if 'ProductID' in combined_df.columns and 'ProductID' in products.columns:
        combined_df = combined_df.merge(products[['ProductID', 'ProductName']], on='ProductID', how='left')
        # Move ProductName to be right after ProductID for readability
        cols = list(combined_df.columns)
        if 'ProductName' in cols:
            cols.insert(1, cols.pop(cols.index('ProductName')))
            combined_df = combined_df[cols]
    else:
        print('Warning: ProductID column missing in combined_df or products, skipping ProductName merge.')

script_base = os.path.splitext(os.path.basename(__file__))[0]
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Output')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'{script_base}.csv')
combined_df.to_csv(output_path, index=False)
print(f'Saved combined model metrics and next {predict_days} day(s) predictions to {output_path}')