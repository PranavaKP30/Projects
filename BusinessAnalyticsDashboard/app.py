"""
Retail Analytics ML Dashboard (Streamlit)

How to run:
    streamlit run app.py

This will launch the dashboard in your browser.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path

# --- Language Toggle ---
LANGS = {"English": "en", "தமிழ்": "ta"}
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'
lang_choice = st.sidebar.radio("Language / மொழி", list(LANGS.keys()), index=0)
st.session_state['lang'] = LANGS[lang_choice]
lang = st.session_state['lang']


# --- UI Strings ---
STRINGS = {
    'en': {
        'title': "Business Analytics ML Dashboard",
        'upload_header': "Upload Data Files",
        'products': "Products",
        'sales': "Sales",
        'inventory': "Inventory",
        'upload_products': "Upload products.csv",
        'upload_sales': "Upload sales.csv",
        'upload_inventory': "Upload inventory.csv",
        'data_preview': "Data Preview",
        'run': "Run",
        'download_csv': "Download CSV",
        'no_csv': "No output CSV found.",
        'info': "This is a Streamlit-based desktop dashboard for your business ML models. Extend each tab for custom visualizations and advanced controls!",
        'Daily Sales Prediction': "Daily Sales Prediction",
        'Expected Demand Predictor': "Expected Demand Predictor",
        'Restock Predictor': "Restock Predictor",
        'Price Optimization': "Price Optimization",
        'Promotion Effectiveness': "Promotion Effectiveness",
        'Stockout Prediction': "Stockout Prediction",
        'Inventory Turnover': "Inventory Turnover",
        'Expiry Risk Prediction': "Expiry Risk Prediction",
        'ABC Analysis': "ABC Analysis"
    },
    'ta': {
        'title': "வணிக பகுப்பாய்வு எம்.எல். டாஷ்போர்டு",
        'upload_header': "தரவுகள் பதிவேற்றவும்",
        'products': "பொருட்கள்",
        'sales': "விற்பனை",
        'inventory': "சரக்கு",
        'upload_products': "products.csv பதிவேற்றவும்",
        'upload_sales': "sales.csv பதிவேற்றவும்",
        'upload_inventory': "inventory.csv பதிவேற்றவும்",
        'data_preview': "தரவு முன்னோட்டம்",
        'run': "இயக்கு",
        'download_csv': "CSV பதிவிறக்கவும்",
        'no_csv': "CSV வெளியீடு இல்லை.",
        'info': "இது உங்கள் வணிக எம்.எல். மாதிரிகளுக்கான Streamlit அடிப்படையிலான டெஸ்க்டாப் டாஷ்போர்டு. ஒவ்வொரு தாவலிலும் விரிவான காட்சிப்படுத்தல்கள் மற்றும் மேம்பட்ட கட்டுப்பாடுகளைச் சேர்க்கவும்!",
        'Daily Sales Prediction': "தினசரி விற்பனை கணிப்பு",
        'Expected Demand Predictor': "எதிர்பார்க்கப்படும் தேவை கணிப்பு",
        'Restock Predictor': "மீண்டும் சரக்கு நிரப்பு கணிப்பு",
        'Price Optimization': "விலை மேம்படுத்தல்",
        'Promotion Effectiveness': "விளம்பர விளைவு",
        'Stockout Prediction': "சரக்கு முடிவு கணிப்பு",
        'Inventory Turnover': "சரக்கு சுழற்சி",
        'Expiry Risk Prediction': "காலாவதி ஆபத்து கணிப்பு",
        'ABC Analysis': "ABC பகுப்பு"
    }
}

st.set_page_config(page_title=STRINGS[lang]['title'], layout="wide")
st.title(STRINGS[lang]['title'])

# --- Sidebar: File Upload ---
st.sidebar.header(STRINGS[lang]['upload_header'])

def upload_and_cache(label, key):
    file = st.sidebar.file_uploader(label, type=["csv"], key=key)
    file_path = f"Data/{key}.csv"
    output_dir = "Output"
    def clear_output_files():
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                fpath = os.path.join(output_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass

    if file:
        df = pd.read_csv(file)
        st.session_state[key + '_df'] = df
        # Save to Data/ for model scripts
        Path("Data").mkdir(exist_ok=True)
        df.to_csv(file_path, index=False)
        st.session_state[key + '_path'] = file_path
        return df
    else:
        # If no file is uploaded, remove from session and delete from Data/
        st.session_state[key + '_df'] = None
        st.session_state[key + '_path'] = None
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        clear_output_files()
        return None

products = upload_and_cache(STRINGS[lang]['products'] + " CSV", "products")
sales = upload_and_cache(STRINGS[lang]['sales'] + " CSV", "sales")
inventory = upload_and_cache(STRINGS[lang]['inventory'] + " CSV", "inventory")

# --- Data Preview ---
st.header(STRINGS[lang]['data_preview'])
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader(STRINGS[lang]['products'])
    if products is not None:
        st.dataframe(products.head(20))
    else:
        st.info(STRINGS[lang]['upload_products'])
with col2:
    st.subheader(STRINGS[lang]['sales'])
    if sales is not None:
        st.dataframe(sales.head(20))
    else:
        st.info(STRINGS[lang]['upload_sales'])
with col3:
    st.subheader(STRINGS[lang]['inventory'])
    if inventory is not None:
        st.dataframe(inventory.head(20))
    else:
        st.info(STRINGS[lang]['upload_inventory'])

# --- Model Tabs ---
model_names = [
    'Daily Sales Prediction',
    'Expected Demand Predictor',
    'Restock Predictor',
    'Price Optimization',
    'Promotion Effectiveness',
    'Inventory Turnover'
]
tabs = st.tabs([STRINGS[lang][name] for name in model_names])

# --- Model Integration Example (Skeleton) ---
import plotly.express as px
import plotly.graph_objects as go

def run_model(model_script, output_dir=None, output_pngs=None):
    import subprocess
    import time
    # Run the model script (assumes CLI interface)
    st.info(f"{STRINGS[lang]['run']} {model_script}...")
    subprocess.run(["python", f"Models/{model_script}"], capture_output=True, text=True)
    # Wait for output
    time.sleep(1)
    # Output CSV is always in Output/ and named after the script
    script_base = os.path.splitext(model_script)[0]
    output_csv = f"{script_base}.csv"
    csv_path = os.path.join("Output", output_csv)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        st.dataframe(df)
        st.download_button(STRINGS[lang]['download_csv'], df.to_csv(index=False), file_name=output_csv)
        # --- Dynamic Visualizations ---
        # Show summary stats if available
        st.subheader("Key Metrics & Predictions")
        # Try to show metrics and predictions for common columns
        metric_cols = [c for c in df.columns if any(x in c.lower() for x in ["mae","rmse","accuracy","f1","precision","recall","expectedrevenue","optimalprice","abc_class","expiry","stockout","alert"])]
        if metric_cols:
            st.write(df[metric_cols].describe(include='all').T)
        # Show time series or bar charts for predictions
        pred_cols = [c for c in df.columns if "predicted" in c.lower() or "forecast" in c.lower() or "expected" in c.lower()]
        date_cols = [c for c in df.columns if "date" in c.lower()]
        if pred_cols and date_cols:
            for pred_col in pred_cols:
                for date_col in date_cols:
                    try:
                        fig = px.line(df, x=date_col, y=pred_col, color='ProductName' if 'ProductName' in df.columns else 'ProductID', title=f"{pred_col.replace('_', ' ')} over {date_col.replace('_', ' ')}")
                        fig.update_layout(xaxis_title="Date", yaxis_title=pred_col.replace('_', ' '), legend_title="Product")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
        elif pred_cols:
            for pred_col in pred_cols:
                try:
                    fig = px.bar(df, x='ProductName' if 'ProductName' in df.columns else 'ProductID', y=pred_col, title=f"{pred_col.replace('_', ' ')} by Product")
                    fig.update_layout(xaxis_title="Product", yaxis_title=pred_col.replace('_', ' '))
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
        # ABC Analysis pie chart
        if 'ABC_Class' in df.columns:
            fig = px.pie(df, names='ABC_Class', title='ABC Class Distribution')
            fig.update_layout(legend_title="Class")
            st.plotly_chart(fig, use_container_width=True)
        # Inventory Turnover class bar
        if 'TurnoverClass' in df.columns:
            fig = px.bar(df, x='ProductName' if 'ProductName' in df.columns else 'ProductID', y='TurnoverRatio', color='TurnoverClass', title='Inventory Turnover by Product')
            fig.update_layout(xaxis_title="Product", yaxis_title="Turnover Ratio", legend_title="Class")
            st.plotly_chart(fig, use_container_width=True)
        # Expiry risk pie
        if 'PredictedExpiryRisk' in df.columns:
            fig = px.pie(df, names='PredictedExpiryRisk', title='Predicted Expiry Risk Distribution')
            fig.update_layout(legend_title="Risk Level")
            st.plotly_chart(fig, use_container_width=True)
        # Stockout pie
        if 'PredictedStockout' in df.columns:
            fig = px.pie(df, names='PredictedStockout', title='Predicted Stockout Distribution')
            fig.update_layout(legend_title="Stockout")
            st.plotly_chart(fig, use_container_width=True)
        # Restock alert pie
        if 'NextDayAlert' in df.columns:
            fig = px.pie(df, names='NextDayAlert', title='Next Day Restock Alert Distribution')
            fig.update_layout(legend_title="Alert")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(STRINGS[lang]['no_csv'])
    # Show PNGs if any (from per-model output dir)
    if output_pngs and output_dir:
        for png in output_pngs:
            png_path = os.path.join(output_dir, png)
            if os.path.exists(png_path):
                st.image(png_path)

# --- Example for first model ---

with tabs[0]:
    st.header(STRINGS[lang]['Daily Sales Prediction'])
    # Tamil localization for all UI, table, and graph labels
    def t(s):
        return {
            "Product ID": "பொருள் ஐடி",
            "Product Name": "பொருள் பெயர்",
            "Predicted Sales": "முன்னறிவிக்கப்பட்ட விற்பனை",
            "Top 3 Selling Products for Today": "இன்று சிறந்த 3 விற்பனை பொருட்கள்",
            "Today's Product Details & Forecast": "இன்றைய பொருள் விவரங்கள் மற்றும் முன்னறிவிப்பு",
            "Search Product Name": "பொருள் பெயரைத் தேடவும்",
            "No match for search.": "தேடலில் பொருந்தவில்லை.",
            "Units Sold": "விற்பனை அலகுகள்",
            "Date": "தேதி",
            "Type": "வகை",
            "All Predicted Sales for Today": "இன்றைய அனைத்து முன்னறிவிக்கப்பட்ட விற்பனைகள்",
            "Predicted Date": "முன்னறிவிக்கப்பட்ட தேதி",
            "No predictions found for today.": "இன்றைக்கு முன்னறிவிப்புகள் இல்லை."
        }.get(s, s) if lang == 'ta' else s
    if st.button(f"{STRINGS[lang]['run']} {STRINGS[lang]['Daily Sales Prediction']}"):
        import subprocess
        import time
        st.info(f"{STRINGS[lang]['run']} dailySalesPrediction.py ...")
        subprocess.run(["python", "Models/dailySalesPrediction.py"], capture_output=True, text=True)
        time.sleep(1)
        csv_path = os.path.join("Output", "dailySalesPrediction.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # --- New logic: Identify date columns (actual dates) and prediction columns ---
            # Date columns: columns matching YYYY-MM-DD format
            import re
            date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            date_cols = [c for c in df.columns if date_pattern.match(str(c))]
            # Prediction columns: same as date columns (each date column is predicted sales for that date)
            # Show columns: ProductID, ProductName, all date columns
            show_cols = ["ProductID", "ProductName"] + date_cols
            # --- Key Metrics: Top 3 selling products for today ---
            st.subheader(t("Top 3 Selling Products for Today"))
            if date_cols:
                # Find the most recent date column
                date_objs = [pd.to_datetime(dc, errors='coerce') for dc in date_cols]
                latest_date_idx = date_objs.index(max(date_objs))
                latest_date_col = date_cols[latest_date_idx]
                # Top 3 products by predicted sales for latest date
                top3 = df.sort_values(latest_date_col, ascending=False).head(3)
            st.table(top3[["ProductID", "ProductName", latest_date_col]].rename(columns={
                "ProductID": t("Product ID"),
                "ProductName": t("Product Name"),
                latest_date_col: t("Predicted Sales"),
            }))

            # --- Today's Product Details & Forecast: Search Bar and Tabs ---
            st.subheader(t("Today's Product Details & Forecast"))
            product_ids = df["ProductID"].unique()
            product_names = [df.loc[df["ProductID"] == pid, "ProductName"].iloc[0] if "ProductName" in df.columns else str(pid) for pid in product_ids]
            # Add horizontal scroll for tabs if too many products
            st.markdown("""
                <style>
                .stTabs [role='tablist'] {
                    overflow-x: auto;
                    display: flex;
                    flex-wrap: nowrap;
                    scrollbar-width: thin;
                }
                .stTabs [role='tab'] {
                    flex: 0 0 auto;
                }
                </style>
            """, unsafe_allow_html=True)
            # --- Product Search Bar ---
            selected_product = st.text_input(t("Search Product Name"), "")
            # --- Product Tabs ---
            prod_tabs = st.tabs(product_names)
            # Load sales history once
            sales_hist = None
            try:
                sales_hist = pd.read_csv("Data/sales.csv")
                if "product_id" in sales_hist.columns:
                    sales_hist = sales_hist.rename(columns={"product_id": "ProductID"})
                if "units_sold" in sales_hist.columns:
                    sales_hist = sales_hist.rename(columns={"units_sold": "Quantity"})
                if "sale_date" in sales_hist.columns:
                    sales_hist = sales_hist.rename(columns={"sale_date": "OrderDate"})
                if "OrderDate" in sales_hist.columns:
                    sales_hist["OrderDate"] = pd.to_datetime(sales_hist["OrderDate"], errors="coerce")
            except Exception as e:
                sales_hist = None
                st.warning(f"Could not load historical sales for product tabs: {e}")
            # --- Show tab content: always show content for each tab, but highlight if search matches ---
            for i, pid in enumerate(product_ids):
                pname = product_names[i]
                with prod_tabs[i]:
                    # If search is present and doesn't match this tab, show a message
                    if selected_product and selected_product.lower() not in str(pname).lower():
                        st.info(t("No match for search."))
                    else:
                        st.markdown(f"### {pname}")
                        # Show predicted sales for the most recent predicted date
                        pred_row = df[df["ProductID"] == pid]
                        if not pred_row.empty and date_cols:
                            date_objs = [pd.to_datetime(dc, errors='coerce') for dc in date_cols]
                            latest_date_idx = date_objs.index(max(date_objs))
                            latest_date_col = date_cols[latest_date_idx]
                            st.write("**Predicted Sales for Today:**")
                            st.table(pred_row[["ProductID", "ProductName", latest_date_col]].rename(columns={
                                "ProductID": "Product ID",
                                "ProductName": "Product Name",
                                latest_date_col: "Predicted Sales",
                            }))
                        # Show sales forecast graph for this product
                        fig = go.Figure()
                        # Historical sales
                        if sales_hist is not None:
                            hist = sales_hist[sales_hist["ProductID"] == pid]
                            if not hist.empty:
                                order_date_col = "OrderDate" if "OrderDate" in hist.columns else ("sale_date" if "sale_date" in hist.columns else hist.columns[0])
                                fig.add_trace(go.Scatter(x=hist[order_date_col], y=hist["Quantity"], mode="lines+markers", name="Historical Sales", line=dict(color="#1f77b4")))
                        # Predicted sales for all available dates
                        if not pred_row.empty and date_cols:
                            y_pred = pred_row[date_cols].values.flatten().astype(float)
                            x_dates = [pd.to_datetime(dc, errors='coerce') for dc in date_cols]
                            today = pd.Timestamp.today().normalize()
                            # Add all predictions except today as a subtle line
                            other_mask = [(not pd.isnull(xd)) and xd != today for xd in x_dates]
                            if any(other_mask):
                                fig.add_trace(go.Scatter(
                                    x=[x for x, m in zip(x_dates, other_mask) if m],
                                    y=[y for y, m in zip(y_pred, other_mask) if m],
                                    mode="lines+markers",
                                    name="Predicted Sales",
                                    marker=dict(size=8, color="#B0B0B0", symbol="circle", line=dict(width=1, color="#888")),
                                    line=dict(color="#B0B0B0", width=2, dash="dot"),
                                    text=[f"{pname}: {val:.1f}" for val, m in zip(y_pred, other_mask) if m],
                                    hoverinfo='text'
                                ))
                            # Add today's prediction as a separate highlighted marker
                            today_mask = [(not pd.isnull(xd)) and xd == today for xd in x_dates]
                            if any(today_mask):
                                fig.add_trace(go.Scatter(
                                    x=[x for x, m in zip(x_dates, today_mask) if m],
                                    y=[y for y, m in zip(y_pred, today_mask) if m],
                                    mode="markers",
                                    name="Today's Predicted Sales",
                                    marker=dict(size=18, color="#FF5733", symbol="star", line=dict(width=2, color="#C70039")),
                                    text=[f"{pname}: {val:.1f}" for val, m in zip(y_pred, today_mask) if m],
                                    hoverinfo='text'
                                ))
                        fig.update_layout(
                            title=str(pname),
                            xaxis_title=t("Date"),
                            yaxis_title=t("Units Sold"),
                            legend_title=t("Type"),
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # --- All Predicted Sales for Today ---
            st.subheader(t("All Predicted Sales for Today"))
            # Only show predictions for the current date
            all_today = []
            if date_cols:
                today_str = pd.Timestamp.today().strftime('%Y-%m-%d')
                if today_str in date_cols:
                    for pid in product_ids:
                        pred_row = df[df["ProductID"] == pid]
                        if not pred_row.empty:
                            row = pred_row.iloc[0]
                            # Use Tamil column names if needed
                            product_id_col = t("Product ID")
                            product_name_col = t("Product Name")
                            predicted_sales_col = t("Predicted Sales")
                            predicted_date_col = t("Predicted Date")
                            # Some output files may have Tamil column names already, so check for both
                            product_id_val = row["ProductID"] if "ProductID" in row else row.get(product_id_col, pid)
                            product_name_val = row["ProductName"] if "ProductName" in row else row.get(product_name_col, str(pid))
                            predicted_sales_val = row[today_str] if today_str in row else row.get(predicted_sales_col, None)
                            all_today.append({
                                product_id_col: product_id_val,
                                product_name_col: product_name_val,
                                predicted_sales_col: predicted_sales_val,
                                predicted_date_col: today_str
                            })
                    if all_today:
                        all_today_df = pd.DataFrame(all_today)
                        st.dataframe(all_today_df)
                        st.download_button(
                            label=STRINGS[lang]['download_csv'],
                            data=all_today_df.to_csv(index=False),
                            file_name="dailySalesPrediction.csv"
                        )
                    else:
                        st.warning(t("No predictions found for today."))
                else:
                    st.warning(t("No predictions found for today."))
            else:
                st.warning(t("No predictions found for today."))
        else:
            st.warning(STRINGS[lang]['no_csv'])

# --- Repeat for other models (skeleton) ---

with tabs[1]:
    st.header(STRINGS[lang]['Expected Demand Predictor'])
    # Tamil localization for all UI, table, and graph labels
    def t(s):
        return {
            "Number of Days to Predict": "முன்னறிவிக்க வேண்டிய நாட்கள்",
            "Top 3 Products by Predicted Total Sales": "முன்னறிவிக்கப்பட்ட மொத்த விற்பனையில் சிறந்த 3 பொருட்கள்",
            "Product ID": "பொருள் ஐடி",
            "Product Name": "பொருள் பெயர்",
            "Category": "வகை",
            "Predicted Total Sales": "முன்னறிவிக்கப்பட்ட மொத்த விற்பனை",
            "All Predicted Demand": "முன்னறிவிக்கப்பட்ட அனைத்து தேவை",
            "Predicted Demand Visualization": "முன்னறிவிக்கப்பட்ட தேவை காட்சிப்படுத்தல்",
            "Predicted Total Sales for {cat} Category": "{cat} வகையில் முன்னறிவிக்கப்பட்ட மொத்த விற்பனை",
            "Predicted Total Sales by Product": "பொருள்படி முன்னறிவிக்கப்பட்ட மொத்த விற்பனை"
        }.get(s, s) if lang == 'ta' else s
    num_days = st.number_input(t("Number of Days to Predict"), min_value=1, max_value=60, value=7, step=1)
    if st.button(f"{STRINGS[lang]['run']} {STRINGS[lang]['Expected Demand Predictor']}"):
        import subprocess
        import time
        st.info(f"{STRINGS[lang]['run']} expectedDemandPredictor.py for {num_days} days ...")
        subprocess.run(["python", "Models/expectedDemandPredictor.py", "--predict_days", str(num_days)], capture_output=True, text=True)
        time.sleep(1)
        csv_path = os.path.join("Output", "expectedDemandPredictor.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # --- Only show ProductID, ProductName, NumDays, PredictedTotalSales, Category ---
            # Support both 'Days' and 'NumDays' column names
            days_col = "Days" if "Days" in df.columns else ("NumDays" if "NumDays" in df.columns else None)
            show_cols = [col for col in ["ProductID", "ProductName", "Category", "PredictedTotalSales"] if col in df.columns]
            # --- Top 3 Products Section ---
            st.subheader(t("Top 3 Products by Predicted Total Sales"))
            if "PredictedTotalSales" in df.columns:
                top3 = df.sort_values("PredictedTotalSales", ascending=False).head(3)
            st.table(top3[show_cols].rename(columns={
                "ProductID": t("Product ID"),
                "ProductName": t("Product Name"),
                "Category": t("Category"),
                "PredictedTotalSales": t("Predicted Total Sales")
            }))
            # --- All Predictions Table ---
            st.subheader(t("All Predicted Demand"))
            st.dataframe(
                df[show_cols].rename(columns={
                    "ProductID": t("Product ID"),
                    "ProductName": t("Product Name"),
                    "Category": t("Category"),
                    "PredictedTotalSales": t("Predicted Total Sales")
                })
            )
            st.download_button(
                label=STRINGS[lang]['download_csv'],
                data=df[show_cols].rename(columns={
                    "ProductID": t("Product ID"),
                    "ProductName": t("Product Name"),
                    "Category": t("Category"),
                    "PredictedTotalSales": t("Predicted Total Sales")
                }).to_csv(index=False),
                file_name="expectedDemandPredictor.csv"
            )
            # --- Visualization Section ---
            st.subheader(t("Predicted Demand Visualization"))
            # --- Product Category Tabs ---
            cat_col_local = t("Category") if t("Category") in df.columns else "Category"
            if cat_col_local in df.columns:
                categories = df[cat_col_local].unique()
                cat_tabs = st.tabs([str(cat) for cat in categories])
                for i, cat in enumerate(categories):
                    with cat_tabs[i]:
                        cat_df = df[df[cat_col_local] == cat]
                        # Sort by predicted total sales descending
                        cat_df = cat_df.sort_values("PredictedTotalSales", ascending=False)
                        bar_y = cat_df['ProductName'] if 'ProductName' in cat_df.columns else cat_df['ProductID']
                        bar_x = cat_df['PredictedTotalSales']
                        num_days_col = cat_df[days_col] if days_col and days_col in cat_df.columns else None
                        hover_text = [
                            f"Product: {bar_y.iloc[j]}<br>Predicted Total Sales: {bar_x.iloc[j]}<br>Number of Days: {num_days_col.iloc[j]}"
                            for j in range(len(cat_df))
                        ]
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Bar(
                            y=bar_y,
                            x=bar_x,
                            orientation='h',
                            marker=dict(color=bar_x, colorscale='Blues', line=dict(color='black', width=1)),
                            hovertext=hover_text,
                            hoverinfo='text',
                            text=[f"{bar_x.iloc[j]:.1f}" for j in range(len(cat_df))],
                            textposition='outside',
                        ))
                        fig.update_layout(
                            title=t("Predicted Total Sales for {cat} Category").format(cat=cat),
                            xaxis_title=t("Predicted Total Sales"),
                            yaxis_title=t("Product Name"),
                            margin=dict(l=120, r=40, t=60, b=40),
                            plot_bgcolor='rgba(0,0,0,0)',
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: show all products in one bar graph
                try:
                    df_sorted = df.sort_values("PredictedTotalSales", ascending=False)
                    bar_y = df_sorted['ProductName'] if 'ProductName' in df_sorted.columns else df_sorted['ProductID']
                    bar_x = df_sorted['PredictedTotalSales']
                    num_days_col = df_sorted[days_col] if days_col and days_col in df_sorted.columns else None
                    hover_text = [
                        f"Product: {bar_y.iloc[i]}<br>Predicted Total Sales: {bar_x.iloc[i]}<br>Number of Days: {num_days_col.iloc[i]}"
                        for i in range(len(df_sorted))
                    ]
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(
                        y=bar_y,
                        x=bar_x,
                        orientation='h',
                        marker=dict(color=bar_x, colorscale='Blues', line=dict(color='black', width=1)),
                        hovertext=hover_text,
                        hoverinfo='text',
                        text=[f"{bar_x.iloc[i]:.1f}" for i in range(len(df_sorted))],
                        textposition='outside',
                    ))
                    fig.update_layout(
                        title=t("Predicted Total Sales by Product"),
                        xaxis_title=t("Predicted Total Sales"),
                        yaxis_title=t("Product Name"),
                        margin=dict(l=120, r=40, t=60, b=40),
                        plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
        else:
            st.warning(STRINGS[lang]['no_csv'])


with tabs[2]:
    st.header(STRINGS[lang]['Restock Predictor'])
    # Tamil localization for all UI, table, and graph labels
    def t(s):
        return {
            "Top 3 Products to Restock (Next Day)": "அடுத்த நாள் மீண்டும் நிரப்ப வேண்டிய சிறந்த 3 பொருட்கள்",
            "Product ID": "பொருள் ஐடி",
            "Product Name": "பொருள் பெயர்",
            "Next Day Restock Qty": "அடுத்த நாள் மீண்டும் நிரப்ப அளவு",
            "Category": "வகை",
            "All Restock Predictions": "அனைத்து மீண்டும் நிரப்ப முன்னறிவிப்புகள்",
            "Restock Visualization by Category": "வகைப்படி மீண்டும் நிரப்ப காட்சிப்படுத்தல்",
            "Restock Qty for {cat} Category": "{cat} வகையில் மீண்டும் நிரப்ப அளவு",
            "Next Day Restock Qty by Product": "பொருள்படி அடுத்த நாள் மீண்டும் நிரப்ப அளவு",
            "Required columns not found in output CSV. Showing first 3 rows.": "தேவையான பத்திகள் CSV-இல் இல்லை. முதல் 3 வரிசைகள் காட்டப்படுகின்றன."
        }.get(s, s) if lang == 'ta' else s
    if st.button(f"{STRINGS[lang]['run']} {STRINGS[lang]['Restock Predictor']}"):
        import subprocess
        import time
        st.info(f"{STRINGS[lang]['run']} restockPredictor.py ...")
        subprocess.run(["python", "Models/restockPredictor.py"], capture_output=True, text=True)
        time.sleep(1)
        csv_path = os.path.join("Output", "restockPredictor.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # --- Top 3 Products to Restock ---
            st.subheader(t("Top 3 Products to Restock (Next Day)"))
            # Find columns for ProductID, ProductName, NextDayRestockQty (robust matching)
            id_col = None
            name_col = None
            qty_col = None
            for col in df.columns:
                col_norm = col.lower().replace(" ","").replace("-","").replace("_","")
                if col_norm in ["productid", "productid"]:
                    id_col = col
                elif col_norm in ["productname", "productname"]:
                    name_col = col
                elif col_norm in ["nextdayrestockqty", "nextdayrestockquantity", "restockqty", "restockquantity"]:
                    qty_col = col
            if id_col and name_col and qty_col:
                # Only display these columns, sorted by qty descending
                top3 = df.sort_values(qty_col, ascending=False)[[id_col, name_col, qty_col]].head(3)
                st.table(top3.rename(columns={
                    id_col: t("Product ID"),
                    name_col: t("Product Name"),
                    qty_col: t("Next Day Restock Qty")
                }))
            else:
                st.warning(t("Required columns not found in output CSV. Showing first 3 rows."))
                st.table(df.head(3))
            # --- All Predictions Table ---
            st.subheader(t("All Restock Predictions"))
            # Find category column robustly
            cat_col = None
            for col in df.columns:
                col_norm = col.lower().replace(" ","").replace("-","").replace("_","")
                if col_norm in ["category"]:
                    cat_col = col
                    break
            if id_col and name_col and qty_col:
                show_cols = [id_col, name_col, qty_col]
                if cat_col:
                    show_cols.insert(2, cat_col)  # Insert Category after ProductName
                all_restock_df = df[show_cols].copy()
                # Only show rows where Next Day Restock Qty > 0
                all_restock_df = all_restock_df[all_restock_df[qty_col] > 0]
                all_restock_df = all_restock_df.rename(columns={
                    id_col: t("Product ID"),
                    name_col: t("Product Name"),
                    qty_col: t("Next Day Restock Qty"),
                    cat_col: t("Category") if cat_col else cat_col
                })
                st.dataframe(all_restock_df)
                st.download_button(
                    label=STRINGS[lang]['download_csv'],
                    data=all_restock_df.to_csv(index=False),
                    file_name="restockPredictor.csv"
                )
                # --- Visualization Section ---
                st.subheader(t("Restock Visualization by Category"))
                if cat_col:
                    cat_col_local = t("Category") if t("Category") in all_restock_df.columns else cat_col
                    categories = all_restock_df[cat_col_local].unique()
                    cat_tabs = st.tabs([str(cat) for cat in categories])
                    for i, cat in enumerate(categories):
                        with cat_tabs[i]:
                            cat_df = all_restock_df[all_restock_df[cat_col_local] == cat]
                            # Use display column name for sorting and plotting
                            qty_col_disp = t("Next Day Restock Qty")
                            if qty_col_disp in cat_df.columns:
                                cat_df = cat_df.sort_values(qty_col_disp, ascending=False)
                                # Use display column names for bar_y
                                if t("Product Name") in cat_df.columns:
                                    bar_y = cat_df[t("Product Name")]
                                elif t("Product ID") in cat_df.columns:
                                    bar_y = cat_df[t("Product ID")]
                                else:
                                    bar_y = None
                                bar_x = cat_df[qty_col_disp]
                                hover_text = [
                                    f"Product: {bar_y.iloc[j]}<br>Restock Qty: {bar_x.iloc[j]}<br>Category: {cat}"
                                    for j in range(len(cat_df))
                                ]
                                import plotly.graph_objects as go
                                fig = go.Figure(go.Bar(
                                    y=bar_y,
                                    x=bar_x,
                                    orientation='h',
                                    marker=dict(color=bar_x, colorscale='Oranges', line=dict(color='black', width=1)),
                                    hovertext=hover_text,
                                    hoverinfo='text',
                                    text=[f"{bar_x.iloc[j]:.0f}" for j in range(len(cat_df))],
                                    textposition='outside',
                                ))
                                fig.update_layout(
                                    title=t("Restock Qty for {cat} Category").format(cat=cat),
                                    xaxis_title=t("Next Day Restock Qty"),
                                    yaxis_title=t("Product Name"),
                                    margin=dict(l=120, r=40, t=60, b=40),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    yaxis=dict(autorange="reversed"),
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"restock_chart_{cat}_{i}")
                            else:
                                st.warning(f"Column '{qty_col_disp}' not found in category dataframe.")
                            fig.update_layout(
                                title=t("Restock Qty for {cat} Category").format(cat=cat),
                                xaxis_title=t("Next Day Restock Qty"),
                                yaxis_title=t("Product Name"),
                                margin=dict(l=120, r=40, t=60, b=40),
                                plot_bgcolor='rgba(0,0,0,0)',
                                yaxis=dict(autorange="reversed"),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    # Fallback: show all products in one bar graph
                    try:
                        df_sorted = all_restock_df.sort_values("Next Day Restock Qty", ascending=False)
                        bar_y = df_sorted['Product Name'] if 'Product Name' in df_sorted.columns else df_sorted['Product ID']
                        bar_x = df_sorted['Next Day Restock Qty']
                        hover_text = [
                            f"Product: {bar_y.iloc[i]}<br>Restock Qty: {bar_x.iloc[i]}"
                            for i in range(len(df_sorted))
                        ]
                        import plotly.graph_objects as go
                        fig = go.Figure(go.Bar(
                            y=bar_y,
                            x=bar_x,
                            orientation='h',
                            marker=dict(color=bar_x, colorscale='Oranges', line=dict(color='black', width=1)),
                            hovertext=hover_text,
                            hoverinfo='text',
                            text=[f"{bar_x.iloc[i]:.0f}" for i in range(len(df_sorted))],
                            textposition='outside',
                        ))
                        fig.update_layout(
                            title=t("Next Day Restock Qty by Product"),
                            xaxis_title=t("Next Day Restock Qty"),
                            yaxis_title=t("Product Name"),
                            margin=dict(l=120, r=40, t=60, b=40),
                            plot_bgcolor='rgba(0,0,0,0)',
                            yaxis=dict(autorange="reversed"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
            else:
                st.dataframe(df)
                st.download_button(
                    label=STRINGS[lang]['download_csv'],
                    data=df.to_csv(index=False),
                    file_name="restockPredictor.csv"
                )
        else:
            st.warning(STRINGS[lang]['no_csv'])


with tabs[3]:
    st.header(STRINGS[lang]['Price Optimization'])
    # Tamil localization for all UI, table, and graph labels
    def t(s):
        return {
            "Top 3 Products by Expected Revenue": "எதிர்பார்க்கப்படும் வருவாயில் சிறந்த 3 பொருட்கள்",
            "Product ID": "பொருள் ஐடி",
            "Product Name": "பொருள் பெயர்",
            "Optimal Price": "சிறந்த விலை",
            "Expected Revenue": "எதிர்பார்க்கப்படும் வருவாய்",
            "Category": "வகை",
            "All Price Optimization Predictions": "அனைத்து விலை மேம்படுத்தல் முன்னறிவிப்புகள்",
            "Price Optimization Visualization by Category": "வகைப்படி விலை மேம்படுத்தல் காட்சிப்படுத்தல்",
            "Expected Revenue for {cat} Category": "{cat} வகையில் எதிர்பார்க்கப்படும் வருவாய்",
            "Expected Revenue by Product": "பொருள்படி எதிர்பார்க்கப்படும் வருவாய்",
            "No valid categories found for visualization.": "காட்சிப்படுத்தலுக்கு செல்லுபடியாகும் வகைகள் இல்லை."
        }.get(s, s) if lang == 'ta' else s
    if st.button(f"{STRINGS[lang]['run']} {STRINGS[lang]['Price Optimization']}"):
        import subprocess
        import time
        st.info(f"{STRINGS[lang]['run']} priceOptimizationPrediction.py ...")
        subprocess.run(["python", "Models/priceOptimizationPrediction.py"], capture_output=True, text=True)
        time.sleep(1)
        csv_path = os.path.join("Output", "priceOptimizationPrediction.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # --- Top 3 Products by Expected Revenue ---
            st.subheader(t("Top 3 Products by Expected Revenue"))
            # Robust column matching
            id_col = None
            name_col = None
            price_col = None
            revenue_col = None
            cat_col = None
            for col in df.columns:
                col_norm = col.lower().replace(" ","").replace("-","").replace("_","")
                if col_norm in ["productid"]:
                    id_col = col
                elif col_norm in ["productname"]:
                    name_col = col
                elif col_norm in ["optimalprice","priceoptimized","optimizedprice"]:
                    price_col = col
                elif col_norm in ["expectedrevenue","revenue"]:
                    revenue_col = col
                elif col_norm in ["category"]:
                    cat_col = col
            show_cols = [id_col, name_col]
            if cat_col:
                show_cols.append(cat_col)
            if price_col:
                show_cols.append(price_col)
            if revenue_col:
                show_cols.append(revenue_col)
            # Only display these columns, sorted by expected revenue descending
            if revenue_col:
                top3 = df.sort_values(revenue_col, ascending=False)[show_cols].head(3)
                st.table(top3.rename(columns={
                    id_col: t("Product ID"),
                    name_col: t("Product Name"),
                    price_col: t("Optimal Price") if price_col else price_col,
                    revenue_col: t("Expected Revenue") if revenue_col else revenue_col,
                    cat_col: t("Category") if cat_col else cat_col
                }))
            else:
                st.warning("Required columns not found in output CSV. Showing first 3 rows.")
                st.table(df.head(3))
            # --- All Price Optimization Predictions ---
            st.subheader(t("All Price Optimization Predictions"))
            all_priceopt_df = df[show_cols].copy()
            all_priceopt_df = all_priceopt_df.rename(columns={
                id_col: t("Product ID"),
                name_col: t("Product Name"),
                price_col: t("Optimal Price") if price_col else price_col,
                revenue_col: t("Expected Revenue") if revenue_col else revenue_col,
                cat_col: t("Category") if cat_col else cat_col
            })
            st.dataframe(all_priceopt_df)
            st.download_button(
                label=STRINGS[lang]['download_csv'],
                data=all_priceopt_df.to_csv(index=False),
                file_name="priceOptimizationPrediction.csv"
            )
            # --- Visualization Section ---
            st.subheader(t("Price Optimization Visualization by Category"))
            if cat_col:
                cat_col_local = t("Category") if t("Category") in all_priceopt_df.columns else "Category"
                categories = [cat for cat in all_priceopt_df[cat_col_local].unique() if pd.notnull(cat) and str(cat).strip() != ""]
                if categories:
                    cat_tabs = st.tabs([str(cat) for cat in categories])
                    for i, cat in enumerate(categories):
                        with cat_tabs[i]:
                            cat_df = all_priceopt_df[all_priceopt_df[cat_col_local] == cat]
                            # Use display column name for sorting and plotting
                            expected_revenue_col = t("Expected Revenue")
                            optimal_price_col = t("Optimal Price")
                            product_name_col = t("Product Name")
                            product_id_col = t("Product ID")
                            if expected_revenue_col in cat_df.columns:
                                cat_df = cat_df.sort_values(expected_revenue_col, ascending=False)
                                bar_y = cat_df[product_name_col] if product_name_col in cat_df.columns else (cat_df[product_id_col] if product_id_col in cat_df.columns else None)
                                bar_x = cat_df[expected_revenue_col]
                                price_x = cat_df[optimal_price_col] if optimal_price_col in cat_df.columns else None
                                hover_text = [
                                    f"Product: {bar_y.iloc[j]}<br>Expected Revenue: {bar_x.iloc[j]}<br>Optimal Price: {price_x.iloc[j]}<br>Category: {cat}"
                                    for j in range(len(cat_df))
                                ] if bar_x is not None and price_x is not None else None
                                import plotly.graph_objects as go
                                fig = go.Figure(go.Bar(
                                    y=bar_y,
                                    x=bar_x,
                                    orientation='h',
                                    marker=dict(color=bar_x, colorscale='Greens', line=dict(color='black', width=1)),
                                    hovertext=hover_text,
                                    hoverinfo='text',
                                    text=[f"{bar_x.iloc[j]:.2f}" for j in range(len(cat_df))] if bar_x is not None else None,
                                    textposition='outside',
                                ))
                                fig.update_layout(
                                    title=t("Expected Revenue for {cat} Category").format(cat=cat),
                                    xaxis_title=expected_revenue_col,
                                    yaxis_title=product_name_col,
                                    margin=dict(l=120, r=40, t=60, b=40),
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    yaxis=dict(autorange="reversed"),
                                )
                                st.plotly_chart(fig, use_container_width=True, key=f"priceopt_chart_{cat}_{i}")
                            else:
                                st.warning(f"Column '{expected_revenue_col}' not found in category dataframe.")
                else:
                    st.info(t("No valid categories found for visualization."))
            else:
                # Fallback: show all products in one bar graph
                try:
                    df_sorted = all_priceopt_df.sort_values("Expected Revenue", ascending=False)
                    bar_y = df_sorted['Product Name'] if 'Product Name' in df_sorted.columns else df_sorted['Product ID']
                    bar_x = df_sorted['Expected Revenue'] if 'Expected Revenue' in df_sorted.columns else None
                    price_x = df_sorted['Optimal Price'] if 'Optimal Price' in df_sorted.columns else None
                    hover_text = [
                        f"Product: {bar_y.iloc[i]}<br>Expected Revenue: {bar_x.iloc[i]}<br>Optimal Price: {price_x.iloc[i]}"
                        for i in range(len(df_sorted))
                    ] if bar_x is not None and price_x is not None else None
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(
                        y=bar_y,
                        x=bar_x,
                        orientation='h',
                        marker=dict(color=bar_x, colorscale='Greens', line=dict(color='black', width=1)),
                        hovertext=hover_text,
                        hoverinfo='text',
                        text=[f"{bar_x.iloc[i]:.2f}" for i in range(len(df_sorted))] if bar_x is not None else None,
                        textposition='outside',
                    ))
                    fig.update_layout(
                        title=t("Expected Revenue by Product"),
                        xaxis_title=t("Expected Revenue"),
                        yaxis_title=t("Product Name"),
                        margin=dict(l=120, r=40, t=60, b=40),
                        plot_bgcolor='rgba(0,0,0,0)',
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
        else:
            st.warning(STRINGS[lang]['no_csv'])


with tabs[4]:
    st.header(STRINGS[lang]['Promotion Effectiveness'])
    if st.button(f"{STRINGS[lang]['run']} {STRINGS[lang]['Promotion Effectiveness']}"):
        import subprocess
        import time
        st.info(f"{STRINGS[lang]['run']} promotionEffectivenessModel.py ...")
        subprocess.run(["python", "Models/promotionEffectivenessModel.py"], capture_output=True, text=True)
        time.sleep(1)
        csv_path = os.path.join("Output", "promotionEffectivenessModel.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # --- Aggregate: average uplift, quantity, and expected no-promo per product ---
            # Robust column matching
            id_col = None
            name_col = None
            uplift_col = None
            cat_col = None
            qty_col = None
            no_promo_col = None
            for col in df.columns:
                col_norm = col.lower().replace(" ","").replace("-","").replace("_","")
                if col_norm == "productid":
                    id_col = col
                elif col_norm == "productname":
                    name_col = col
                elif col_norm in ["uplift","promotionuplift","effectiveness","avgpromotionuplift"]:
                    uplift_col = col
                elif col_norm == "category":
                    cat_col = col
                elif col_norm in ["quantity","qty","soldquantity","unitsold"]:
                    qty_col = col
                elif col_norm in ["expectednopromo","expectedquantitynopromo","expectednopromotion"]:
                    no_promo_col = col
            # Build group columns
            group_cols = [id_col]
            if name_col:
                group_cols.append(name_col)
            if cat_col:
                group_cols.append(cat_col)
            # Aggregate
            agg_dict = {}
            if uplift_col:
                agg_dict[uplift_col] = 'mean'
            if qty_col:
                agg_dict[qty_col] = 'mean'
            if no_promo_col:
                agg_dict[no_promo_col] = 'mean'
            if id_col and uplift_col:
                agg_df = df.groupby(group_cols, as_index=False).agg(agg_dict)
                # Rename columns for display
                if lang == 'ta':
                    rename_dict = {
                        id_col: "பொருள் ஐடி",
                        name_col: "பொருள் பெயர்" if name_col else name_col,
                        uplift_col: "சராசரி விளம்பர உயர்வு",
                        cat_col: "வகை" if cat_col else cat_col,
                        qty_col: "சராசரி அளவு" if qty_col else qty_col,
                        no_promo_col: "சராசரி விளம்பரமில்லா எதிர்பார்ப்பு" if no_promo_col else no_promo_col
                    }
                else:
                    rename_dict = {
                        id_col: "Product ID",
                        name_col: "Product Name" if name_col else name_col,
                        uplift_col: "Avg Promotion Uplift",
                        cat_col: "Category" if cat_col else cat_col,
                        qty_col: "Avg Quantity" if qty_col else qty_col,
                        no_promo_col: "Avg Expected No Promo" if no_promo_col else no_promo_col
                    }
                agg_df = agg_df.rename(columns=rename_dict)
                # --- Top 3 Products by Avg Promotion Uplift ---
                st.subheader("சராசரி விளம்பர உயர்வில் சிறந்த 3 பொருட்கள்" if lang == 'ta' else "Top 3 Products by Avg Promotion Uplift")
                top3 = agg_df.sort_values(("சராசரி விளம்பர உயர்வு" if lang == 'ta' else "Avg Promotion Uplift"), ascending=False).head(3)
                st.table(top3[[c for c in [
                    "பொருள் ஐடி" if lang == 'ta' else "Product ID",
                    "பொருள் பெயர்" if lang == 'ta' else "Product Name",
                    "சராசரி விளம்பர உயர்வு" if lang == 'ta' else "Avg Promotion Uplift",
                    "வகை" if lang == 'ta' else "Category"
                ] if c in top3.columns]])

                # --- Visualization: Avg Promotion Uplift by Product, tabbed by Category and Uplift Sign ---
                st.subheader("வகை மற்றும் உயர்வு வகைப்படி சராசரி விளம்பர உயர்வு" if lang == 'ta' else "Avg Promotion Uplift Visualization by Category and Uplift Sign")
                uplift_col_name = "சராசரி விளம்பர உயர்வு" if lang == 'ta' else "Avg Promotion Uplift"
                cat_col_name = "வகை" if lang == 'ta' else "Category"
                pname_col = "பொருள் பெயர்" if lang == 'ta' else "Product Name"
                pid_col = "பொருள் ஐடி" if lang == 'ta' else "Product ID"
                if cat_col_name in agg_df.columns and uplift_col_name in agg_df.columns:
                    categories = agg_df[cat_col_name].unique()
                    cat_tabs = st.tabs([str(cat) for cat in categories])
                    for i, cat in enumerate(categories):
                        with cat_tabs[i]:
                            cat_df = agg_df[agg_df[cat_col_name] == cat]
                            # Split by positive/negative uplift
                            pos_df = cat_df[cat_df[uplift_col_name] > 0]
                            neg_df = cat_df[cat_df[uplift_col_name] <= 0]
                            st.markdown("**உயர்வு உள்ள பொருட்கள்**" if lang == 'ta' else "**Products with Positive Uplift**")
                            if not pos_df.empty:
                                st.dataframe(pos_df[[c for c in [pname_col, uplift_col_name] if c in pos_df.columns]].sort_values(uplift_col_name, ascending=False))
                                import plotly.graph_objects as go
                                fig = go.Figure(go.Bar(
                                    x=pos_df[pname_col] if pname_col in pos_df.columns else pos_df[pid_col],
                                    y=pos_df[uplift_col_name],
                                    marker=dict(color=pos_df[uplift_col_name], colorscale='Blues'),
                                    text=[f"{val:.2f}" for val in pos_df[uplift_col_name]],
                                    textposition='outside',
                                ))
                                fig.update_layout(
                                    title=(f'{cat} வகையில் உயர்வு உள்ள பொருட்கள்' if lang == 'ta' else f'Positive Avg Promotion Uplift for {cat}'),
                                    xaxis_title=("பொருள்" if lang == 'ta' else "Product"),
                                    yaxis_title=(uplift_col_name),
                                    margin=dict(l=60, r=40, t=60, b=40),
                                    plot_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("இந்த வகையில் உயர்வு உள்ள பொருட்கள் இல்லை." if lang == 'ta' else "No products with positive uplift in this category.")
                            st.markdown("**உயர்வு இல்லாத அல்லது குறைந்த பொருட்கள்**" if lang == 'ta' else "**Products with Negative or Zero Uplift**")
                            if not neg_df.empty:
                                st.dataframe(neg_df[[c for c in [pname_col, uplift_col_name] if c in neg_df.columns]].sort_values(uplift_col_name))
                                import plotly.graph_objects as go
                                fig = go.Figure(go.Bar(
                                    x=neg_df[pname_col] if pname_col in neg_df.columns else neg_df[pid_col],
                                    y=neg_df[uplift_col_name],
                                    marker=dict(color=neg_df[uplift_col_name], colorscale='Reds'),
                                    text=[f"{val:.2f}" for val in neg_df[uplift_col_name]],
                                    textposition='outside',
                                ))
                                fig.update_layout(
                                    title=(f'{cat} வகையில் உயர்வு இல்லாத/குறைந்த பொருட்கள்' if lang == 'ta' else f'Negative/Zero Avg Promotion Uplift for {cat}'),
                                    xaxis_title=("பொருள்" if lang == 'ta' else "Product"),
                                    yaxis_title=(uplift_col_name),
                                    margin=dict(l=60, r=40, t=60, b=40),
                                    plot_bgcolor='rgba(0,0,0,0)'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("இந்த வகையில் உயர்வு இல்லாத/குறைந்த பொருட்கள் இல்லை." if lang == 'ta' else "No products with negative or zero uplift in this category.")
                else:
                    st.info("வகை அல்லது சராசரி விளம்பர உயர்வு பத்திகள் காணப்படவில்லை." if lang == 'ta' else "Category or Avg Promotion Uplift column missing for visualization.")

                # --- All Products Table ---
                st.dataframe(agg_df)
                st.download_button(
                    label=STRINGS[lang]['download_csv'],
                    data=agg_df.to_csv(index=False),
                    file_name="promotionEffectivenessModel.csv"
                )
            else:
                st.dataframe(df)
                st.download_button(
                    label=STRINGS[lang]['download_csv'],
                    data=df.to_csv(index=False),
                    file_name="promotionEffectivenessModel.csv"
                )
        else:
            st.warning(STRINGS[lang]['no_csv'])



# --- Inventory Turnover Tab ---

with tabs[5]:
    st.header(f"{STRINGS[lang]['Inventory Turnover']} ({STRINGS[lang]['ABC Analysis']})")
    if st.button(f"{STRINGS[lang]['run']} {STRINGS[lang]['Inventory Turnover']}"):
        import subprocess
        import time
        st.info(f"{STRINGS[lang]['run']} inventoryTurnoverModel.py & abcAnalysisModel.py ..." if lang == 'en' else f"{STRINGS[lang]['run']} inventoryTurnoverModel.py மற்றும் abcAnalysisModel.py ...")
        subprocess.run(["python", "Models/inventoryTurnoverModel.py"], capture_output=True, text=True)
        subprocess.run(["python", "Models/abcAnalysisModel.py"], capture_output=True, text=True)
        time.sleep(1)
        csv_path = os.path.join("Output", "inventoryTurnoverModel.csv")
        abc_csv_path = os.path.join("Output", "abcAnalysisModel.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Try to merge ABC info if available
            if os.path.exists(abc_csv_path):
                abc_df = pd.read_csv(abc_csv_path)
                # Try to merge on ProductID or Product Name
                merge_col = None
                for col in ["ProductID", "Product Name", "ProductName"]:
                    if col in df.columns and col in abc_df.columns:
                        merge_col = col
                        break
                if merge_col:
                    df = pd.merge(df, abc_df[[merge_col, "ABC_Class"]], on=merge_col, how="left")
                elif "ProductID" in df.columns and "ProductID" in abc_df.columns:
                    df = pd.merge(df, abc_df[["ProductID", "ABC_Class"]], on="ProductID", how="left")
                elif "Product Name" in df.columns and "Product Name" in abc_df.columns:
                    df = pd.merge(df, abc_df[["Product Name", "ABC_Class"]], on="Product Name", how="left")
                else:
                    # Fallback: no merge
                    pass
            # --- Main Table: Only show selected columns ---
            # Robust column matching
            id_col = None
            name_col = None
            units_col = None
            avg_inv_col = None
            turnover_col = None
            turnover_class_col = None
            cat_col = None
            abc_col = None
            for col in df.columns:
                col_norm = col.lower().replace(" ","").replace("-","").replace("_","")
                if col_norm in ["productid"]:
                    id_col = col
                elif col_norm in ["productname"]:
                    name_col = col
                elif col_norm in ["totalunitssold","unitssold","soldunits"]:
                    units_col = col
                elif col_norm in ["averageinventory","avginventory"]:
                    avg_inv_col = col
                elif col_norm in ["turnoverratio","inventoryturnoverratio"]:
                    turnover_col = col
                elif col_norm in ["turnoverclass","predictedturnoverclass"]:
                    turnover_class_col = col
                elif col_norm in ["category"]:
                    cat_col = col
                elif col_norm in ["abcclass","abc_class"]:
                    abc_col = col
            show_cols = [id_col, name_col, units_col, avg_inv_col, turnover_col, turnover_class_col, cat_col, abc_col]
            show_cols = [c for c in show_cols if c]
            main_df = df[show_cols].copy()
            # Tamil column names
            if lang == 'ta':
                main_df = main_df.rename(columns={
                    id_col: "பொருள் ஐடி",
                    name_col: "பொருள் பெயர்",
                    units_col: "மொத்த விற்பனை அலகுகள்",
                    avg_inv_col: "சராசரி சரக்கு",
                    turnover_col: "சுழற்சி விகிதம்",
                    turnover_class_col: "முன்னறிவிக்கப்பட்ட சுழற்சி வகுப்பு",
                    cat_col: "வகை",
                    abc_col: "ABC வகுப்பு"
                })
            else:
                main_df = main_df.rename(columns={
                    id_col: "Product ID",
                    name_col: "Product Name",
                    units_col: "Total Units Sold",
                    avg_inv_col: "Avg Inventory",
                    turnover_col: "Turnover Ratio",
                    turnover_class_col: "Predicted Turnover Class",
                    cat_col: "Category",
                    abc_col: "ABC Class"
                })

            # --- Visualization: Inventory Turnover Ratio by Product (Tabbed Table & Bar Chart) ---
            # Tamil labels for turnover classes
            turnover_class_colors = {
                "fast": "#2ca02c",
                "medium": "#ff7f0e",
                "slow": "#d62728"
            }
            turnover_classes = [c for c in ["fast", "medium", "slow"] if any(main_df[("முன்னறிவிக்கப்பட்ட சுழற்சி வகுப்பு" if lang == 'ta' else "Predicted Turnover Class")].str.lower() == c)]
            tab_labels = [
                ("வேகமான" if tc == "fast" and lang == 'ta' else
                 "நடுத்தர" if tc == "medium" and lang == 'ta' else
                 "மெதுவான" if tc == "slow" and lang == 'ta' else tc.capitalize())
                for tc in turnover_classes
            ]
            tabs_tc = st.tabs(tab_labels)
            for i, tclass in enumerate(turnover_classes):
                with tabs_tc[i]:
                    class_df = main_df[main_df[("முன்னறிவிக்கப்பட்ட சுழற்சி வகுப்பு" if lang == 'ta' else "Predicted Turnover Class")].str.lower() == tclass]
                    # --- Top 3 Products by Turnover Ratio for this class ---
                    st.subheader((f"{tab_labels[i]} வகை சிறந்த 3 பொருட்கள்" if lang == 'ta' else f"Top 3 {tclass.capitalize()} Turnover Products"))
                    top3 = class_df.sort_values(("சுழற்சி விகிதம்" if lang == 'ta' else "Turnover Ratio"), ascending=False).head(3)
                    st.table(top3[[c for c in [
                        "பொருள் ஐடி" if lang == 'ta' else "Product ID",
                        "பொருள் பெயர்" if lang == 'ta' else "Product Name",
                        "சுழற்சி விகிதம்" if lang == 'ta' else "Turnover Ratio",
                        "மொத்த விற்பனை அலகுகள்" if lang == 'ta' else "Total Units Sold",
                        "சராசரி சரக்கு" if lang == 'ta' else "Avg Inventory",
                        "வகை" if lang == 'ta' else "Category",
                        "ABC வகுப்பு" if lang == 'ta' else "ABC Class"
                    ] if c in top3.columns]])
                    # --- All Products Table for this class ---
                    st.subheader((f"{tab_labels[i]} வகை அனைத்து பொருட்கள்" if lang == 'ta' else f"All {tclass.capitalize()} Turnover Products"))
                    st.dataframe(class_df[[c for c in [
                        "பொருள் ஐடி" if lang == 'ta' else "Product ID",
                        "பொருள் பெயர்" if lang == 'ta' else "Product Name",
                        "சுழற்சி விகிதம்" if lang == 'ta' else "Turnover Ratio",
                        "மொத்த விற்பனை அலகுகள்" if lang == 'ta' else "Total Units Sold",
                        "சராசரி சரக்கு" if lang == 'ta' else "Avg Inventory",
                        "வகை" if lang == 'ta' else "Category",
                        "ABC வகுப்பு" if lang == 'ta' else "ABC Class"
                    ] if c in class_df.columns]].sort_values(("சுழற்சி விகிதம்" if lang == 'ta' else "Turnover Ratio"), ascending=False))
                    # --- Bar chart for this class ---
                    import plotly.graph_objects as go
                    if not class_df.empty:
                        # Color by ABC class if available, else by turnover class
                        abc_col_name = "ABC வகுப்பு" if lang == 'ta' else "ABC Class"
                        pname_col = "பொருள் பெயர்" if lang == 'ta' else "Product Name"
                        turnover_col_name = "சுழற்சி விகிதம்" if lang == 'ta' else "Turnover Ratio"
                        if abc_col_name in class_df.columns:
                            abc_colors = {"A": "#2ca02c", "B": "#ff7f0e", "C": "#d62728"}
                            fig = go.Figure()
                            for abc in ["A", "B", "C"]:
                                abc_df = class_df[class_df[abc_col_name].astype(str).str.upper() == abc]
                                if not abc_df.empty:
                                    fig.add_trace(go.Bar(
                                        x=abc_df[pname_col],
                                        y=abc_df[turnover_col_name],
                                        marker=dict(color=abc_colors[abc]),
                                        name=(f"ABC {abc}" if lang == 'en' else f"ABC {abc}"),
                                        text=[f"{val:.2f}" for val in abc_df[turnover_col_name]],
                                        textposition='outside',
                                    ))
                            fig.update_layout(
                                title=(f"{tab_labels[i]} வகை பொருட்களின் சுழற்சி விகிதம்" if lang == 'ta' else f"Inventory Turnover Ratio by Product ({tclass.capitalize()})"),
                                xaxis_title=("பொருள் பெயர்" if lang == 'ta' else "Product Name"),
                                yaxis_title=("சுழற்சி விகிதம்" if lang == 'ta' else "Turnover Ratio"),
                                margin=dict(l=60, r=40, t=60, b=40),
                                plot_bgcolor='rgba(0,0,0,0)',
                                legend_title=("ABC வகுப்பு" if lang == 'ta' else "ABC Class")
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig = go.Figure(go.Bar(
                                x=class_df[pname_col],
                                y=class_df[turnover_col_name],
                                marker=dict(color=turnover_class_colors[tclass]),
                                text=[f"{val:.2f}" for val in class_df[turnover_col_name]],
                                textposition='outside',
                                name=tab_labels[i]
                            ))
                            fig.update_layout(
                                title=(f"{tab_labels[i]} வகை பொருட்களின் சுழற்சி விகிதம்" if lang == 'ta' else f"Inventory Turnover Ratio by Product ({tclass.capitalize()})"),
                                xaxis_title=("பொருள் பெயர்" if lang == 'ta' else "Product Name"),
                                yaxis_title=("சுழற்சி விகிதம்" if lang == 'ta' else "Turnover Ratio"),
                                margin=dict(l=60, r=40, t=60, b=40),
                                plot_bgcolor='rgba(0,0,0,0)',
                                legend_title=("சுழற்சி வகுப்பு" if lang == 'ta' else "Turnover Class")
                            )
                            st.plotly_chart(fig, use_container_width=True)

            st.subheader("சரக்கு சுழற்சி அட்டவணை (ABC வகுப்புடன்)" if lang == 'ta' else "Inventory Turnover Table (with ABC Class)")
            st.dataframe(main_df)
            st.download_button(
                label=STRINGS[lang]['download_csv'],
                data=main_df.to_csv(index=False),
                file_name="inventoryTurnoverModel.csv"
            )
        else:
            st.warning(STRINGS[lang]['no_csv'])

st.markdown("---")
st.info(STRINGS[lang]['info'])