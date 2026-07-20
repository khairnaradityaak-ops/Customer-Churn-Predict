#!/usr/bin/env python
# coding: utf-8

# In[4]:





# In[5]:




# In[6]:


import pandas as pd
import numpy as np


# In[7]:


np.random.seed(42)
N = 70000 
tenure = np.random.randint(1, 73, N)
monthly_charges = np.round(np.random.uniform(20, 120, N), 2)
total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 50, N), 2)
total_charges = np.clip(total_charges, 0, None) 
contract = np.random.choice(
    ["Month-to-Month", "One Year", "Two Year"],
    N,
    p=[0.55, 0.25, 0.20]  
)
internet_service = np.random.choice(
    ["DSL", "Fiber Optic", "No"],
    N,
    p=[0.35, 0.45, 0.20]
)
payment_method = np.random.choice(
    ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
    N,
    p=[0.35, 0.25, 0.20, 0.20]
)
tech_support = np.random.choice(["Yes", "No"], N, p=[0.40, 0.60])
num_support_calls = np.random.randint(0, 11, N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
churn_prob = (
    0.05                                                        # Base churn rate
    + 0.25 * (contract == "Month-to-Month")                    # Monthly = risky
    + 0.10 * (monthly_charges > 80)                            # High bill
    + 0.10 * (tenure < 12)                                     # New customers
    + 0.08 * (tech_support == "No")                            # No support
    + 0.05 * (internet_service == "Fiber Optic")               # Fiber complaints
    + 0.02 * (senior_citizen == 1)                             # Senior customers
    + 0.01 * (num_support_calls > 5)                           # Many complaints
)
churn_prob = np.clip(churn_prob, 0, 1)
churn = (np.random.rand(N) < churn_prob).astype(int)
df = pd.DataFrame({
    "CustomerID":        [f"CUST{str(i).zfill(6)}" for i in range(1, N+1)],
    "Tenure":            tenure,
    "MonthlyCharges":    monthly_charges,
    "TotalCharges":      total_charges,
    "Contract":          contract,
    "InternetService":   internet_service,
    "PaymentMethod":     payment_method,
    "TechSupport":       tech_support,
    "NumSupportCalls":   num_support_calls,
    "SeniorCitizen":     senior_citizen,
    "Churn":             churn   # 1 = Churned, 0 = Stayed
})
df.to_csv("churn_data.csv", index=False)
print(f"Dataset created: {N} customers")
print(f"Churn rate: {churn.mean()*100:.1f}%")
print(f"Columns: {list(df.columns)}")
print(df.head())


# In[8]:


import pandas as pd
import numpy as np


# In[9]:


df = pd.read_csv("churn_data.csv")


# In[10]:


print("=" * 55)
print("1. BASIC SHAPE & INFO")
print("=" * 55)
print(f"Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
print(f"\nColumn data types:\n{df.dtypes}")

print("\n" + "=" * 55)
print("2. MISSING VALUES CHECK")
print("=" * 55)
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")

print("\n" + "=" * 55)
print("3. DUPLICATE ROWS CHECK")
print("=" * 55)
dupes = df.duplicated().sum()
print(f"Duplicate rows: {dupes}")
if dupes > 0:
    df = df.drop_duplicates()
    print(f"Removed {dupes} duplicates. New shape: {df.shape}")

print("\n" + "=" * 55)
print("4. CHURN DISTRIBUTION")
print("=" * 55)
churn_counts = df["Churn"].value_counts()
churn_pct    = df["Churn"].value_counts(normalize=True) * 100
print(f"Not Churned (0): {churn_counts[0]:,}  ({churn_pct[0]:.1f}%)")
print(f"Churned     (1): {churn_counts[1]:,}  ({churn_pct[1]:.1f}%)")
print("NOTE: Imbalanced dataset — we handle this during modelling")

print("\n" + "=" * 55)
print("5. NUMERICAL FEATURE STATISTICS")
print("=" * 55)
num_cols = ["Tenure", "MonthlyCharges", "TotalCharges", "NumSupportCalls"]
print(df[num_cols].describe().round(2))

print("\n" + "=" * 55)
print("6. CHURN RATE BY CONTRACT TYPE")
print("=" * 55)
# Group by contract and calculate churn rate — key business insight
contract_churn = df.groupby("Contract")["Churn"].mean() * 100
for contract, rate in contract_churn.items():
    print(f"  {contract:<20} Churn Rate: {rate:.1f}%")

print("\n" + "=" * 55)
print("7. CHURN RATE BY INTERNET SERVICE")
print("=" * 55)
internet_churn = df.groupby("InternetService")["Churn"].mean() * 100
for service, rate in internet_churn.items():
    print(f"  {service:<20} Churn Rate: {rate:.1f}%")

print("\n" + "=" * 55)
print("8. CHURN RATE BY TECH SUPPORT")
print("=" * 55)
support_churn = df.groupby("TechSupport")["Churn"].mean() * 100
for support, rate in support_churn.items():
    print(f"  Tech Support {support:<6} Churn Rate: {rate:.1f}%")

print("\n" + "=" * 55)
print("9. AVERAGE MONTHLY CHARGES: CHURNED VS NOT")
print("=" * 55)
avg_charges = df.groupby("Churn")["MonthlyCharges"].mean()
print(f"  Not Churned: ${avg_charges[0]:.2f}/month")
print(f"  Churned:     ${avg_charges[1]:.2f}/month")

print("\n" + "=" * 55)
print("10. CORRELATION WITH CHURN (numerical features)")
print("=" * 55)
# Shows which numerical features have strongest relationship with churn
correlations = df[num_cols + ["Churn"]].corr()["Churn"].drop("Churn").sort_values(ascending=False)
for col, corr in correlations.items():
    direction = "positive" if corr > 0 else "negative"
    print(f"  {col:<22} Correlation: {corr:+.3f}  ({direction})")

print("\n✅ EDA Complete — Data is clean and ready for feature engineering")


# In[ ]:





# In[ ]:





# In[11]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


# In[12]:


df = pd.read_csv("churn_data.csv")


# In[13]:


print("=" * 55)
print("1. FEATURE ENGINEERING — Creating new features")
print("=" * 55)

# Average charge per month of service
# High value = customer paying more relative to tenure
df["ChargePerMonth"] = df["TotalCharges"] / (df["Tenure"] + 1)
print("Created: ChargePerMonth (TotalCharges / Tenure)")

# Engagement score — longer tenure + lower support calls = loyal customer
df["EngagementScore"] = df["Tenure"] - df["NumSupportCalls"] * 2
print("Created: EngagementScore (Tenure - NumSupportCalls*2)")

# High value customer flag — top 25% by monthly charges
charge_threshold = df["MonthlyCharges"].quantile(0.75)
df["IsHighValue"] = (df["MonthlyCharges"] > charge_threshold).astype(int)
print(f"Created: IsHighValue (MonthlyCharges > ${charge_threshold:.0f})")

# New customer flag — less than 12 months tenure
df["IsNewCustomer"] = (df["Tenure"] < 12).astype(int)
print("Created: IsNewCustomer (Tenure < 12 months)")

print("\n" + "=" * 55)
print("2. ENCODING CATEGORICAL VARIABLES")
print("=" * 55)
# ML models only understand numbers, not text
# So we convert text categories into numbers

# Label Encoding — converts categories to 0, 1, 2 etc.
le = LabelEncoder()

cat_cols = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]
for col in cat_cols:
    df[col + "_Encoded"] = le.fit_transform(df[col])
    unique_vals = df[col].unique()
    encoded_vals = le.transform(unique_vals)
    mapping = dict(zip(unique_vals, encoded_vals))
    print(f"  {col}: {mapping}")

print("\n" + "=" * 55)
print("3. SELECTING FINAL FEATURES")
print("=" * 55)

# These are the features (X) we feed into the model
feature_cols = [
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "NumSupportCalls",
    "SeniorCitizen",
    "ChargePerMonth",       # engineered
    "EngagementScore",      # engineered
    "IsHighValue",          # engineered
    "IsNewCustomer",        # engineered
    "Contract_Encoded",
    "InternetService_Encoded",
    "PaymentMethod_Encoded",
    "TechSupport_Encoded",
]

X = df[feature_cols]
y = df["Churn"]   # Target variable (what we want to predict)

print(f"Features selected: {len(feature_cols)}")
for f in feature_cols:
    print(f"  - {f}")

print("\n" + "=" * 55)
print("4. TRAIN / TEST SPLIT")
print("=" * 55)
# 80% of data used to train the model
# 20% held back to test how well it performs on unseen data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test
    random_state=42,     # For reproducibility
    stratify=y           # Ensures same churn ratio in both splits
)
print(f"Training set:  {X_train.shape[0]:,} rows")
print(f"Test set:      {X_test.shape[0]:,} rows")
print(f"Churn in train: {y_train.mean()*100:.1f}%")
print(f"Churn in test:  {y_test.mean()*100:.1f}%")

print("\n" + "=" * 55)
print("5. FEATURE SCALING")
print("=" * 55)
# StandardScaler makes all features have mean=0 and std=1
# This prevents features with large numbers (TotalCharges)
# from dominating features with small numbers (SeniorCitizen)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit on train only!
X_test_scaled  = scaler.transform(X_test)        # Apply same scale to test

print("Applied StandardScaler to all features")
print("Mean of scaled training features:", X_train_scaled.mean().round(4))
print("Std  of scaled training features:", X_train_scaled.std().round(4))

# Save processed data for next step
import joblib
joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols), "processed_data.pkl")
print("\n✅ Preprocessing complete — saved to processed_data.pkl")


# In[ ]:





# In[ ]:





# In[14]:


import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


# In[15]:


X_train, X_test, y_train, y_test, scaler, feature_cols = joblib.load("processed_data.pkl")

print("=" * 55)
print("1. HANDLING CLASS IMBALANCE")
print("=" * 55)
# Our dataset has ~70% non-churn vs ~30% churn
# class_weight='balanced' tells the model to pay more
# attention to the minority class (churned customers)
print(f"Training churn rate: {y_train.mean()*100:.1f}%")
print("Using class_weight='balanced' to handle imbalance")

print("\n" + "=" * 55)
print("2. TRAINING RANDOM FOREST")
print("=" * 55)
# Random Forest = many decision trees voting together
# n_estimators = number of trees (more = better but slower)
# max_depth = how deep each tree can grow
rf_model = RandomForestClassifier(
    n_estimators=100,        # 100 decision trees
    max_depth=10,            # Limit depth to prevent overfitting
    class_weight="balanced", # Handle imbalanced classes
    random_state=42,
    n_jobs=-1                # Use all CPU cores for speed
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest trained!")

print("\n" + "=" * 55)
print("3. TRAINING GRADIENT BOOSTING (XGBoost equivalent)")
print("=" * 55)
# Gradient Boosting = builds trees sequentially,
# each tree fixing errors of the previous one
gb_model = GradientBoostingClassifier(
    n_estimators=100,    # 100 boosting rounds
    learning_rate=0.1,   # How much each tree contributes
    max_depth=5,         # Shallower trees than RF
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print("Gradient Boosting trained!")

print("\n" + "=" * 55)
print("4. MODEL EVALUATION METRICS EXPLAINED")
print("=" * 55)
print("""
  Accuracy  = Overall correct predictions / total predictions
  Precision = Of predicted churners, how many actually churned?
  Recall    = Of actual churners, how many did we catch?
  F1 Score  = Balance between Precision and Recall

  For churn: RECALL is most important!
  Missing a churner (false negative) costs more than
  a false alarm (false positive) in business terms.
""")

def evaluate_model(name, y_true, y_pred):
    print(f"\n── {name} ──────────────────────────")
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Precision: {prec*100:.1f}%")
    print(f"  Recall:    {rec*100:.1f}%")
    print(f"  F1 Score:  {f1*100:.1f}%")
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted NO  Predicted YES")
    print(f"  Actual NO   :     {cm[0][0]:>6,}        {cm[0][1]:>6,}")
    print(f"  Actual YES  :     {cm[1][0]:>6,}        {cm[1][1]:>6,}")
    print(f"\n  True Negatives (TN):  {cm[0][0]:,}  — Correctly predicted NOT churn")
    print(f"  False Positives (FP): {cm[0][1]:,}  — Wrongly flagged as churn")
    print(f"  False Negatives (FN): {cm[1][0]:,}  — Missed actual churners (costly!)")
    print(f"  True Positives (TP):  {cm[1][1]:,}  — Correctly caught churners")
    return acc, prec, rec, f1

rf_metrics = evaluate_model("RANDOM FOREST", y_test, rf_pred)
gb_metrics = evaluate_model("GRADIENT BOOSTING", y_test, gb_pred)

print("\n" + "=" * 55)
print("5. FEATURE IMPORTANCE (what drives churn?)")
print("=" * 55)
# Random Forest gives us feature importances
# Higher = that feature has more influence on churn prediction
importances = rf_model.feature_importances_
sorted_idx  = np.argsort(importances)[::-1]
print("  Top features driving churn prediction:")
for i in range(min(8, len(feature_cols))):
    idx = sorted_idx[i]
    print(f"  {i+1}. {feature_cols[idx]:<25} Importance: {importances[idx]:.4f}")

print("\n" + "=" * 55)
print("6. CHOOSING BEST MODEL")
print("=" * 55)
best = "Random Forest" if rf_metrics[3] >= gb_metrics[3] else "Gradient Boosting"
best_model = rf_model if best == "Random Forest" else gb_model
print(f"Best model: {best} (highest F1 score)")

# Save best model
joblib.dump(best_model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
print("\n✅ Model saved to churn_model.pkl")
print("✅ Ready for dashboard in Step 5")


# In[ ]:





# In[2]:





# In[16]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib


# In[18]:


st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Load model and data ────────────────────────────────────
@st.cache_data   # Cache so it doesn't reload on every interaction
def load_data():
    df = pd.read_csv("churn_data.csv")
    return df

@st.cache_resource
def load_model():
    model       = joblib.load("churn_model.pkl")
    scaler      = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, scaler, feature_cols

df = load_data()
model, scaler, feature_cols = load_model()

# ── Header ─────────────────────────────────────────────────
st.title("📊 Customer Churn Analytics Dashboard")
st.markdown("**Built for:** Telecom Business Intelligence Team | **Model:** Random Forest")
st.markdown("---")

# ── KPI Metrics Row ────────────────────────────────────────
# These are the summary numbers at the top — key KPIs
col1, col2, col3, col4 = st.columns(4)

total_customers = len(df)
total_churned   = df["Churn"].sum()
churn_rate      = df["Churn"].mean() * 100
avg_monthly     = df[df["Churn"]==1]["MonthlyCharges"].mean()
revenue_at_risk = df[df["Churn"]==1]["MonthlyCharges"].sum()

col1.metric("Total Customers",    f"{total_customers:,}")
col2.metric("Churned Customers",  f"{total_churned:,}")
col3.metric("Churn Rate",         f"{churn_rate:.1f}%")
col4.metric("Monthly Revenue at Risk", f"${revenue_at_risk:,.0f}")

st.markdown("---")

# ── Sidebar Filters ────────────────────────────────────────
st.sidebar.header("🔍 Filter Data")

contract_filter = st.sidebar.multiselect(
    "Contract Type",
    options=df["Contract"].unique(),
    default=df["Contract"].unique()
)

internet_filter = st.sidebar.multiselect(
    "Internet Service",
    options=df["InternetService"].unique(),
    default=df["InternetService"].unique()
)

tenure_range = st.sidebar.slider(
    "Tenure (months)",
    min_value=int(df["Tenure"].min()),
    max_value=int(df["Tenure"].max()),
    value=(int(df["Tenure"].min()), int(df["Tenure"].max()))
)

# Apply filters
filtered_df = df[
    (df["Contract"].isin(contract_filter)) &
    (df["InternetService"].isin(internet_filter)) &
    (df["Tenure"].between(tenure_range[0], tenure_range[1]))
]

st.markdown(f"**Showing {len(filtered_df):,} customers** based on filters")

# ── Charts Row 1 ───────────────────────────────────────────
st.subheader("📈 Churn Analysis by Segment")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Churn Rate by Contract Type**")
    contract_churn = filtered_df.groupby("Contract")["Churn"].mean() * 100
    contract_churn = contract_churn.reset_index()
    contract_churn.columns = ["Contract", "Churn Rate (%)"]
    contract_churn["Churn Rate (%)"] = contract_churn["Churn Rate (%)"].round(1)
    st.bar_chart(contract_churn.set_index("Contract"))

with col2:
    st.markdown("**Churn Rate by Internet Service**")
    internet_churn = filtered_df.groupby("InternetService")["Churn"].mean() * 100
    internet_churn = internet_churn.reset_index()
    internet_churn.columns = ["Internet Service", "Churn Rate (%)"]
    internet_churn["Churn Rate (%)"] = internet_churn["Churn Rate (%)"].round(1)
    st.bar_chart(internet_churn.set_index("Internet Service"))

# ── Charts Row 2 ───────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.markdown("**Monthly Charges: Churned vs Retained**")
    churned     = filtered_df[filtered_df["Churn"]==1]["MonthlyCharges"]
    not_churned = filtered_df[filtered_df["Churn"]==0]["MonthlyCharges"]
    charge_compare = pd.DataFrame({
        "Churned":     churned.describe()[["mean","50%","75%"]],
        "Not Churned": not_churned.describe()[["mean","50%","75%"]]
    })
    charge_compare.index = ["Average", "Median", "75th Percentile"]
    st.dataframe(charge_compare.round(2), use_container_width=True)

with col4:
    st.markdown("**Churn Rate by Tenure Group**")
    filtered_df2 = filtered_df.copy()
    filtered_df2["Tenure Group"] = pd.cut(
        filtered_df2["Tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"]
    )
    tenure_churn = filtered_df2.groupby("Tenure Group", observed=True)["Churn"].mean() * 100
    st.bar_chart(tenure_churn)

st.markdown("---")

# ── Live Churn Predictor ────────────────────────────────────
st.subheader("🔮 Live Churn Risk Predictor")
st.markdown("Enter a customer's details to predict their churn probability:")

pred_col1, pred_col2, pred_col3 = st.columns(3)

with pred_col1:
    tenure_input   = st.slider("Tenure (months)", 1, 72, 12)
    monthly_input  = st.slider("Monthly Charges ($)", 20, 120, 65)
    support_input  = st.slider("Support Calls", 0, 10, 2)

with pred_col2:
    contract_input  = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
    internet_input  = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
    tech_input      = st.selectbox("Tech Support", ["Yes", "No"])

with pred_col3:
    payment_input   = st.selectbox("Payment Method", ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"])
    senior_input    = st.selectbox("Senior Citizen", ["No", "Yes"])

if st.button("🔍 Predict Churn Risk", type="primary"):
    # Encode inputs exactly as we did in training
    contract_map  = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
    internet_map  = {"DSL": 0, "Fiber Optic": 1, "No": 2}
    payment_map   = {"Bank Transfer": 0, "Credit Card": 1, "Electronic Check": 2, "Mailed Check": 3}
    tech_map      = {"No": 0, "Yes": 1}

    total_charges  = tenure_input * monthly_input
    charge_per_mo  = total_charges / (tenure_input + 1)
    engagement     = tenure_input - support_input * 2
    is_high_value  = 1 if monthly_input > 90 else 0
    is_new         = 1 if tenure_input < 12 else 0

    input_data = np.array([[
        tenure_input,
        monthly_input,
        total_charges,
        support_input,
        1 if senior_input == "Yes" else 0,
        charge_per_mo,
        engagement,
        is_high_value,
        is_new,
        contract_map[contract_input],
        internet_map[internet_input],
        payment_map[payment_input],
        tech_map[tech_input],
    ]])

    input_scaled = scaler.transform(input_data)
    churn_prob   = model.predict_proba(input_scaled)[0][1] * 100

    if churn_prob >= 60:
        st.error(f"🔴 HIGH CHURN RISK: {churn_prob:.1f}% — Immediate retention action recommended")
    elif churn_prob >= 35:
        st.warning(f"🟡 MEDIUM CHURN RISK: {churn_prob:.1f}% — Monitor and engage proactively")
    else:
        st.success(f"🟢 LOW CHURN RISK: {churn_prob:.1f}% — Customer appears loyal")

st.markdown("---")
st.caption("Customer Churn Analytics Dashboard | Built by Aditya Khairnar | Powered by Random Forest ML Model")


# In[ ]:




