"""
==================================================================
Customer Churn Analysis & Business Intelligence Dashboard
==================================================================
Module: churn_analysis.py
Purpose: End-to-end business analytics pipeline — data cleaning,
         exploratory analysis, KPI calculation, and chart generation
         to support a customer retention business case.

Dataset: Telco Customer Churn (Kaggle / IBM sample dataset)
Author:  [Your Name]
==================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 150
IMG_DIR = "../images"
os.makedirs(IMG_DIR, exist_ok=True)


# ==================================================================
# STEP 1: DATA UNDERSTANDING
# ==================================================================
# Column reference (for stakeholder documentation):
#   customerID       - Unique identifier for each customer (not predictive, used as key)
#   gender           - Customer gender (Male/Female)
#   SeniorCitizen    - 1 if customer is 65+, else 0
#   Partner          - Whether customer has a partner (Yes/No)
#   Dependents       - Whether customer has dependents (Yes/No)
#   tenure           - Number of months the customer has stayed with the company
#   PhoneService     - Whether customer has phone service
#   MultipleLines    - Whether customer has multiple phone lines
#   InternetService  - Type of internet service (DSL / Fiber optic / No)
#   OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
#   StreamingTV, StreamingMovies - add-on services (Yes/No/No internet service)
#   Contract         - Contract term (Month-to-month / One year / Two year)
#   PaperlessBilling - Whether customer uses paperless billing
#   PaymentMethod    - How the customer pays (Electronic check, Mailed check,
#                      Bank transfer (automatic), Credit card (automatic))
#   MonthlyCharges   - Current monthly bill amount ($)
#   TotalCharges     - Total amount billed to the customer to date ($)
#   Churn            - Target variable: did the customer leave? (Yes/No)


def load_data(path="../data/telco_churn.csv"):
    """Load raw data from CSV."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    return df


# ==================================================================
# STEP 2: DATA CLEANING
# ==================================================================
def clean_data(df):
    """
    Cleaning steps performed (each documented for audit/stakeholder review):

    1. Duplicate removal      - drop exact duplicate rows (by customerID)
    2. Datatype correction    - TotalCharges is stored as text due to blank
                                 strings for brand-new customers; convert to numeric
    3. Missing value handling - blank TotalCharges (tenure=0 customers) filled
                                 with their MonthlyCharges (1 month billed)
    4. Outlier detection      - flag (not remove) customers with unusually high
                                 MonthlyCharges using the IQR method, for awareness
    5. Feature formatting     - standardize Yes/No -> 1/0 flags for analysis;
                                 create tenure buckets for cohort-style analysis
    """
    before = len(df)
    df = df.drop_duplicates(subset="customerID")
    dupes_removed = before - len(df)

    # Datatype correction + missing value handling
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_total_charges = df["TotalCharges"].isna().sum()
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    # Outlier detection (IQR method) — flagged, not dropped, since these
    # are legitimate high-value customers, not data errors
    q1, q3 = df["MonthlyCharges"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    df["is_high_value_outlier"] = df["MonthlyCharges"] > upper_bound

    # Feature formatting
    df["Churn_Flag"] = df["Churn"].map({"Yes": 1, "No": 0})
    df["SeniorCitizen_Label"] = df["SeniorCitizen"].map({1: "Senior", 0: "Non-Senior"})

    bins = [0, 6, 12, 24, 48, 72]
    tenure_labels = ["0-6 mo", "6-12 mo", "1-2 yr", "2-4 yr", "4-6 yr"]
    df["tenure_group"] = pd.cut(df["tenure"], bins=bins, labels=tenure_labels, include_lowest=True)

    print("--- Data Cleaning Summary ---")
    print(f"Duplicate rows removed: {dupes_removed}")
    print(f"Missing TotalCharges filled: {missing_total_charges}")
    print(f"High-value outliers flagged: {df['is_high_value_outlier'].sum()}")
    print(f"Final row count: {len(df)}")

    return df


# ==================================================================
# STEP 3: BUSINESS KPI CALCULATION
# ==================================================================
def calculate_kpis(df):
    """Calculate headline business KPIs for the executive summary / dashboard."""
    total_customers = len(df)
    churned = df["Churn_Flag"].sum()
    active = total_customers - churned
    churn_rate = churned / total_customers * 100

    avg_monthly_charges = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    monthly_revenue = df["MonthlyCharges"].sum()
    revenue_active = df.loc[df["Churn_Flag"] == 0, "MonthlyCharges"].sum()
    # Estimated revenue lost = churned customers' monthly charges x 12 (annualized)
    revenue_lost_annualized = df.loc[df["Churn_Flag"] == 1, "MonthlyCharges"].sum() * 12

    # Simple CLV estimate = avg monthly charge x avg tenure (months)
    clv_estimate = avg_monthly_charges * avg_tenure

    high_risk_pct = (
        df[(df["Contract"] == "Month-to-month") & (df["tenure"] <= 12)].shape[0]
        / total_customers * 100
    )

    kpis = {
        "total_customers": int(total_customers),
        "active_customers": int(active),
        "churned_customers": int(churned),
        "churn_rate_pct": round(churn_rate, 1),
        "avg_monthly_charges": round(avg_monthly_charges, 2),
        "avg_tenure_months": round(avg_tenure, 1),
        "monthly_revenue_total": round(monthly_revenue, 2),
        "monthly_revenue_active": round(revenue_active, 2),
        "estimated_annual_revenue_lost": round(revenue_lost_annualized, 2),
        "customer_lifetime_value_estimate": round(clv_estimate, 2),
        "high_risk_customer_pct": round(high_risk_pct, 1),
    }

    print("\n--- Business KPIs ---")
    for k, v in kpis.items():
        print(f"  {k}: {v}")

    return kpis


# ==================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (charts + business insight)
# ==================================================================
def chart_overall_churn(df):
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = df["Churn"].value_counts()
    ax.pie(counts, labels=["Stayed", "Churned"], autopct="%1.1f%%",
           colors=["#2E86AB", "#E63946"], startangle=90, textprops={"fontsize": 12})
    ax.set_title("Overall Customer Churn Rate", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/01_overall_churn.png")
    plt.close()


def chart_churn_by_contract(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = df.groupby("Contract")["Churn_Flag"].mean().sort_values(ascending=False) * 100
    vals.plot(kind="bar", color="#E63946", ax=ax)
    ax.set_ylabel("Churn Rate (%)"); ax.set_xlabel("Contract Type")
    ax.set_title("Churn Rate by Contract Type", fontsize=14, fontweight="bold")
    plt.xticks(rotation=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/02_churn_by_contract.png")
    plt.close()
    return vals


def chart_churn_by_tenure(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = df.groupby("tenure_group", observed=True)["Churn_Flag"].mean() * 100
    vals.plot(kind="bar", color="#2E86AB", ax=ax)
    ax.set_ylabel("Churn Rate (%)"); ax.set_xlabel("Customer Tenure")
    ax.set_title("Churn Rate by Customer Tenure", fontsize=14, fontweight="bold")
    plt.xticks(rotation=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/03_churn_by_tenure.png")
    plt.close()
    return vals


def chart_charges_vs_churn(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn",
                palette=["#2E86AB", "#E63946"], legend=False, ax=ax)
    ax.set_title("Monthly Charges: Churned vs Retained Customers", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/04_charges_vs_churn.png")
    plt.close()


def chart_churn_by_internet(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    vals = df.groupby("InternetService")["Churn_Flag"].mean().sort_values(ascending=False) * 100
    vals.plot(kind="bar", color="#F4A261", ax=ax)
    ax.set_ylabel("Churn Rate (%)"); ax.set_xlabel("Internet Service Type")
    ax.set_title("Churn Rate by Internet Service Type", fontsize=14, fontweight="bold")
    plt.xticks(rotation=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/05_churn_by_internet.png")
    plt.close()
    return vals


def chart_churn_by_payment(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    vals = df.groupby("PaymentMethod")["Churn_Flag"].mean().sort_values(ascending=False) * 100
    vals.plot(kind="bar", color="#9B5DE5", ax=ax)
    ax.set_ylabel("Churn Rate (%)"); ax.set_xlabel("Payment Method")
    ax.set_title("Churn Rate by Payment Method", fontsize=14, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/06_churn_by_payment.png")
    plt.close()
    return vals


def chart_churn_by_senior(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = df.groupby("SeniorCitizen_Label")["Churn_Flag"].mean() * 100
    vals.plot(kind="bar", color="#00B4A6", ax=ax)
    ax.set_ylabel("Churn Rate (%)"); ax.set_xlabel("")
    ax.set_title("Churn Rate: Senior vs Non-Senior Citizens", fontsize=14, fontweight="bold")
    plt.xticks(rotation=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/07_churn_by_senior.png")
    plt.close()
    return vals


def chart_churn_by_gender(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = df.groupby("gender")["Churn_Flag"].mean() * 100
    vals.plot(kind="bar", color="#457B9D", ax=ax)
    ax.set_ylabel("Churn Rate (%)"); ax.set_xlabel("")
    ax.set_title("Churn Rate by Gender", fontsize=14, fontweight="bold")
    plt.xticks(rotation=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/08_churn_by_gender.png")
    plt.close()
    return vals


def chart_correlation_heatmap(df):
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Churn_Flag", "SeniorCitizen"]
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f", ax=ax,
                cbar_kws={"label": "Correlation"})
    ax.set_title("Correlation Heatmap: Numeric Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/09_correlation_heatmap.png")
    plt.close()


def chart_distributions(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.histplot(df["tenure"], bins=30, color="#2E86AB", ax=axes[0])
    axes[0].set_title("Tenure Distribution")
    sns.histplot(df["MonthlyCharges"], bins=30, color="#E63946", ax=axes[1])
    axes[1].set_title("Monthly Charges Distribution")
    sns.histplot(df["TotalCharges"], bins=30, color="#F4A261", ax=axes[2])
    axes[2].set_title("Total Charges Distribution")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/10_distributions.png")
    plt.close()


def chart_customer_segmentation(df):
    """Simple segmentation: tenure vs monthly charges, colored by churn."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for churn_val, color, label in [(0, "#2E86AB", "Stayed"), (1, "#E63946", "Churned")]:
        subset = df[df["Churn_Flag"] == churn_val]
        ax.scatter(subset["tenure"], subset["MonthlyCharges"], alpha=0.4, s=15,
                   color=color, label=label)
    ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Customer Segmentation: Tenure vs Monthly Charges", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/11_customer_segmentation.png")
    plt.close()


def run_full_eda(df):
    """Run all EDA charts and return key aggregates for the report."""
    chart_overall_churn(df)
    contract_churn = chart_churn_by_contract(df)
    tenure_churn = chart_churn_by_tenure(df)
    chart_charges_vs_churn(df)
    internet_churn = chart_churn_by_internet(df)
    payment_churn = chart_churn_by_payment(df)
    senior_churn = chart_churn_by_senior(df)
    gender_churn = chart_churn_by_gender(df)
    chart_correlation_heatmap(df)
    chart_distributions(df)
    chart_customer_segmentation(df)

    print("\nAll 11 EDA charts saved to /images.")

    return {
        "contract_churn": contract_churn.to_dict(),
        "tenure_churn": tenure_churn.to_dict(),
        "internet_churn": internet_churn.to_dict(),
        "payment_churn": payment_churn.to_dict(),
        "senior_churn": senior_churn.to_dict(),
        "gender_churn": gender_churn.to_dict(),
    }


# ==================================================================
# STEP 5: OPTIONAL — BASELINE PREDICTIVE MODEL
# (Kept simple/optional per project scope: business insight > ML)
# ==================================================================
def optional_baseline_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    model_df = df.copy()
    cat_cols = model_df.select_dtypes(include="object").columns.drop(["customerID", "Churn"])
    le = LabelEncoder()
    for col in cat_cols:
        model_df[col] = le.fit_transform(model_df[col])

    features = ["tenure", "MonthlyCharges", "TotalCharges", "Contract",
                "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport"]
    X = model_df[features]
    y = model_df["Churn_Flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n[Optional] Baseline model accuracy: {acc:.1%}")
    return round(acc * 100, 1)


# ==================================================================
# MAIN PIPELINE
# ==================================================================
if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    kpis = calculate_kpis(df)
    eda_results = run_full_eda(df)
    model_accuracy = optional_baseline_model(df)

    # Save everything needed for the report / dashboard
    output = {"kpis": kpis, "eda": eda_results, "model_accuracy": model_accuracy}
    with open("../reports/analysis_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Also export a cleaned dataset for SQL / Power BI use
    df.to_csv("../data/telco_churn_cleaned.csv", index=False)
    print("\nCleaned dataset exported for SQL/Power BI use.")
    print("Analysis complete. Results saved to reports/analysis_results.json")
