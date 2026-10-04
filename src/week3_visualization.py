"""
Week 3 — Financial Data Visualization & Reporting
Loads Week 2 cleaned data and recreates the Week 3 visual outputs.
Run from the repository root:
    python src/week3_visualization.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned" / "quarterly_cleaned.csv"
OUT = ROOT / "reports" / "week3" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COMPANIES = ["TCS", "Infosys", "HCLTech", "Wipro"]
QUARTERS = [
    "Q1 FY24","Q2 FY24","Q3 FY24","Q4 FY24",
    "Q1 FY25","Q2 FY25","Q3 FY25","Q4 FY25",
    "Q1 FY26","Q2 FY26","Q3 FY26","Q4 FY26","Q1 FY27"
]

def load_data():
    df = pd.read_csv(DATA)
    df["Fiscal Quarter"] = pd.Categorical(df["Fiscal Quarter"], categories=QUARTERS, ordered=True)
    return df.sort_values(["Company", "Fiscal Quarter"])

def save_series_chart(df, column, title, ylabel, filename, quarters=QUARTERS):
    plt.figure(figsize=(10.8, 5.8))
    ax = plt.gca()
    x = list(range(len(quarters)))
    for company in COMPANIES:
        s = df[(df["Company"] == company) & (df["Fiscal Quarter"].isin(quarters))]
        s = s.sort_values("Fiscal Quarter")
        ax.plot(x, s[column], marker="o", linewidth=2, label=company)
    ax.set_title(title, fontsize=16, pad=14, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(quarters, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=4, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=220, bbox_inches="tight")
    plt.close()

def main():
    df = load_data()
    save_series_chart(df, "Revenue (INR Cr)", "Quarterly Revenue Trend — Common Panel", "Revenue (₹ crore)", "01_revenue_trend.png")
    indexed = df.copy()
    base = indexed.groupby("Company")["Revenue (INR Cr)"].transform("first")
    indexed["Revenue Index"] = indexed["Revenue (INR Cr)"] / base * 100
    save_series_chart(indexed, "Revenue Index", "Revenue Index — Q1 FY24 = 100", "Index (Q1 FY24 = 100)", "02_revenue_index.png")
    yoy = df[df["Fiscal Quarter"].isin(QUARTERS[4:])]
    save_series_chart(yoy, "YoY Revenue Growth (%)", "Year-on-Year Revenue Growth — INR Reporting Basis", "YoY Growth (%)", "03_yoy_growth.png", QUARTERS[4:])
    save_series_chart(df, "Operating Margin (%)", "Operating Margin Trend — Standardized Week 2 Panel", "Panel Operating Margin (%)", "04_operating_margin.png")
    save_series_chart(df, "Net Margin (%)", "Net Margin Trend — Standardized Week 2 Panel", "Net Margin (%)", "05_net_margin.png")
    print("Week 3 figures created in:", OUT)

if __name__ == "__main__":
    main()
