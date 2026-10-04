"""Reproduce the Week 2 cleaning layer.

Reads data/raw/quarterly_raw.csv and writes data/cleaned/quarterly_cleaned.csv.
The script intentionally keeps source metadata and flags Q1 FY27 as issuer-cross-checked.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "quarterly_raw.csv"
OUT = ROOT / "data" / "cleaned" / "quarterly_cleaned.csv"

def main() -> None:
    df = pd.read_csv(RAW)
    required = ["Company", "Period End", "Fiscal Quarter", "Revenue (INR Cr)", "Operating Profit (INR Cr)", "Net Profit (INR Cr)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Revenue (INR Cr)"] = pd.to_numeric(df["Revenue (INR Cr)"], errors="raise")
    df["Operating Profit (INR Cr)"] = pd.to_numeric(df["Operating Profit (INR Cr)"], errors="raise")
    df["Net Profit (INR Cr)"] = pd.to_numeric(df["Net Profit (INR Cr)"], errors="raise")
    df["Period Order"] = df["Fiscal Quarter"].map({q:i for i,q in enumerate([
        "Q1 FY24","Q2 FY24","Q3 FY24","Q4 FY24","Q1 FY25","Q2 FY25","Q3 FY25","Q4 FY25",
        "Q1 FY26","Q2 FY26","Q3 FY26","Q4 FY26","Q1 FY27"])})
    if df["Period Order"].isna().any():
        raise ValueError("Unrecognized fiscal-quarter label detected")
    df = df.sort_values(["Company", "Period Order"]).reset_index(drop=True)

    df["QoQ Revenue Growth (%)"] = df.groupby("Company")["Revenue (INR Cr)"].pct_change() * 100
    df["YoY Revenue Growth (%)"] = df.groupby("Company")["Revenue (INR Cr)"].pct_change(4) * 100
    df["Operating Margin (%)"] = df["Operating Profit (INR Cr)"] / df["Revenue (INR Cr)"] * 100
    df["Net Margin (%)"] = df["Net Profit (INR Cr)"] / df["Revenue (INR Cr)"] * 100
    df["Unit"] = "INR Cr"
    df["Data Status"] = df["Fiscal Quarter"].eq("Q1 FY27").map({True:"Q1 FY27 standardized; issuer cross-check completed", False:"Historical standardized; secondary extraction"})
    df = df.drop(columns=["Period Order"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.4f")
    print(f"Wrote {len(df)} rows to {OUT}")

if __name__ == "__main__":
    main()
