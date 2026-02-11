import pandas as pd
import numpy as np
import os

# -----------------------------
# Helper Functions
# -----------------------------

def is_index_like(series, row_count):
    """Check if column is serial number / ID"""
    if not pd.api.types.is_numeric_dtype(series):
        return False

    if series.nunique() < row_count * 0.9:
        return False

    diffs = series.dropna().sort_values().diff().dropna()
    return diffs.nunique() == 1


def probe_numeric(series):
    """Try converting text column to numbers"""
    converted = pd.to_numeric(series, errors="coerce")
    success_rate = converted.notna().mean()
    return converted, success_rate


def missing_risk(pct):
    if pct < 5:
        return "Low"
    elif pct < 30:
        return "Medium"
    else:
        return "High"


# -----------------------------
# Core Analyzer
# -----------------------------

def analyze_dataset(csv_path):

    df = pd.read_csv(csv_path)
    rows, cols = df.shape

    results = []

    for col in df.columns:
        series = df[col]

        row = {
            "column": col,
            "data_type": "",
            "missing_percent": round(series.isna().mean() * 100, 2),
            "trust": "",
            "remarks": ""
        }

        # 1️⃣ Check Identifier column
        if is_index_like(series, rows):
            row["data_type"] = "Identifier"
            row["trust"] = "Ignored"
            row["remarks"] = "Serial number / ID column"
            results.append(row)
            continue

        miss_risk = missing_risk(row["missing_percent"])

        # 2️⃣ Numeric column
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = series
            row["data_type"] = "Numeric"

        # 3️⃣ Object column
        elif pd.api.types.is_object_dtype(series):
            converted, success = probe_numeric(series)

            if success >= 0.9:
                numeric_series = converted
                row["data_type"] = "Numeric (from text)"
                row["remarks"] = "Converted text to numbers"
            else:
                row["data_type"] = "Categorical"
                row["trust"] = "Needs Cleaning" if miss_risk != "Low" else "Reliable"
                row["remarks"] = "Category data"
                results.append(row)
                continue
        else:
            row["data_type"] = "Other"
            row["trust"] = "Needs Cleaning"
            row["remarks"] = "Unknown data type"
            results.append(row)
            continue

        # 4️⃣ Statistics for numeric columns
        mean = numeric_series.mean()
        median = numeric_series.median()
        std = numeric_series.std()
        iqr = numeric_series.quantile(0.75) - numeric_series.quantile(0.25)

        distorted = abs(mean - median) > (0.5 * std if std > 0 else 0)
        unstable = std > iqr if iqr > 0 else False

        # 5️⃣ Balanced Trust Decision
        if miss_risk == "High":
            row["trust"] = "High Risk"
            row["remarks"] = "Too many missing values"

        elif distorted and unstable:
            row["trust"] = "High Risk"
            row["remarks"] = "Outliers affecting data"

        elif miss_risk == "Medium" or distorted:
            row["trust"] = "Needs Cleaning"
            row["remarks"] = "Some issues detected"

        else:
            row["trust"] = "Reliable"
            row["remarks"] = "Looks consistent"

        results.append(row)

    result_df = pd.DataFrame(results)

    # Dataset level verdict
    high_risk_count = (result_df["trust"] == "High Risk").sum()

    if high_risk_count > len(result_df) * 0.4:
        verdict = "Dataset is NOT reliable"
    elif high_risk_count > 0:
        verdict = "Dataset needs cleaning"
    else:
        verdict = "Dataset looks safe"

    return result_df, verdict


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":

    path = r"C:\Users\ASUS\Documents\inomatics\heavy_sample.csv"

    df_result, verdict = analyze_dataset(path)

    print("\n📊 Data Trust Analysis Completed")
    print("\nResult:")
    print(df_result)

    print("\nFinal Verdict:", verdict)

    # Ask user for export
    choice = input("\nDo you want to export this result as CSV? (yes/no): ").strip().lower()

    if choice in ["yes", "y"]:

        os.makedirs("processed", exist_ok=True)

        base_name = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join("processed", f"{base_name}_result.csv")

        df_result.to_csv(output_path, index=False)

        print(f"\n📁 Result saved at: {output_path}")

    else:
        print("\nResult not saved.")
