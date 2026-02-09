import pandas as pd
import numpy as np
import os

# -----------------------------
# Helper Functions
# -----------------------------

def is_index_like(series, row_count):
    if not pd.api.types.is_numeric_dtype(series):
        return False
    if series.nunique() < row_count * 0.9:
        return False
    diffs = series.dropna().sort_values().diff().dropna()
    return diffs.nunique() == 1


def probe_numeric(series):
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

        # Index-like column
        if is_index_like(series, rows):
            row["data_type"] = "Identifier"
            row["trust"] = "Ignored"
            row["remarks"] = "Index / serial number column"
            results.append(row)
            continue

        miss_risk = missing_risk(row["missing_percent"])

        # Numeric column
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = series
            row["data_type"] = "Numeric"

        # Object column
        elif pd.api.types.is_object_dtype(series):
            converted, success = probe_numeric(series)
            if success >= 0.9:
                numeric_series = converted
                row["data_type"] = "Numeric (from text)"
                row["remarks"] = "Converted text to numbers"
            else:
                row["data_type"] = "Categorical"
                row["trust"] = "Needs Cleaning" if miss_risk != "Low" else "Reliable"
                row["remarks"] = "Category based data"
                results.append(row)
                continue
        else:
            row["data_type"] = "Other"
            row["trust"] = "Needs Cleaning"
            row["remarks"] = "Unclear data type"
            results.append(row)
            continue

        # Statistics
        mean = numeric_series.mean()
        median = numeric_series.median()
        std = numeric_series.std()
        iqr = numeric_series.quantile(0.75) - numeric_series.quantile(0.25)

        distorted = abs(mean - median) > (0.5 * std if std > 0 else 0)
        unstable = std > iqr if iqr > 0 else False

        if miss_risk == "High" or distorted or unstable:
            row["trust"] = "High Risk"
            row["remarks"] = "Outliers or unstable values"
        elif miss_risk == "Medium":
            row["trust"] = "Needs Cleaning"
            row["remarks"] = "Some missing values"
        else:
            row["trust"] = "Reliable"
            row["remarks"] = "Looks consistent"

        results.append(row)

    result_df = pd.DataFrame(results)

    # -----------------------------
    # Save result to processed folder
    # -----------------------------
    os.makedirs("processed", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join("processed", f"{base_name}_result.csv")

    result_df.to_csv(output_path, index=False)

    return result_df, output_path


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    path = r"C:\Users\ASUS\Documents\inomatics\sample.csv"
    df_result, saved_path = analyze_dataset(path)

    print("\n✅ Data Trust Analysis Completed")
    print(f"📁 Result saved at: {saved_path}")
    print("\nPreview:")
    print(df_result.head())
