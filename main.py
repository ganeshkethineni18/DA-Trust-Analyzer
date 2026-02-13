import pandas as pd
import numpy as np
import os
import re


def is_index_like(series, row_count):
    if not pd.api.types.is_numeric_dtype(series):
        return False

    if series.nunique() < row_count * 0.9:
        return False

    diffs = series.dropna().sort_values().diff().dropna()
    return diffs.nunique() == 1


def clean_numeric_text(series):
    # Remove currency symbols, commas, percentage signs
    return series.astype(str).str.replace(r"[,$%]", "", regex=True)


def probe_numeric(series):
    cleaned = clean_numeric_text(series)
    converted = pd.to_numeric(cleaned, errors="coerce")
    success_rate = converted.notna().mean()
    return converted, success_rate


def analyze_dataset(csv_path):

    df = pd.read_csv(csv_path)
    rows = len(df)

    results = []

    for col in df.columns:
        series = df[col]

        missing_count = series.isna().sum()
        missing_percent = round((missing_count / rows) * 100, 2)

        row = {
            "column": col,
            "data_type": "",
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "distorted": False,
            "unstable": False,
            "outlier_count": 0,
            "trust": "",
            "remarks": ""
        }

        # Identifier detection
        if is_index_like(series, rows):
            row["data_type"] = "Identifier"
            row["trust"] = "Ignored"
            row["remarks"] = "Serial number or ID column"
            results.append(row)
            continue

        # Numeric detection
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = series.dropna()
            row["data_type"] = "Numeric"

        elif pd.api.types.is_object_dtype(series):
            converted, success = probe_numeric(series)

            if success >= 0.9:
                numeric_series = converted.dropna()
                row["data_type"] = "Numeric (from text)"
                row["remarks"] = "Converted text to numbers"
            else:
                row["data_type"] = "Categorical"

                if missing_percent >= 30:
                    row["trust"] = "High Risk"
                elif missing_percent >= 5:
                    row["trust"] = "Needs Cleaning"
                else:
                    row["trust"] = "Reliable"

                row["remarks"] = "Category data"
                results.append(row)
                continue
        else:
            row["data_type"] = "Other"
            row["trust"] = "Needs Cleaning"
            row["remarks"] = "Unknown data type"
            results.append(row)
            continue

        # If numeric but all values missing
        if len(numeric_series) == 0:
            row["trust"] = "High Risk"
            row["remarks"] = "All values missing"
            results.append(row)
            continue

        # 📊 Statistical Checks
        mean = numeric_series.mean()
        median = numeric_series.median()
        std = numeric_series.std()
        q1 = numeric_series.quantile(0.25)
        q3 = numeric_series.quantile(0.75)
        iqr = q3 - q1

        # Skewness (better distortion detection)
        skewness = numeric_series.skew()
        distorted = abs(skewness) > 0.5

        # Instability check
        unstable = std > iqr if iqr > 0 else False

        # Outlier detection (IQR rule)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = numeric_series[(numeric_series < lower_bound) | (numeric_series > upper_bound)]
        outlier_count = len(outliers)

        row["distorted"] = distorted
        row["unstable"] = unstable
        row["outlier_count"] = outlier_count

        # Trust logic
        if missing_percent >= 30:
            row["trust"] = "High Risk"
            row["remarks"] = "Too many missing values"

        elif outlier_count > rows * 0.1:
            row["trust"] = "High Risk"
            row["remarks"] = "High number of outliers"

        elif distorted or unstable or missing_percent >= 5:
            row["trust"] = "Needs Cleaning"
            row["remarks"] = "Distribution issues detected"

        else:
            row["trust"] = "Reliable"
            row["remarks"] = "Looks consistent"

        results.append(row)

    result_df = pd.DataFrame(results)

    high_risk_count = (result_df["trust"] == "High Risk").sum()

    if high_risk_count > len(result_df) * 0.4:
        verdict = "Dataset is NOT reliable"
    elif high_risk_count > 0:
        verdict = "Dataset needs cleaning"
    else:
        verdict = "Dataset looks safe"

    return result_df, verdict


if __name__ == "__main__":

    path = r"heavy_sample.csv"

    df_result, verdict = analyze_dataset(path)

    print("\nData Trust Analysis Completed\n")
    print(df_result)
    print("\nFinal Verdict:", verdict)

    choice = input("\nExport result as CSV? (yes/no): ").strip().lower()

    if choice in ["yes", "y"]:
        os.makedirs("processed", exist_ok=True)

        base_name = os.path.splitext(os.path.basename(path))[0]
        output_path = os.path.join("processed", f"{base_name}_result.csv")

        df_result.to_csv(output_path, index=False)
        print(f"\nResult saved at: {output_path}")
    else:
        print("\nResult not saved.")
