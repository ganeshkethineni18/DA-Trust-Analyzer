"""
Data Trust Analyzer

Evaluates dataset reliability before analysis by detecting:
- Missing values
- Distribution distortion (skewness)
- Instability (STD vs IQR)
- Outliers (IQR method)
- Identifier columns
"""

import pandas as pd
import numpy as np
import os


# ------------------------------------------------------------
# Identifier Detection
# ------------------------------------------------------------
def is_index_like(column, total_rows):
    if not pd.api.types.is_numeric_dtype(column):
        return False

    if column.nunique() < total_rows * 0.9:
        return False

    diffs = column.dropna().sort_values().diff().dropna()
    return diffs.nunique() == 1


# ------------------------------------------------------------
# Safe Numeric Conversion
# ------------------------------------------------------------
def try_convert_to_number(column):
    cleaned = (
        column.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )

    converted = pd.to_numeric(cleaned, errors="coerce")
    return converted, converted.notna().mean()


# ------------------------------------------------------------
# Core Analyzer
# ------------------------------------------------------------
def analyze_dataset(file_path):

    df = pd.read_csv(file_path)
    total_rows = len(df)

    results = []

    for column_name in df.columns:

        column = df[column_name]
        missing_count = column.isna().sum()
        missing_percent = (missing_count / total_rows) * 100

        result = {
            "column": column_name,
            "data_type": "",
            "missing_count": missing_count,
            "missing_percent": round(missing_percent, 2),
            "distorted": False,
            "unstable": False,
            "outlier_count": 0,
            "trust": "",
            "remarks": ""
        }

        # Identifier
        if is_index_like(column, total_rows):
            result.update({
                "data_type": "Identifier",
                "trust": "Ignored",
                "remarks": "Identifier column"
            })
            results.append(result)
            continue

        # Numeric Detection
        if pd.api.types.is_numeric_dtype(column):
            numeric_values = column.dropna()
            result["data_type"] = "Numeric"
        else:
            converted, success = try_convert_to_number(column)

            if success >= 0.9:
                numeric_values = converted.dropna()
                result["data_type"] = "Numeric (converted)"
            else:
                result["data_type"] = "Categorical"
                result["trust"] = (
                    "High Risk" if missing_percent >= 30
                    else "Needs Cleaning" if missing_percent >= 5
                    else "Reliable"
                )
                result["remarks"] = "Categorical column"
                results.append(result)
                continue

        # Empty Numeric Column
        if len(numeric_values) == 0:
            result.update({
                "trust": "High Risk",
                "remarks": "All values missing"
            })
            results.append(result)
            continue

        # Statistical Checks
        std = numeric_values.std()
        q1 = numeric_values.quantile(0.25)
        q3 = numeric_values.quantile(0.75)
        iqr = q3 - q1
        skew = numeric_values.skew()

        result["distorted"] = abs(skew) > 0.5
        result["unstable"] = iqr > 0 and std > iqr

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = numeric_values[
            (numeric_values < lower) | (numeric_values > upper)
        ]
        result["outlier_count"] = len(outliers)

        # Trust Classification
        if missing_percent >= 30:
            result["trust"] = "High Risk"
            result["remarks"] = "Too many missing values"

        elif len(outliers) > total_rows * 0.1:
            result["trust"] = "High Risk"
            result["remarks"] = "Excessive outliers"

        elif result["distorted"] or result["unstable"] or missing_percent >= 5:
            result["trust"] = "Needs Cleaning"
            result["remarks"] = "Distribution issues detected"

        else:
            result["trust"] = "Reliable"
            result["remarks"] = "Column is reliable"

        results.append(result)

    results_df = pd.DataFrame(results)

    high_risk_count = (results_df["trust"] == "High Risk").sum()

    if high_risk_count > len(results_df) * 0.4:
        verdict = "Dataset is NOT reliable"
    elif high_risk_count > 0:
        verdict = "Dataset needs cleaning"
    else:
        verdict = "Dataset is safe for analysis"

    return results_df, verdict


# ------------------------------------------------------------
# Run Script
# ------------------------------------------------------------
if __name__ == "__main__":

    FILE_PATH = "numerical_cleaning_advanced.csv"

    results, verdict = analyze_dataset(FILE_PATH)

    print("\n=== DATA TRUST ANALYSIS REPORT ===\n")
    print(results)
    print("\nFinal Verdict:", verdict)

    save = input("\nExport report as CSV? (yes/no): ").lower()

    if save == "yes":
        os.makedirs("processed", exist_ok=True)
        results.to_csv("processed/data_trust_report.csv", index=False)
        print("\nReport saved to processed/data_trust_report.csv")
    else:
        print("\nExport skipped.")
