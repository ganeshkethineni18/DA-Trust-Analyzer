"""
Data Trust Analyzer
-------------------

This tool evaluates the reliability of a dataset before analysis.

It detects:
- Missing values
- Data type validity
- Distribution distortion (skewness)
- Data instability
- Outliers using IQR method
- Identifier columns

Output:
Column-level trust report and overall dataset verdict.

Author: Kethineni Venkata Ganesh
"""

import pandas as pd
import numpy as np
import os


# ------------------------------------------------------------
# Helper Function: Detect Identifier Columns
# ------------------------------------------------------------
def is_index_like(column: pd.Series, total_rows: int) -> bool:
    """
    Detects if a column behaves like a serial number or ID column.
    """

    if not pd.api.types.is_numeric_dtype(column):
        return False

    unique_values = column.nunique()

    # Identifier columns usually have unique values close to total rows
    if unique_values < total_rows * 0.9:
        return False

    sorted_values = column.dropna().sort_values()
    differences = sorted_values.diff().dropna()

    # Sequential difference indicates index-like column
    return differences.nunique() == 1


# ------------------------------------------------------------
# Helper Function: Convert Text to Numeric Safely
# ------------------------------------------------------------
def try_convert_to_number(column: pd.Series):
    """
    Attempts to convert text-based numeric columns to proper numeric type.
    Removes common symbols like $, commas, and %.
    """

    cleaned = column.astype(str)
    cleaned = cleaned.str.replace("$", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)

    converted = pd.to_numeric(cleaned, errors="coerce")

    success_rate = converted.notna().mean()

    return converted, success_rate


# ------------------------------------------------------------
# Core Function: Analyze Dataset Trust
# ------------------------------------------------------------
def analyze_dataset(file_path: str):
    """
    Performs trust analysis on dataset.

    Returns:
        DataFrame with column-level analysis
        Overall dataset trust verdict
    """

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

        # ----------------------------------------------------
        # Identifier detection
        # ----------------------------------------------------
        if is_index_like(column, total_rows):

            result["data_type"] = "Identifier"
            result["trust"] = "Ignored"
            result["remarks"] = "Identifier column"

            results.append(result)
            continue

        # ----------------------------------------------------
        # Numeric detection
        # ----------------------------------------------------
        if pd.api.types.is_numeric_dtype(column):

            numeric_values = column.dropna()
            result["data_type"] = "Numeric"

        else:

            converted_column, success_rate = try_convert_to_number(column)

            if success_rate >= 0.9:

                numeric_values = converted_column.dropna()
                result["data_type"] = "Numeric (converted)"

            else:

                result["data_type"] = "Categorical"

                if missing_percent >= 30:
                    result["trust"] = "High Risk"
                elif missing_percent >= 5:
                    result["trust"] = "Needs Cleaning"
                else:
                    result["trust"] = "Reliable"

                result["remarks"] = "Categorical column"

                results.append(result)
                continue

        # ----------------------------------------------------
        # Handle empty numeric columns
        # ----------------------------------------------------
        if len(numeric_values) == 0:

            result["trust"] = "High Risk"
            result["remarks"] = "All values missing"

            results.append(result)
            continue

        # ----------------------------------------------------
        # Statistical analysis
        # ----------------------------------------------------
        mean_value = numeric_values.mean()
        median_value = numeric_values.median()
        std_dev = numeric_values.std()

        q1 = numeric_values.quantile(0.25)
        q3 = numeric_values.quantile(0.75)
        iqr = q3 - q1

        skew_value = numeric_values.skew()

        # Distortion detection
        if abs(skew_value) > 0.5:
            result["distorted"] = True

        # Instability detection
        if iqr > 0 and std_dev > iqr:
            result["unstable"] = True

        # Outlier detection using IQR rule
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = numeric_values[
            (numeric_values < lower_bound) |
            (numeric_values > upper_bound)
        ]

        result["outlier_count"] = len(outliers)

        # ----------------------------------------------------
        # Trust classification
        # ----------------------------------------------------
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

    # ----------------------------------------------------
    # Dataset-level verdict
    # ----------------------------------------------------
    high_risk_count = (results_df["trust"] == "High Risk").sum()

    if high_risk_count > len(results_df) * 0.4:
        verdict = "Dataset is NOT reliable"

    elif high_risk_count > 0:
        verdict = "Dataset needs cleaning"

    else:
        verdict = "Dataset is safe for analysis"

    return results_df, verdict


# ------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":

    DATA_FILE = "heavy_sample.csv"

    analysis_result, dataset_verdict = analyze_dataset(DATA_FILE)

    print("\n=== DATA TRUST ANALYSIS REPORT ===\n")
    print(analysis_result)

    print("\nFinal Verdict:", dataset_verdict)

    # Optional export
    save = input("\nExport report to CSV? (yes/no): ").lower()

    if save == "yes":

        os.makedirs("processed", exist_ok=True)

        output_path = "processed/data_trust_report.csv"

        analysis_result.to_csv(output_path, index=False)

        print("\nReport saved at:", output_path)

    else:
        print("\nExport skipped.")
