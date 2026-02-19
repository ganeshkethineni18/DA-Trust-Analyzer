


import pandas as pd
import numpy as np
import os



def check_if_identifier(col_data, total_rows):

    if not pd.api.types.is_numeric_dtype(col_data):
        return False


    if col_data.nunique() < total_rows * 0.9:
        return False

    
    sorted_values = col_data.dropna().sort_values()
    differences = sorted_values.diff().dropna()

    if differences.nunique() == 1:
        return True

    return False


def convert_text_to_number(col_data):

    cleaned = col_data.astype(str)
    cleaned = cleaned.str.replace("$", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)

    numeric_version = pd.to_numeric(cleaned, errors="coerce")
    conversion_success = numeric_version.notna().mean()

    return numeric_version, conversion_success


def analyze_one_column(col_name, col_data, total_rows):

    missing_count = col_data.isna().sum()
    missing_percent = (missing_count / total_rows) * 100

    column_type = ""
    is_distorted = False
    is_unstable = False
    number_of_outliers = 0
    trust_level = ""
    notes = ""

    if check_if_identifier(col_data, total_rows):
        return [
            col_name,
            "Identifier",
            missing_count,
            round(missing_percent, 2),
            False,
            False,
            0,
            "Ignored",
            "Identifier column"
        ]

    if pd.api.types.is_numeric_dtype(col_data):
        numeric_values = col_data.dropna()
        column_type = "Numeric"

    else:
        converted_values, success_rate = convert_text_to_number(col_data)

        if success_rate >= 0.9:
            numeric_values = converted_values.dropna()
            column_type = "Numeric (converted)"
        else:
            column_type = "Categorical"

            if missing_percent >= 30:
                trust_level = "High Risk"
            elif missing_percent >= 5:
                trust_level = "Needs Cleaning"
            else:
                trust_level = "Reliable"

            notes = "Categorical column"

            return [
                col_name,
                column_type,
                missing_count,
                round(missing_percent, 2),
                False,
                False,
                0,
                trust_level,
                notes
            ]

    if len(numeric_values) == 0:
        return [
            col_name,
            column_type,
            missing_count,
            round(missing_percent, 2),
            False,
            False,
            0,
            "High Risk",
            "All values missing"
        ]
    
    std_value = numeric_values.std()
    q1 = numeric_values.quantile(0.25)
    q3 = numeric_values.quantile(0.75)
    iqr_value = q3 - q1
    skew_value = numeric_values.skew()

    if abs(skew_value) > 0.5:
        is_distorted = True

    if iqr_value > 0 and std_value > iqr_value:
        is_unstable = True

    lower_limit = q1 - 1.5 * iqr_value
    upper_limit = q3 + 1.5 * iqr_value

    outliers = numeric_values[
        (numeric_values < lower_limit) |
        (numeric_values > upper_limit)
    ]

    number_of_outliers = len(outliers)


    if missing_percent >= 30:
        trust_level = "High Risk"
        notes = "Too many missing values"

    elif number_of_outliers > total_rows * 0.1:
        trust_level = "High Risk"
        notes = "Too many outliers"

    elif is_distorted or is_unstable or missing_percent >= 5:
        trust_level = "Needs Cleaning"
        notes = "Distribution issues detected"

    else:
        trust_level = "Reliable"
        notes = "Column is reliable"

    return [
        col_name,
        column_type,
        missing_count,
        round(missing_percent, 2),
        is_distorted,
        is_unstable,
        number_of_outliers,
        trust_level,
        notes
    ]

def analyze_dataset(file_path):

    data = pd.read_csv(file_path)
    total_rows = len(data)

    final_report = []

    for column_name in data.columns:
        column_result = analyze_one_column(
            column_name,
            data[column_name],
            total_rows
        )
        final_report.append(column_result)

    report_df = pd.DataFrame(final_report, columns=[
        "column",
        "data_type",
        "missing_count",
        "missing_percent",
        "distorted",
        "unstable",
        "outlier_count",
        "trust",
        "remarks"
    ])

    high_risk_columns = (report_df["trust"] == "High Risk").sum()

    if high_risk_columns > len(report_df) * 0.4:
        overall_verdict = "Dataset is NOT reliable"
    elif high_risk_columns > 0:
        overall_verdict = "Dataset needs cleaning"
    else:
        overall_verdict = "Dataset is safe for analysis"

    return report_df, overall_verdict

if __name__ == "__main__":

    FILE_PATH = "numerical_cleaning_advanced.csv"

    results_table, final_verdict = analyze_dataset(FILE_PATH)

    print("\n=== DATA TRUST ANALYSIS REPORT ===\n")
    print(results_table)
    print("\nFinal Verdict:", final_verdict)

    user_choice = input("\nExport report as CSV? (yes/no): ").lower()

    if user_choice == "yes":
        os.makedirs("processed", exist_ok=True)
        results_table.to_csv("processed/data_trust_report.csv", index=False)
        print("\nReport saved to processed/data_trust_report.csv")
    else:
        print("\nExport skipped.")
