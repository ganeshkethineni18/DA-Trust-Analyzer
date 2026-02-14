import pandas as pd
import numpy as np
import os


def is_index_like(column, total_rows):

    if not pd.api.types.is_numeric_dtype(column):
        return False

    unique_values = column.nunique()

    if unique_values < total_rows * 0.9:
        return False

    sorted_values = column.dropna().sort_values()
    differences = sorted_values.diff().dropna()

    if differences.nunique() == 1:
        return True

    return False


def try_convert_to_number(column):

 # Remove currency symbols and percentage signs

    cleaned_column = column.astype(str)
    cleaned_column = cleaned_column.str.replace("$", "", regex=False)
    cleaned_column = cleaned_column.str.replace(",", "", regex=False)
    cleaned_column = cleaned_column.str.replace("%", "", regex=False)

    converted_column = pd.to_numeric(cleaned_column, errors="coerce")

    success_rate = converted_column.notna().mean()

    return converted_column, success_rate


def analyze_dataset(file_path):

    df = pd.read_csv(file_path)

    total_rows = len(df)

    final_results = []

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

# Check identifier

        if is_index_like(column, total_rows):
            result["data_type"] = "Identifier"
            result["trust"] = "Ignored"
            result["remarks"] = "Serial number column"
            final_results.append(result)
            continue

# Check numeric

        if pd.api.types.is_numeric_dtype(column):

            numeric_values = column.dropna()
            result["data_type"] = "Numeric"

        else:
            converted_column, success = try_convert_to_number(column)

            if success >= 0.9:
                numeric_values = converted_column.dropna()
                result["data_type"] = "Numeric (from text)"
                result["remarks"] = "Converted from text"
            else:
                result["data_type"] = "Categorical"

                if missing_percent >= 30:
                    result["trust"] = "High Risk"
                elif missing_percent >= 5:
                    result["trust"] = "Needs Cleaning"
                else:
                    result["trust"] = "Reliable"

                result["remarks"] = "Category data"
                final_results.append(result)
                continue

# If numeric but empty

        if len(numeric_values) == 0:
            result["trust"] = "High Risk"
            result["remarks"] = "All values missing"
            final_results.append(result)
            continue

# Basic statistics

        mean_value = numeric_values.mean()
        median_value = numeric_values.median()
        standard_deviation = numeric_values.std()

        q1 = numeric_values.quantile(0.25)
        q3 = numeric_values.quantile(0.75)
        iqr = q3 - q1

# Check skewness

        skew_value = numeric_values.skew()

        if abs(skew_value) > 0.5:
            result["distorted"] = True

# Check instability

        if iqr > 0 and standard_deviation > iqr:
            result["unstable"] = True

# Outlier detection

        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        outliers = numeric_values[
            (numeric_values < lower_limit) |
            (numeric_values > upper_limit)
        ]

        result["outlier_count"] = len(outliers)

# Final trust decision

        if missing_percent >= 30:
            result["trust"] = "High Risk"
            result["remarks"] = "Too many missing values"

        elif len(outliers) > total_rows * 0.1:
            result["trust"] = "High Risk"
            result["remarks"] = "Too many outliers"

        elif result["distorted"] or result["unstable"] or missing_percent >= 5:
            result["trust"] = "Needs Cleaning"
            result["remarks"] = "Distribution issues detected"

        else:
            result["trust"] = "Reliable"
            result["remarks"] = "Looks consistent"

        final_results.append(result)

    results_dataframe = pd.DataFrame(final_results)

    high_risk_columns = (results_dataframe["trust"] == "High Risk").sum()

    if high_risk_columns > len(results_dataframe) * 0.4:
        final_verdict = "Dataset is NOT reliable"
    elif high_risk_columns > 0:
        final_verdict = "Dataset needs cleaning"
    else:
        final_verdict = "Dataset looks safe"

    return results_dataframe, final_verdict


if __name__ == "__main__":

    file_path = "heavy_sample.csv"

    results, verdict = analyze_dataset(file_path)

    print("\nData Trust Analysis Completed\n")
    print(results)
    print("\nFinal Verdict:", verdict)

    save_option = input("\nExport result as CSV? (yes/no): ").lower()

    if save_option == "yes":
        os.makedirs("processed", exist_ok=True)
        output_file = "processed/result.csv"
        results.to_csv(output_file, index=False)
        print("\nSaved to:", output_file)
    else:
        print("\nResult not saved.")
