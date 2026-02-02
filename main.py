import pandas as pd
import numpy as np

# -----------------------------
# Helper Functions
# -----------------------------

def is_index_like(series, row_count):
    """Detect index / serial number columns"""
    if not pd.api.types.is_numeric_dtype(series):
        return False

    unique_count = series.nunique()
    if unique_count < row_count * 0.9:
        return False

    sorted_vals = series.dropna().sort_values()
    diffs = sorted_vals.diff().dropna()

    return diffs.nunique() == 1


def probe_numeric(series):
    """Try converting object column to numeric safely"""
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
    report = []
    column_results = []

    report.append("📊 DATA TRUST ANALYSIS REPORT\n")
    report.append(f"Rows: {rows}")
    report.append(f"Columns: {cols}\n")

    for col in df.columns:
        col_info = {
            "column": col,
            "type": None,
            "trust": None,
            "notes": []
        }

        series = df[col]

        # -------------------------
        # Step 1: Index-like check
        # -------------------------
        if is_index_like(series, rows):
            col_info["type"] = "Identifier"
            col_info["trust"] = "Ignored"
            col_info["notes"].append("Index-like column (excluded from stats)")
            column_results.append(col_info)
            continue

        # -------------------------
        # Step 2: Missing values
        # -------------------------
        missing_pct = series.isna().mean() * 100
        miss_risk = missing_risk(missing_pct)

        col_info["notes"].append(f"Missing: {missing_pct:.2f}% ({miss_risk})")

        # -------------------------
        # Step 3: Type handling
        # -------------------------
        if pd.api.types.is_numeric_dtype(series):
            col_info["type"] = "Numeric"
            numeric_series = series

        elif pd.api.types.is_object_dtype(series):
            converted, success = probe_numeric(series)

            if success >= 0.9:
                col_info["type"] = "Numeric (from object)"
                numeric_series = converted
                col_info["notes"].append("Object column safely converted to numeric")
            else:
                col_info["type"] = "Categorical"
                col_info["trust"] = "Needs Cleaning" if miss_risk != "Low" else "Reliable"
                column_results.append(col_info)
                continue
        else:
            col_info["type"] = "Other"
            col_info["trust"] = "Needs Cleaning"
            column_results.append(col_info)
            continue

        # -------------------------
        # Step 4: Statistics
        # -------------------------
        mean = numeric_series.mean()
        median = numeric_series.median()
        std = numeric_series.std()

        q1 = numeric_series.quantile(0.25)
        q3 = numeric_series.quantile(0.75)
        iqr = q3 - q1

        col_info["notes"].append(f"Mean: {mean:.3f}, Median: {median:.3f}")
        col_info["notes"].append(f"STD: {std:.3f}, IQR: {iqr:.3f}")

        # -------------------------
        # Step 5: Distortion checks
        # -------------------------
        distortion = abs(mean - median)

        high_distortion = distortion > (0.5 * std if std > 0 else 0)
        unstable = std > iqr if iqr > 0 else False

        if mean > median:
            skew = "Right"
        elif mean < median:
            skew = "Left"
        else:
            skew = "Symmetric"

        col_info["notes"].append(f"Skewness: {skew}")

        # -------------------------
        # Step 6: Trust decision
        # -------------------------
        if miss_risk == "High" or high_distortion or unstable:
            col_info["trust"] = "High Risk"
        elif miss_risk == "Medium":
            col_info["trust"] = "Needs Cleaning"
        else:
            col_info["trust"] = "Reliable"

        column_results.append(col_info)

    # -----------------------------
    # Dataset-level verdict
    # -----------------------------
    high_risk_count = sum(1 for c in column_results if c["trust"] == "High Risk")

    if high_risk_count > len(column_results) * 0.4:
        verdict = "❌ NOT RELIABLE"
    elif high_risk_count > 0:
        verdict = "⚠️ REQUIRES CLEANING"
    else:
        verdict = "✅ SAFE FOR ANALYSIS"

    # -----------------------------
    # Report Generation
    # -----------------------------
    report.append("COLUMN ANALYSIS:\n")

    for c in column_results:
        report.append(f"• {c['column']}")
        report.append(f"  Type: {c['type']}")
        report.append(f"  Trust: {c['trust']}")
        for note in c["notes"]:
            report.append(f"  - {note}")
        report.append("")

    report.append(f"\nFINAL VERDICT: {verdict}")

    return "\n".join(report)


# -----------------------------
# Run Example
# -----------------------------
if __name__ == "__main__":
    path = "C:\\Users\\ASUS\\Documents\\inomatics\\sample.csv"   # <-- replace with your CSV path
    print(analyze_dataset(path))
