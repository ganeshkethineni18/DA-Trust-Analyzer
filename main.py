import numpy as np
import pandas as pd


def generate_heavy_dataset(
    rows=50000,
    missing_rate=0.08,
    outlier_rate=0.03,
    duplicate_rate=0.02,
    seed=42,
    output_file="heavy_sample.csv"
):
    """
    Generates a large, messy dataset to test Data Trust Analyzer.

    rows           → number of rows
    missing_rate   → percentage of missing values
    outlier_rate   → percentage of extreme values
    duplicate_rate → percentage of duplicate rows
    """

    np.random.seed(seed)

    # -------------------------------------------------
    # 1️⃣ Create Identifier Column (Serial Number)
    # -------------------------------------------------
    s_no = np.arange(1, rows + 1)

    # -------------------------------------------------
    # 2️⃣ Create Numeric Columns
    # -------------------------------------------------

    # Normal distributed salary
    salary = np.random.normal(loc=60000, scale=15000, size=rows)

    # Bonus depends on salary (correlated column)
    bonus = salary * np.random.uniform(0.05, 0.2, rows)

    # Total income derived from salary + bonus
    total_income = salary + bonus

    # Age between 18–65
    age = np.random.randint(18, 65, rows).astype(float)

    # Expenses with skewed distribution
    expenses = np.random.exponential(scale=20000, size=rows)

    # -------------------------------------------------
    # 3️⃣ Inject Outliers
    # -------------------------------------------------
    outlier_count = int(rows * outlier_rate)
    outlier_indices = np.random.choice(rows, outlier_count, replace=False)

    salary[outlier_indices] *= 6
    expenses[outlier_indices] *= 8

    # -------------------------------------------------
    # 4️⃣ Categorical Columns
    # -------------------------------------------------
    gender = np.random.choice(["Male", "Female", "Other"], rows)

    department = np.random.choice(
        ["HR", "Engineering", "Sales", "Marketing", "Finance", "Support"],
        rows
    )

    # High-cardinality city column
    city = np.random.choice([f"City_{i}" for i in range(500)], rows)

    # -------------------------------------------------
    # 5️⃣ Numeric Stored as Text (Dirty Column)
    # -------------------------------------------------
    rating_numeric = np.random.randint(1, 6, rows)
    rating_text = rating_numeric.astype(str)

    # Add noise to rating column
    noise_indices = np.random.choice(rows, int(rows * 0.05), replace=False)
    rating_text[noise_indices] = "unknown"

    # -------------------------------------------------
    # 6️⃣ Inject Missing Values
    # -------------------------------------------------
    def inject_missing(array):
        mask = np.random.rand(rows) < missing_rate
        array = array.astype(object)
        array[mask] = np.nan
        return array

    age = inject_missing(age)
    salary = inject_missing(salary)
    bonus = inject_missing(bonus)
    total_income = inject_missing(total_income)
    expenses = inject_missing(expenses)
    gender = inject_missing(gender)

    # -------------------------------------------------
    # 7️⃣ Build Final DataFrame
    # -------------------------------------------------
    df = pd.DataFrame({
        "S.No": s_no,
        "Age": age,
        "Salary": salary,
        "Bonus": bonus,
        "Total_Income": total_income,
        "Expenses": expenses,
        "Gender": gender,
        "Department": department,
        "City": city,
        "Rating": rating_text
    })

    # -------------------------------------------------
    # 8️⃣ Add Duplicate Rows
    # -------------------------------------------------
    duplicate_count = int(rows * duplicate_rate)
    duplicates = df.sample(duplicate_count)
    df = pd.concat([df, duplicates], ignore_index=True)

    # -------------------------------------------------
    # 9️⃣ Save Dataset
    # -------------------------------------------------
    df.to_csv(output_file, index=False)

    print("\n✅ Heavy dataset generated successfully")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Saved as: {output_file}")
    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    generate_heavy_dataset(rows=50000)
