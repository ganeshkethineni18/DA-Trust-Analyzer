# 📊 Data Trust Analyzer

A Python-based tool that evaluates dataset reliability **before analysis or modeling**.

---

## 🔍 Why This Project?

In real-world data analysis, datasets often contain:

- Missing values
- Skewed distributions
- Outliers
- Unstable columns
- Mixed data types

If these issues are ignored, analysis results become misleading.

This project ensures that a dataset is statistically reliable before proceeding to further analysis.

---

## ⚙️ Features

- ✅ Missing value detection (count + percentage)
- ✅ Identifier column detection
- ✅ Skewness-based distortion detection
- ✅ Instability detection using standard deviation vs IQR
- ✅ Outlier detection using IQR method
- ✅ Numeric and categorical validation
- ✅ Column-level trust classification
- ✅ Dataset-level trust verdict
- ✅ Optional CSV report export

---

## 🧠 How It Works

Each column is analyzed independently:

1. Detect missing values
2. Validate data type
3. Compute:
   - Mean
   - Median
   - Standard Deviation
   - IQR
   - Skewness
4. Detect outliers using IQR rule
5. Classify column as:
   - Reliable
   - Needs Cleaning
   - High Risk

Finally, an overall dataset verdict is generated.

---

## 📁 Project Structure

