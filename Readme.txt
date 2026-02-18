# 📊 Data Trust Analyzer

## Overview

This project is a Python-based Data Trust Analyzer that evaluates the reliability of a dataset before performing analysis or modeling. It checks for missing values, distribution distortion, instability, and outliers using statistical methods.

The goal is to ensure that data is trustworthy before drawing conclusions from it.

---

## Features

- Detects missing values (count and percentage)
- Identifies identifier/serial number columns
- Converts numeric text (e.g., "$5000", "45%") into proper numbers
- Detects skewed distributions using skewness
- Detects instability using Standard Deviation and IQR comparison
- Detects outliers using the IQR rule
- Classifies each column as:
  - Reliable
  - Needs Cleaning
  - High Risk
- Provides overall dataset trust verdict
- Option to export results as CSV

---

## Concepts Used

- File Handling
- Data Preprocessing
- Descriptive Statistics
- Skewness Analysis
- IQR Method (Outlier Detection)
- Exception Handling
- Structured Functional Design
- Conditional Logic

---

## Technologies Used

- Language: Python
- Libraries: Pandas, NumPy
- Programming Style: Modular & Functional Design
- Platform: Cross-platform (Windows/Linux/macOS)

---

## How to Run

1. Install Python (version 3.x)
2. Install required libraries:
3. Run Main.Py

