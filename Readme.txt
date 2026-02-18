📊 Data Trust Analyzer
Overview

This project is a Python-based Data Trust Analyzer that evaluates the reliability of a dataset before performing analysis or modeling. It checks for missing values, distribution distortion, instability, and outliers using statistical methods.

The goal is to ensure that data is trustworthy before drawing conclusions from it.

Features

Detects missing values (count and percentage)

Identifies identifier/serial number columns

Converts numeric text (e.g., "$5000", "45%") into proper numbers

Detects skewed distributions using skewness

Detects instability using Standard Deviation and IQR comparison

Detects outliers using the IQR rule

Classifies each column as:

Reliable

Needs Cleaning

High Risk

Provides overall dataset trust verdict

Option to export results as CSV

Concepts Used

File Handling

Data Preprocessing

Descriptive Statistics

Skewness Analysis

IQR Method (Outlier Detection)

Exception Handling

Object-Oriented Thinking (Structured Functions)

Conditional Logic

Technologies Used

Language: Python

Libraries: Pandas, NumPy

Programming Style: Modular & Functional Design

Platform: Cross-platform (Windows/Linux/macOS)

How to Run

Install Python (version 3.x)

Install required libraries:

pip install pandas numpy


Place your dataset CSV file in the project folder

Run the program:

python data_trust_analyzer.py


View the column-level trust report in the terminal

Optionally export the report as a CSV file

Use Case

This project is useful for:

Data Analysts who want to validate dataset reliability

Students learning EDA and preprocessing

Anyone who wants to understand how statistics detect data issues

Pre-modeling data validation workflows

Author

Kethineni Venkata Ganesh
Aspiring Data Analyst | Python & SQL Enthusiast

License

This project is created for educational and portfolio purposes.