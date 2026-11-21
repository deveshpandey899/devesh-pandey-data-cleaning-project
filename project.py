import pandas as pd 
import numpy as np

df = pd.read_csv(r"C:\Users\harsh\Desktop\devesh vs\indian_healthcare_patients_10k.csv")
# print(df)

df.info()

import pandas as pd
import numpy as np

# Load CSV and convert hidden string nulls to actual NaN
df = pd.read_csv(
    r"C:\Users\harsh\Desktop\devesh vs\indian_healthcare_patients_10k.csv",
    na_values=["", "NA", "N/A", "null", "None", "?", "-"]
)

print("1. MISSING VALUES:\n", df.isnull().sum())
print("\n2. DUPLICATE ROWS:", df.duplicated().sum())
print("3. DUPLICATE PATIENT IDs:", df['Patient_ID'].duplicated().sum())
print("\n4. STATISTICAL SUMMARY (Check min/max for weird numbers):\n", df.describe())

# Check unique values in categorical columns for messy spelling/casing
for col in ['Gender', 'Blood_Group', 'Admission_Type', 'Outcome']:
    print(f"\nUnique in {col}:", df[col].unique())
    
# Remove exact duplicates and duplicate Patient_IDs
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=['Patient_ID'], keep='first', inplace=True)

# Clean whitespace and restore NaNs in text columns
text_cols = [
    'Patient_ID', 'Gender', 'Blood_Group', 'City', 
    'Department', 'Primary_Diagnosis', 'Admission_Type', 
    'Insurance_Provider', 'Outcome'
]

for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({'nan': np.nan, 'None': np.nan})

# Fix capitalization
df['Patient_ID'] = df['Patient_ID'].str.upper()
df['Blood_Group'] = df['Blood_Group'].str.upper()

title_cols = ['Gender', 'City', 'Department', 'Primary_Diagnosis', 'Admission_Type', 'Insurance_Provider', 'Outcome']
for col in title_cols:
    df[col] = df[col].str.title()

print("\n--- AFTER STAGE 2 (CLEANED CATEGORIES) ---")
for col in ['Gender', 'Blood_Group', 'Admission_Type', 'Outcome']:
    print(f"Unique in {col}:", df[col].unique())
    
# Remove exact duplicates and duplicate Patient_IDs
df.drop_duplicates(inplace=True)
df.drop_duplicates(subset=['Patient_ID'], keep='first', inplace=True)

# Clean whitespace and restore NaNs in text columns
text_cols = [
    'Patient_ID', 'Gender', 'Blood_Group', 'City', 
    'Department', 'Primary_Diagnosis', 'Admission_Type', 
    'Insurance_Provider', 'Outcome'
]

for col in text_cols:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace({'nan': np.nan, 'None': np.nan})

# Fix capitalization
df['Patient_ID'] = df['Patient_ID'].str.upper()
df['Blood_Group'] = df['Blood_Group'].str.upper()

title_cols = ['Gender', 'City', 'Department', 'Primary_Diagnosis', 'Admission_Type', 'Insurance_Provider', 'Outcome']
for col in title_cols:
    df[col] = df[col].str.title()

print("\n--- AFTER STAGE 2 (CLEANED CATEGORIES) ---")
for col in ['Gender', 'Blood_Group', 'Admission_Type', 'Outcome']:
    print(f"Unique in {col}:", df[col].unique())
    
# Fill missing Outcome with 'Unknown' and other text columns with mode
df['Outcome'] = df['Outcome'].fillna('Unknown')
for col in ['Gender', 'Blood_Group', 'City', 'Department', 'Primary_Diagnosis', 'Admission_Type', 'Insurance_Provider']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Fill numerical missing values with median
num_cols = ['Age', 'Length_of_Stay_Days', 'Systolic_BP', 'Heart_Rate_BPM', 'Billing_Amount_INR']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Convert whole-number columns back to int64
int_cols = ['Age', 'Length_of_Stay_Days', 'Systolic_BP', 'Heart_Rate_BPM']
df[int_cols] = df[int_cols].astype('int64')

# Reset index and verify
df.reset_index(drop=True, inplace=True)
print("\n--- FINAL CLEAN DATA ---")
print(df.info())

# Export cleaned CSV to the same folder
df.to_csv(r"C:\Users\harsh\Desktop\devesh vs\indian_healthcare_patients_cleaned.csv", index=False)
print("\nSaved successfully!")

import pandas as pd
import numpy as np

# Load your newly cleaned dataset
df = pd.read_csv(r"C:\Users\harsh\Desktop\devesh vs\indian_healthcare_patients_cleaned.csv")
df['Admission_Date'] = pd.to_datetime(df['Admission_Date'])

# 1. Extract Date Features
df['Admission_Year'] = df['Admission_Date'].dt.year
df['Admission_Month'] = df['Admission_Date'].dt.month_name()
df['Day_of_Week'] = df['Admission_Date'].dt.day_name()

# 2. Create Age Groups using np.select()
age_conditions = [
    (df['Age'] < 18),
    (df['Age'] >= 18) & (df['Age'] < 45),
    (df['Age'] >= 45) & (df['Age'] < 65),
    (df['Age'] >= 65)
]
age_labels = ['Pediatric (<18)', 'Young Adult (18-44)', 'Middle Aged (45-64)', 'Senior (65+)']
df['Age_Group'] = np.select(age_conditions, age_labels, default='Unknown')

# 3. Categorize Blood Pressure Risk using np.select()
bp_conditions = [
    (df['Systolic_BP'] < 120),
    (df['Systolic_BP'] >= 120) & (df['Systolic_BP'] < 140),
    (df['Systolic_BP'] >= 140)
]
bp_labels = ['Normal', 'Elevated', 'High BP (Hypertension)']
df['BP_Category'] = np.select(bp_conditions, bp_labels, default='Normal')

# 4. Calculate Daily Cost per Stay (avoiding division by zero with np.where)
df['Cost_Per_Day_INR'] = np.where(
    df['Length_of_Stay_Days'] > 0,
    np.round(df['Billing_Amount_INR'] / df['Length_of_Stay_Days'], 2),
    df['Billing_Amount_INR']
)

print(df[['Patient_ID', 'Age_Group', 'BP_Category', 'Cost_Per_Day_INR']].head())

# 1. Department-wise Patient Count & Average Bill
dept_summary = df.groupby('Department').agg(
    Total_Patients=('Patient_ID', 'count'),
    Avg_Stay_Days=('Length_of_Stay_Days', 'mean'),
    Avg_Bill_INR=('Billing_Amount_INR', 'mean')
).round(2).sort_values(by='Total_Patients', ascending=False)

print("\n--- DEPARTMENT SUMMARY ---")
print(dept_summary)

# 2. Outcome Breakdown by Admission Type (%)
outcome_dist = pd.crosstab(df['Admission_Type'], df['Outcome'], normalize='index') * 100
print("\n--- OUTCOME % BY ADMISSION TYPE ---")
print(outcome_dist.round(2))

# 3. Top 5 Cities by Total Hospital Revenue
city_revenue = df.groupby('City')['Billing_Amount_INR'].sum().sort_values(ascending=False).head(5)
print("\n--- TOP 5 CITIES BY REVENUE (INR) ---")
print(city_revenue)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your cleaned dataset
df = pd.read_csv(r"C:\Users\harsh\Desktop\devesh vs\indian_healthcare_patients_cleaned.csv")

# Recreate Age_Group if running as a standalone script
age_conditions = [
    (df['Age'] < 18),
    (df['Age'] >= 18) & (df['Age'] < 45),
    (df['Age'] >= 45) & (df['Age'] < 65),
    (df['Age'] >= 65)
]
age_labels = ['Pediatric (<18)', 'Young Adult (18-44)', 'Middle Aged (45-64)', 'Senior (65+)']
df['Age_Group'] = np.select(age_conditions, age_labels, default='Unknown')

# Set up a 2x2 dashboard grid
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Indian Healthcare Patients (10K) - Analysis Dashboard', fontsize=16, fontweight='bold')

# Chart 1: Patients by Department
dept_counts = df['Department'].value_counts()
axes[0, 0].bar(dept_counts.index, dept_counts.values, color='#2b5c8f')
axes[0, 0].set_title('Patient Count by Department')
axes[0, 0].tick_params(axis='x', rotation=45)

# Chart 2: Average Billing (INR) by Admission Type
avg_bill = df.groupby('Admission_Type')['Billing_Amount_INR'].mean().sort_values()
axes[0, 1].barh(avg_bill.index, avg_bill.values, color='#2e8b57')
axes[0, 1].set_title('Average Billing Amount (INR) by Admission Type')
axes[0, 1].set_xlabel('INR (₹)')

# Chart 3: Patient Distribution by Age Group
age_counts = df['Age_Group'].value_counts()
axes[1, 0].pie(age_counts.values, labels=age_counts.index, autopct='%1.1f%%', startangle=140)
axes[1, 0].set_title('Age Group Distribution')

# Chart 4: Patient Outcomes Breakdown
outcome_counts = df['Outcome'].value_counts()
axes[1, 1].bar(outcome_counts.index, outcome_counts.values, color='#d9534f')
axes[1, 1].set_title('Patient Outcomes')
axes[1, 1].tick_params(axis='x', rotation=30)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the dashboard as an image and display it
plt.savefig(r"C:\Users\harsh\Desktop\devesh vs\healthcare_dashboard.png", dpi=300)
print("Dashboard saved as healthcare_dashboard.png!")
plt.show()