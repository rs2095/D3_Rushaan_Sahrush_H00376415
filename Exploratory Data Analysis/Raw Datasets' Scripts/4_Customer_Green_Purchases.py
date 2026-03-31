import pandas as pd

#load raw dataset
df = pd.read_csv("../../Data/Raw/4_Green Consumption Behavior Dataset/Green_Consumption_Behavior_2024.csv")

print("Initial shape:", df.shape)
print("\nDataset info:\n")
print(df.info())

#missing values and duplicate analysis
print("\nMissing values per column:\n")
missing_counts = df.isnull().sum().sort_values(ascending=False)
print(missing_counts)

duplicate_count = df.duplicated().sum()
print("\nNumber of duplicate rows:")
print(duplicate_count)
row_null_counts = df.isnull().sum(axis=1)

print("\nRow-wise null count distribution:\n")
print(row_null_counts.value_counts().sort_index())

#checks demographic features
demographic_cols = ["Age", "Gender", "Device_Type"]
for col in demographic_cols:
    if col in df.columns:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False).head())

#checks numerical features
numerical_cols = [
    "Time_Spent_on_Green_Products",
    "Green_Consumption_Score"
]

print("\nNumerical feature summary:\n")
print(df[numerical_cols].describe())

