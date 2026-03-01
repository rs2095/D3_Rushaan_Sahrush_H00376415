import pandas as pd

df = pd.read_csv("../../Data/Raw/3_Ecommerce Customer Churn Analysis and Prediction/E Commerce Dataset.csv")

print("Initial shape:", df.shape)
print("\nDataset info:\n")
print(df.info())

#missing values analysis
print("\nMissing values per column:\n")
missing_counts = df.isnull().sum().sort_values(ascending=False)
print(missing_counts)

row_null_counts = df.isnull().sum(axis=1)

print("\nRow-wise null count distribution:\n")
print(row_null_counts.value_counts().sort_index())

#checks top 10 hoursspendonapp
print("\nHourSpendOnApp value counts (top 10):\n")
print(df["HourSpendOnApp"].value_counts().head(10))

#calculates and prints percentage of rows where HourSpendOnApp is 0
zero_hours_ratio = (df["HourSpendOnApp"] == 0).mean() * 100
print(f"\nPercentage of rows with 0 hours spent: {zero_hours_ratio:.2f}%")

#checks churn distribution
print("\nRaw churn distribution:\n")
print(df["Churn"].value_counts())

#checks categorical columns with many distinct values
categorical_cols = df.select_dtypes(include=["object"]).columns

print("\nCategorical column cardinality:")
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique categories")