import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv( "../../Data/Raw/2_E-commerce Customer Data For Behavior Analysis/ecommerce_customer_data_custom_ratios.csv")

print("Initial shape:", df.shape)
print("\nColumn overview:\n")
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

#checks customer frequency distribution
customer_counts = df["Customer ID"].value_counts()

plt.figure(figsize=(8, 4))
customer_counts.hist(bins=50)
plt.title("Customer Transaction Frequency Distribution")
plt.xlabel("Number of Transactions")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

#checks categorical columns with many distinct values
categorical_cols = df.select_dtypes(include=["object"]).columns

print("\nCategorical column cardinality:")
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique categories")

#checks returns column
print("\nReturns value counts:\n")
print(df["Returns"].value_counts())

#checks total unique customers
print("\nNumber of unique customers:", df["Customer ID"].nunique())

