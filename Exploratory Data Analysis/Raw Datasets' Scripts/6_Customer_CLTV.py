import pandas as pd

df = pd.read_csv("../../Data/Raw/6_Predict CLTV of a Customer/train_BRCpofr.csv")

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

#checks cltv distribution
print("\n CLTV Summary Statistics")
print(df["cltv"].describe())

print("\nCLTV value counts (binned):")
print(
    pd.cut(df["cltv"], bins=10)
    .value_counts()
    .sort_index()
)

#checks income column
print("\nRaw income values (sample):")
print(df["income"].value_counts().head(10))

#checks for inconsistent suffixes
income_suffix_counts = df["income"].astype(str).str.extract(r'([A-Za-z]+)$')[0].value_counts()

#checks categorical demographic / policy features
categorical_cols = [
    "gender",
    "qualification",
    "marital_status",
    "policy",
    "type_of_policy",
    "area"
]

for col in categorical_cols:
    if col in df.columns:
        print(f"\n{col} value counts:")
        print(df[col].value_counts(dropna=False))
