import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../../Data/Preprocessed/Datasets/6_customer_cltv_cleaned.csv")
print("Initial shape:", df.shape)
print("\nDataset info:\n")
print(df.info())

#checks categorical columns distribution
categorical_cols = [
    "Customer_Gender",
    "Customer_Residence_Type",
    "Customer_Qualification",
    "Customer_Income",
    "Membership_Type",
    "Total_Subscriptions"
]

#value counts for categorical values and plot distribution
for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())

plt.figure(figsize=(18, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, len(categorical_cols), i)
    sns.countplot(x=col, data=df, palette="viridis")
    plt.title(f"{col} Frequency", fontsize=10)
    plt.xticks(rotation=45, fontsize=8)

plt.tight_layout(pad=2.0)
plt.show()

#checks numerical columns and plot distribution
numeric_cols = [
    "Customer_Live_Time_Value",
    "Loyalty_Engagement_Score"
]

for col in numeric_cols:
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], bins=20, kde=True, color="skyblue")
    plt.title(f"Distribution of {col}", fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

#correlation heatmap
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Scores vs. CLTV", fontsize=14)
plt.show()