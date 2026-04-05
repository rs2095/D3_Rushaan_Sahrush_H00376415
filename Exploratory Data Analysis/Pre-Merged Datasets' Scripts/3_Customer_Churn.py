import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("../../Data/Preprocessed/Datasets/3_customer_churn_cleaned.csv")
print("Initial shape:", df.shape)
print("\nDataset info:\n")
print(df.info())

#checks categorical columns distribution
categorical_cols = [
    "Preferred_Payment_Mode",
    "Marital_Status",
    "Preferred_Login_Device",
    "Churn"
]

plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, len(categorical_cols), i)
    sns.countplot(x=col, data=df, palette="viridis")
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45, fontsize=8)
plt.tight_layout()
plt.show()

#checks numerical columns distribution
numeric_cols = [
    "Hours_Spend_On_Website",
    "Order_Amount_Hike_From_Last_Year",
    "Days_Since_Last_Order",
    "Average_Cash_Back"
]

plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True, color="teal")
    plt.title(f"{col} Distribution")
plt.tight_layout()
plt.show()

#churn correlation heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix[['Churn']].sort_values(by='Churn', ascending=False),
            annot=True, cmap='RdYlGn', center=0)
plt.title("Correlation of Features with Churn")
plt.show()
