import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../../Data/Preprocessed/Datasets/4_customer_green_purchases_cleaned.csv")  # update path
print("Initial shape:", df.shape)
print("\nDataset info:\n")
print(df.info())

#checks categorical columns distribution
categorical_cols = [
    "Referral_Source",
    "Product_Category"
]

plt.figure(figsize=(10, 4))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, len(categorical_cols), i)
    sns.countplot(x=col, data=df, palette="Set2", hue=col, legend=False)
    plt.title(f"{col} Count", fontsize=10)
    plt.xticks(rotation=30, fontsize=8)
plt.tight_layout(pad=2.0)
plt.show()

#checks numerical columns distribution
numeric_cols = [
    "Cultural_Eco_Alignment",
    "Emotional_Guilt_Score",
    "Emotional_Pride_Score",
    "Social_Influence_Score",
    "Green_Knowledge_Level",
    "Time_Spent_on_Green_Products",
    "Clicks_on_Green_Products",
    "Previous_Green_Purchases",
    "Green_Product_Rating",
    "Green_Consumption_Score",
    "Green_Purchase_Made"
]

#plot distribution for numeric columns
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)
    plt.hist(df[col], color="teal", edgecolor="black", alpha=0.7)
    plt.title(f"{col}", fontsize=9, fontweight='bold')
    plt.xlabel("Value", fontsize=8)
    plt.ylabel("Frequency", fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)

plt.tight_layout(pad=3.0)
plt.show()

#target variable distributions
plt.figure(figsize=(6, 4))
sns.histplot(df["Green_Consumption_Score"], bins=15, color="teal", kde=True)
plt.title("Green Consumption Score Distribution", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x="Green_Purchase_Made", data=df, color="teal")
plt.title("Green Purchase Made Distribution (0 = No, 1 = Yes)", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()