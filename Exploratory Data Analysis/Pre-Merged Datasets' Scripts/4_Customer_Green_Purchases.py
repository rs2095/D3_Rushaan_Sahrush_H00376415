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
    "Product_Category",
    "Green_Purchase_Made"
]

plt.figure(figsize=(12, 4))
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(1, len(categorical_cols), i)
    sns.countplot(x=col, data=df, palette="Set2")
    plt.title(f"{col} Count", fontsize=8)
    plt.xticks(rotation=30, fontsize=7)
    plt.yticks(fontsize=7)
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
    "Green_Consumption_Score"
]

#plot distribution for numeric columns
plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 5, i)
    plt.hist(df[col], bins=6, color="lightgreen", edgecolor="black")
    plt.title(f"{col}", fontsize=8)
    plt.xlabel(col, fontsize=7)
    plt.ylabel("Count", fontsize=7)
    plt.xticks(fontsize=6, rotation=20)
    plt.yticks(fontsize=6)
plt.tight_layout(pad=2.0)
plt.show()

plt.figure(figsize=(5,3))
sns.histplot(df["Green_Consumption_Score"], bins=6, color="lightblue", kde=True)
plt.title("Green Consumption Score Distribution", fontsize=10)
plt.xlabel("Green Consumption Score")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

