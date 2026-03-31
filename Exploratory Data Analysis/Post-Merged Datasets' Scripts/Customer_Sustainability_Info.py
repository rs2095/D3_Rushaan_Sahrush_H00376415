import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

plt.style.use('seaborn-v0_8-whitegrid')
PRIMARY_COLOR = "teal"
sns.set_palette([PRIMARY_COLOR])

df = pd.read_csv("../../Data/Processed/Datasets/sustainable_customer_final.csv")
target = "Green_Purchase_Made"

print("Shape:", df.shape)
print("\nColumn overview:\n")
print(df.info())

#target distribution counts and percentages
print("\nGreen Purchase Distribution:")
counts = df[target].value_counts()
pcts = (df[target].value_counts(normalize=True) * 100).round(2)
print(pd.DataFrame({"Count": counts, "Percentage (%)": pcts}))

#target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=target, data=df, color=PRIMARY_COLOR)
plt.title("Distribution of Green Purchase Decisions")
plt.xlabel("Green Purchase Made (0 = No, 1 = Yes)")
plt.tight_layout()
plt.show()

#checks categorical columns distribution
categorical_cols = [
    "Referral_Source",
    "Product_Category"
]

#performs chi-square test and plots purchase rate by category
print("\nCategorical Analysis vs Purchase Decision")
for col in categorical_cols:
    if col in df.columns:
        contingency = pd.crosstab(df[col], df[target])
        _, p, _, _ = chi2_contingency(contingency)
        print(f"{col:20} | Chi-Square p-value: {p:.4f}")

        #calculates purchase rate by category
        plt.figure(figsize=(8, 4))
        purchase_rate = df.groupby(col)[target].mean().sort_values(ascending=False).reset_index()
        sns.barplot(x=col, y=target, data=purchase_rate, color=PRIMARY_COLOR)
        plt.title(f"Green Purchase Rate by {col}")
        plt.ylabel("Average Purchase Rate")
        plt.xticks(rotation=30)
        plt.tight_layout()
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

#distribution of green consumption score
plt.figure(figsize=(8, 4))
sns.histplot(df["Green_Consumption_Score"], bins=15, color=PRIMARY_COLOR, kde=True)
plt.title("Distribution of Green Consumption Score", fontweight='bold')
plt.tight_layout()
plt.show()

#distribution for numeric columns
df[numeric_cols].hist(bins=15, figsize=(14, 10), color=PRIMARY_COLOR, edgecolor="black")
plt.suptitle("Sustainability and Behavioral Feature Distributions", fontsize=15)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#compares behavioral engagement between buyers and non-buyers
action_cols = [
    "Time_Spent_on_Green_Products",
    "Clicks_on_Green_Products",
    "Previous_Green_Purchases"
]
mean_actions = df.groupby(target)[action_cols].mean().T

#average behavioral engagement by outcome
mean_actions.plot(kind="bar", figsize=(10, 5), color=[PRIMARY_COLOR, "#458B74"], edgecolor="black")
plt.title("Average Behavioral Engagement by Purchase Outcome")
plt.ylabel("Mean Value")
plt.xticks(rotation=25)
plt.legend(title="Green Purchase Made", labels=["No (0)", "Yes (1)"])
plt.tight_layout()
plt.show()

#correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = df[numeric_cols + [target]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Sustainability and Behavioral Correlation Matrix", fontsize=14, fontweight='bold')
plt.show()