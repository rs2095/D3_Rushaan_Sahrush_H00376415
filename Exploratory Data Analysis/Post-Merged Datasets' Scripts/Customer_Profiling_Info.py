import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
PRIMARY_COLOR = "teal"

df = pd.read_csv("../../Data/Processed/Datasets/customer_info_final.csv")
target = "Churn"

print("Shape:", df.shape)
print("\nColumn overview:\n")
print(df.info())

#basic demographic info
print("\nCustomer Age Analysis:")
print(f"Minimum Age: {df['Customer_Age'].min()}")
print(f"Maximum Age: {df['Customer_Age'].max()}")
print(f"Mean Age: {df['Customer_Age'].mean():.2f}")
print(f"Median Age: {df['Customer_Age'].median()}")

#target variable distributions
plt.figure(figsize=(6, 4))
sns.countplot(x=target, data=df, color=PRIMARY_COLOR)
plt.title("Churn Distribution")
plt.tight_layout()
plt.show()

#checks categorical columns distribution
categorical_cols = [
    "Customer_Gender",
    "Customer_Income",
    "Customer_Residence_Type",
    "Preferred_Login_Device",
    "Preferred_Payment_Mode",
    "Marital_Status",
    "Preferred_Order_Category_Name"
]

print("\nCategorical Distributions and Chi-Square Tests:")
for col in categorical_cols:
    if col in df.columns:
        counts = df[col].value_counts()
        pcts = df[col].value_counts(normalize=True) * 100
        summary = pd.DataFrame({"Count": counts, "Percentage (%)": pcts.round(2)})
        print(f"\n--- {col} ---")
        print(summary)

        contingency = pd.crosstab(df[col], df[target])
        chi2, p, _, _ = chi2_contingency(contingency)
        print(f"Chi-Square p-value: {p:.4f}")

        #churn rate by category
        plt.figure(figsize=(8, 4))
        churn_rate = df.groupby(col)[target].mean().sort_values(ascending=False).reset_index()
        sns.barplot(x=col, y=target, data=churn_rate, color=PRIMARY_COLOR)
        plt.title(f"Churn Rate by {col}")
        plt.ylabel("Churn Rate (Mean)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

#checks numerical columns distribution
numeric_cols = [
    "Hours_Spend_On_Website",
    "Total_Devices_Registered",
    "Satisfaction_Score",
    "Complain_Count_In_Last_Month",
    "Order_Amount_Hike_From_Last_Year",
    "Discount_Coupon_Used_Count",
    "Days_Since_Last_Order",
    "Average_Cash_Back",
    "Customer_Live_Time_Value"
]

#distribution for CLV
for col in ["Customer_Live_Time_Value"]:
    if col in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], bins=20, kde=True, color=PRIMARY_COLOR)
        plt.title(f"Distribution of {col}", fontweight='bold')
        plt.tight_layout()
        plt.show()

#distribution for numeric columns
df[numeric_cols ].hist(bins=20, figsize=(15, 10), color=PRIMARY_COLOR, edgecolor="black")
plt.suptitle("Numeric Feature Distributions Overview", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#skewness summary
skew_summary = pd.DataFrame({
    "Mean": df[numeric_cols ].mean().round(2),
    "Median": df[numeric_cols ].median().round(2),
    "Skew": df[numeric_cols ].skew().round(3)
})
print("\nNumeric Feature Summary:\n", skew_summary)

#correlation matrix
plt.figure(figsize=(12, 10))
corr_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix of Numeric Features")
plt.show()

#feature engineering and PCA
df['Purchase_Intensity'] = (df['Total_Orders_This_Month'] * df['Total_Amount_Spent']) / 1000
df['Recency'] = df['Days_Since_Last_Order']
df['Avg_Spend_Per_Order'] = df['Total_Amount_Spent'] / (df['Total_Orders_This_Month'] + 1)

eng_features = ['Purchase_Intensity', 'Recency', 'Avg_Spend_Per_Order']

#handles missing values
X_eng = df[eng_features].fillna(df[eng_features].median())

#distribution of engineered features
print("\nEngineered Features Distribution Summary:")
print(df[eng_features].describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(eng_features):
    sns.histplot(df[col], kde=True, color=PRIMARY_COLOR, ax=axes[i], bins=30)
    axes[i].set_title(f"Distribution of {col.replace('_', ' ')}", fontweight='bold')
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


#uses Yeo-Johnson to minimize skewness
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X_eng)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_transformed)
df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]

#PCA scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x='PCA1',
    y='PCA2',
    alpha=0.5,
    s=50,
    color=PRIMARY_COLOR
)
plt.title("PCA 2D Projection of Customer Behavior", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#analyzes and visualizes customer segments with feature averages and churn rates
if 'Segment_Label' in df.columns:
    plt.figure(figsize=(7, 4))
    sns.countplot(x='Segment_Label', data=df, color=PRIMARY_COLOR)
    plt.title("Customer Count per Segment", fontweight='bold')
    plt.show()

    seg_summary = df.groupby('Segment_Label')[eng_features].mean()

    ax = seg_summary.plot(
        kind='bar',
        figsize=(12, 6),
        color=[PRIMARY_COLOR, "#45ada8", "#9de0ad"],
        edgecolor='black',
        width=0.8
    )
    plt.title("Average Engineered Feature Values per Segment", fontweight='bold', fontsize=14)
    plt.ylabel("Mean Value")
    plt.xlabel("Customer Segment")
    plt.xticks(rotation=0)
    plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    if 'Churn' in df.columns:
        plt.figure(figsize=(7, 4))
        sns.barplot(x='Segment_Label', y='Churn', data=df, color=PRIMARY_COLOR, errorbar=None)
        plt.title("Churn Rate by Segment", fontweight='bold')
        plt.ylabel("Churn Probability")
        plt.show()