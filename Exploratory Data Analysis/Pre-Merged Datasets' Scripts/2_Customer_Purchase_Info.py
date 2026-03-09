import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../../Data/Preprocessed/Datasets/2_customer_purchase_info_cleaned.csv")

print("Initial shape:", df.shape)
print("\nColumn overview:\n")
print(df.info())

#order return analysis with plot distribution
print("\nOrder_Returns value counts:")
print(df["Order_Returns"].value_counts())

plt.figure(figsize=(4,3))
sns.countplot(x="Order_Returns", data=df)
plt.title("Order Returns Distribution")
plt.xlabel("Order Returned (1 = Yes, 0 = No)")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()

#customer age analysis with plot distribution
plt.figure(figsize=(6,4))
plt.hist(df["Customer_Age"], bins=10, edgecolor="black")
plt.title("Customer Age Distribution")
plt.xlabel("Customer Age")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()
