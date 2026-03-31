import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

plt.style.use('seaborn-v0_8-whitegrid')
PRIMARY_COLOR = "teal"
sns.set_palette([PRIMARY_COLOR])

products_df = pd.read_csv("../../Data/Preprocessed/Datasets/5_products_available_cleaned.csv")
orders_df = pd.read_csv("../../Data/Preprocessed/Datasets/5_customer_orders_frequency_cleaned.csv")
order_prod_df = pd.read_csv("../../Data/Preprocessed/Datasets/5_order_product_reorders_cleaned.csv")
departments_df = pd.read_csv("../../Data/Preprocessed/Datasets/5_departments_cleaned.csv")

#merges department names into products dataset
products_df = products_df.merge(departments_df, on="Department_ID", how="left")

#creates full merged dataset for analysis
order_prod_merged = order_prod_df.merge(
    products_df[['Product_ID', 'Department_Name']],
    on='Product_ID',
    how='left'
).merge(
    orders_df[['Order_ID', 'Customer_ID']],
    on='Order_ID',
    how='left'
)

#prints dataset overview statistics
print("\nDataset Overview:")
print(f"Total Orders: {orders_df['Order_ID'].nunique()}")
print(f"Total Customers: {orders_df['Customer_ID'].nunique()}")
print(f"Total Products: {products_df['Product_ID'].nunique()}")
print(f"Total Departments: {products_df['Department_ID'].nunique()}")

#gets top departments by purchase frequency
dept_counts = order_prod_merged['Department_Name'].value_counts().head(10)

#plots top departments
plt.figure(figsize=(10, 5))
sns.barplot(x=dept_counts.index, y=dept_counts.values, color=PRIMARY_COLOR)
plt.title("Top Departments by Purchase Frequency", fontweight='bold')
plt.xlabel("Department")
plt.ylabel("Number of Purchases")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#calculates reorder rate by department
dept_reorder = order_prod_merged.groupby('Department_Name')['Was_Product_Reordered'].mean().sort_values(ascending=False).head(10)

#plots reorder rate by department
plt.figure(figsize=(10, 5))
sns.barplot(x=dept_reorder.index, y=dept_reorder.values, color=PRIMARY_COLOR)
plt.title("Average Reorder Rate by Top Departments", fontweight='bold')
plt.ylabel("Reorder Rate")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

#calculates number of orders per customer
orders_per_customer = orders_df.groupby('Customer_ID')['Order_ID'].count()

#distribution of orders per customer
plt.figure(figsize=(8, 4))
sns.histplot(orders_per_customer, bins=30, color=PRIMARY_COLOR, kde=True)
plt.title("Distribution of Orders per Customer", fontweight='bold')
plt.xlabel("Number of Orders")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

#calculates customer reorder rate
cust_reorder = order_prod_merged.groupby('Customer_ID')['Was_Product_Reordered'].mean()

#customer reorder rate distribution
plt.figure(figsize=(8, 4))
sns.histplot(cust_reorder, bins=20, color=PRIMARY_COLOR, kde=True)
plt.title("Customer Reorder Rate Distribution", fontweight='bold')
plt.xlabel("Reorder Rate")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()

#creates basket matrix for association rules
basket = order_prod_df.groupby(['Order_ID', 'Product_ID'])['Was_Product_Reordered'] \
    .max().unstack().fillna(0)
basket[basket > 0] = 1

#generates frequent itemsets and association rules
frequent_itemsets = apriori(basket, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(by='lift', ascending=False).head(10)

#formats rules for visualization
top_rules = rules.copy()
top_rules['rule'] = top_rules['antecedents'].apply(lambda x: ','.join(list(map(str, x)))) + \
                    " ->" + \
                    top_rules['consequents'].apply(lambda x: ','.join(list(map(str, x))))

#plots top association rules by lift
plt.figure(figsize=(10, 6))
sns.barplot(x='lift', y='rule', data=top_rules, color=PRIMARY_COLOR)
plt.title("Top 10 Product Association Rules by Lift", fontweight='bold')
plt.xlabel("Lift Score")
plt.ylabel("Association Rule")
plt.tight_layout()
plt.show()

#prints top association rules
print("\nTop Association rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])