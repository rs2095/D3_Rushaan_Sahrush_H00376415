import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

orders = pd.read_csv("../../Data/Preprocessed/Datasets/5_customer_orders_frequency_cleaned.csv")
order_products = pd.read_csv("../../Data/Preprocessed/Datasets/5_order_product_reorders_cleaned.csv")
products = pd.read_csv("../../Data/Preprocessed/Datasets/5_products_available_cleaned.csv")
department = pd.read_csv("../../Data/Preprocessed/Datasets/5_departments_cleaned.csv")  # department info

print("Orders:", orders.shape)
print("Order_Products:", order_products.shape)
print("Products:", products.shape)
print("Departments:", department.shape)

print("\nDataset info:\n")
print(orders.info())

print("\nDataset info:\n")
print(order_products.info())

print("\nDataset info:\n")
print(products.info())

print("\nDataset info:\n")
print(department.info())

#merging products to department info
products = products.merge(department, on="Department_ID", how="left")

#departments analysis
print("Departments Overview")
print(department.info())
print("\nDepartment Name Counts:")
print(department['Department_Name'].value_counts())

#products analysis
plt.figure(figsize=(8,4))
sns.countplot(x="Department_ID", data=products, palette="Set2")
plt.title("Number of Products per Department (by ID)")
plt.xlabel("Department ID")
plt.ylabel("Number of Products")
plt.tight_layout()
plt.show()

#products per department name analysis
plt.figure(figsize=(10,4))
sns.countplot(y="Department_Name", data=products, palette="Set3", order=products['Department_Name'].value_counts().index)
plt.title("Number of Products per Department Name")
plt.xlabel("Number of Products")
plt.ylabel("Department Name")
plt.tight_layout()
plt.show()

#top 20 customers by number of orders analysis
top_customers = orders['Customer_ID'].value_counts().head(20)
plt.figure(figsize=(10,4))
sns.barplot(x=top_customers.index.astype(str), y=top_customers.values, palette="Set2")
plt.title("Top 20 Customers by Number of Orders")
plt.xlabel("Customer ID")
plt.ylabel("Number of Orders")
plt.tight_layout()
plt.show()

#merging order-product with department info
order_prod_merged = order_products.merge(products[['Product_ID', 'Department_ID', 'Department_Name']], on='Product_ID', how='left')

#order-product reorder counts
plt.figure(figsize=(6,4))
sns.countplot(x="Was_Product_Reordered", data=order_prod_merged, palette="Set1")
plt.title("Product Reorder Counts")
plt.xlabel("Was Product Reordered (0=No, 1=Yes)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

#top 20 products by number of orders analysis
top_products = order_prod_merged['Product_ID'].value_counts().head(20)
plt.figure(figsize=(10,4))
sns.barplot(x=top_products.index.astype(str), y=top_products.values, palette="Set2")
plt.title("Top 20 Products by Number of Orders")
plt.xlabel("Product ID")
plt.ylabel("Times Ordered")
plt.tight_layout()
plt.show()

#top departments by reorder count analysis
top_departments = order_prod_merged.groupby('Department_Name')['Was_Product_Reordered'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,4))
sns.barplot(x=top_departments.index, y=top_departments.values, palette="Set3")
plt.title("Top 10 Departments by Number of Reorders")
plt.xlabel("Department Name")
plt.ylabel("Reorders Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
