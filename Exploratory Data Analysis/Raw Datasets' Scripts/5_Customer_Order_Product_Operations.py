import pandas as pd

orders = pd.read_csv("../../Data/Raw/5_Instacart Market Basket Analysis/orders.csv")
order_products = pd.read_csv("../../Data/Raw/5_Instacart Market Basket Analysis/order_products__prior.csv")
products = pd.read_csv("../../Data/Raw/5_Instacart Market Basket Analysis/products.csv")
department = pd.read_csv("../../Data/Raw/5_Instacart Market Basket Analysis/departments.csv")

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

#missing values analysis
print("\nMissing values in Orders:\n")
missing_counts = orders.isnull().sum().sort_values(ascending=False)
print(missing_counts)

print("\nMissing values in Order_Products:\n")
missing_counts = order_products.isnull().sum().sort_values(ascending=False)
print(missing_counts)

print("\nMissing values in Products:\n")
missing_counts = products.isnull().sum().sort_values(ascending=False)
print(missing_counts)

print("\nMissing values in Department:\n")
missing_counts = department.isnull().sum().sort_values(ascending=False)
print(missing_counts)

#checks order dataset and product dataset link
orders_without_products = set(orders["order_id"]) - set(order_products["order_id"])
print(f"\nOrders without any products: {len(orders_without_products)}")

#checks product usage frequency
unused_products = set(products["product_id"]) - set(order_products["product_id"])
print(f"\nProducts never ordered: {len(unused_products)}")

#checks reorder behaviour patterns
print("\nReorder flag distribution:\n")
print(order_products["reordered"].value_counts())

#counts products per department
products_per_department = (
    products.groupby("department_id")["product_id"]
    .nunique()
    .sort_index()
)

#checks products per department
print("\nProducts per department:\n")
for dept_id, count in products_per_department.items():
    print(f"Department {dept_id}: {count} products")

