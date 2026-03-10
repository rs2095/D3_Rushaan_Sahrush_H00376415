import pandas as pd

#for dataset 1
orders = pd.read_csv("../../Preprocessed/Datasets/5_customer_orders_frequency_cleaned.csv")

print("\nDATASET 1 - Customer Orders")

print("\nDataset info:\n")
print(orders.info())

orders.to_csv(f"../Datasets/mb_customer_orders_final.csv",index=False)

#for dataset 2
department = pd.read_csv("../../Preprocessed/Datasets/5_departments_cleaned.csv")

dup_departments = department["Department_ID"].duplicated().sum()

print("\nDATASET 2 - Departments")
print("Duplicate Department_IDs:", dup_departments)

department.to_csv(f"../Datasets/mb_departments_final.csv",index=False)

#for dataset 3
order_products = pd.read_csv("../../Preprocessed/Datasets/5_order_product_reorders_cleaned.csv")

duplicate_pairs = (
    order_products
    .duplicated(subset=["Order_ID", "Product_ID"])
    .sum()
)

print("\nDATASET 3 - Order Products")
print("(Order_ID, Product_ID) duplicate pairs:", duplicate_pairs)

if duplicate_pairs > 0:
    print("Duplicate product entries within same order")
else:
    print("(Order_ID, Product_ID) pairs are unique")

order_products.to_csv(f"../Datasets/mb_order_product_map_final.csv",index=False)

#for dataset 4
products = pd.read_csv(
    "../../Preprocessed/Datasets/5_products_available_cleaned.csv"
)

dup_products = products["Product_ID"].duplicated().sum()

print("\nDATASET 4 - Products")
print("Duplicate Product_IDs:", dup_products)

products.to_csv(f"../Datasets/mb_products_final.csv",index=False)
