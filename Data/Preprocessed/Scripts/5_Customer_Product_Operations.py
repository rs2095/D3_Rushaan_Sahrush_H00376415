import pandas as pd

products = pd.read_csv("../../Raw/5_Instacart Market Basket Analysis/products.csv")
order_products = pd.read_csv("../Datasets/5_order_product_reorders_cleaned.csv")
departments = pd.read_csv("../../Raw/5_Instacart Market Basket Analysis/departments.csv")

#prints original products count and order product rows
print(f"Original products count: {products['product_id'].nunique()}")
print(f"Order-product rows: {len(order_products)}")

#keeps only products that appear at least once in orders
used_product_ids = set(order_products["Product_ID"].unique())

products = products[products["product_id"].isin(used_product_ids)].copy()

print(f"Products used at least once: {products['product_id'].nunique()}")
print(f"Products dropped (never used): {len(used_product_ids) - products['product_id'].nunique()}")

#drops aisle_id and rows with nulls
if "aisle_id" in products.columns:
    products = products.drop(columns=["aisle_id"])

products = products.dropna()

print(f"Products after dropping nulls: {len(products)}")

#creates sequential product_id new
products = products.sort_values(by="product_id").reset_index(drop=True)
products["Product_ID_New"] = range(1, len(products) + 1)

product_id_mapping = dict(
    zip(products["product_id"], products["Product_ID_New"])
)

#maps to order_products
order_products["Product_ID_New"] = order_products["Product_ID"].map(product_id_mapping)

missing = order_products["Product_ID_New"].isna().sum()
if missing > 0:
    raise ValueError(f" {missing} order-product rows could not be mapped to products")

print("All product IDs mapped correctly")

#drops and renames columns for products dataset
final_products = products.drop(columns=["product_id"]).rename(columns={
    "product_name": "Product_Name",
    "department_id": "Department_ID",
    "Product_ID_New": "Product_ID"
})

#drops and renames columns for order_products dataset
final_order_products = order_products.drop(columns=["Product_ID"]).rename(columns={
    "Product_ID_New": "Product_ID",
    "reordered": "Was_Product_Reordered"
})

#rename columns for department dataset
departments = departments.rename(columns={
    "department_id": "Department_ID",
    "department": "Department_Name"
})

#drops duplicates
departments = departments.drop_duplicates().reset_index(drop=True)

print(f"\nTotal final products: {final_products['Product_ID'].nunique()}")
print(f"Product ID range: {final_products['Product_ID'].min()} → {final_products['Product_ID'].max()}")
print(f"Total order-product rows: {len(final_order_products)}")
print(f"Total departments: {departments.shape[0]}")

final_products.to_csv("../Datasets/5_products_available_cleaned.csv", index=False)
final_order_products.to_csv("../Datasets/5_order_product_reorders_cleaned.csv", index=False)
departments.to_csv("../Datasets/5_departments_cleaned.csv", index=False)
