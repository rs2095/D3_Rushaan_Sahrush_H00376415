import pandas as pd
from scipy.stats import mode

churn_df = pd.read_csv("../../Preprocessed/Datasets/3_customer_churn_cleaned.csv")
cltv_df = pd.read_csv("../../Preprocessed/Datasets/6_customer_cltv_cleaned.csv")
purchase_df = pd.read_csv("../../Preprocessed/Datasets/2_customer_purchase_info_cleaned.csv")

print("Initial dataset sizes:")
print("Churn:", churn_df.shape)
print("CLV:", cltv_df.shape)
print("Purchases:", purchase_df.shape)

#checks if customers have multiple age values across transactions
age_variation = (
    purchase_df
    .groupby("Customer_ID")["Customer_Age"]
    .nunique()
)

inconsistent_age_customers = age_variation[age_variation > 1]

print("\nCustomers with inconsistent ages across orders:")
print(f"Count: {len(inconsistent_age_customers)}")

#aggregates purchase transactions to customer level with total spending, returns, and most frequent age
purchase_agg = (
    purchase_df
    .groupby("Customer_ID")
    .agg(
        Customer_Age=("Customer_Age", lambda x: mode(x, keepdims=True)[0][0]),
        Total_Amount_Spent=("Amount_Spent", "sum"),
        Total_Returns=("Order_Returns", "sum")
    )
    .reset_index()
)

print("\nAggregated purchase dataset shape:", purchase_agg.shape)

#merges churn and clv datasets on customer_id and checks row loss after merge
pre_merge_rows = churn_df.shape[0]

merged_df = churn_df.merge(
    cltv_df,
    on="Customer_ID",
    how="inner"
)

post_merge_rows = merged_df.shape[0]

print("\nChurn and CLV merge:")
print("Rows before:", pre_merge_rows)
print("Rows after:", post_merge_rows)
print("Dropped rows:", pre_merge_rows - post_merge_rows)

#merges aggregated purchase data and checks row loss after merge
pre_purchase_merge = merged_df.shape[0]

merged_df = merged_df.merge(
    purchase_agg,
    on="Customer_ID",
    how="inner"
)

post_purchase_merge = merged_df.shape[0]

print("\nAfter merging purchase data:")
print("Rows before:", pre_purchase_merge)
print("Rows after:", post_purchase_merge)
print("Dropped rows:", pre_purchase_merge - post_purchase_merge)

###feature engineering-preferred order category###
orders_df = pd.read_csv("../../Preprocessed/Datasets/5_customer_orders_frequency_cleaned.csv")
order_products_df = pd.read_csv("../../Preprocessed/Datasets/5_order_product_reorders_cleaned.csv")
products_df = pd.read_csv("../../Preprocessed/Datasets/5_products_available_cleaned.csv")
departments_df = pd.read_csv("../../Preprocessed/Datasets/5_departments_cleaned.csv")

#links customers to purchased products and their departments through orders
cust_order_products = orders_df[["Customer_ID", "Order_ID"]].merge(
    order_products_df[["Order_ID", "Product_ID"]],
    on="Order_ID", how="inner"
)
cust_product_dept = cust_order_products.merge(
    products_df[["Product_ID", "Department_ID"]],
    on="Product_ID", how="inner"
)

#counts department frequency per customer
dept_counts = cust_product_dept.groupby(["Customer_ID", "Department_ID"]).size().reset_index(name="Dept_Count")

#picks most frequent department per customer
preferred_dept = dept_counts.sort_values(["Customer_ID", "Dept_Count"], ascending=[True, False]) \
    .drop_duplicates(subset=["Customer_ID"]) \
    .rename(columns={"Department_ID": "Preferred_Order_Category"})[["Customer_ID", "Preferred_Order_Category"]]

#maps department names
preferred_dept = preferred_dept.merge(
    departments_df.rename(columns={"department_id": "Department_ID", "department": "Department_Name"}),
    left_on="Preferred_Order_Category", right_on="Department_ID",
    how="left"
)
preferred_dept = preferred_dept.rename(columns={"Department_Name": "Preferred_Order_Category_Name"})
preferred_dept = preferred_dept[["Customer_ID", "Preferred_Order_Category", "Preferred_Order_Category_Name"]]

#merges into main dataset
merged_df = merged_df.merge(preferred_dept, on="Customer_ID", how="left")

#fills missing values
merged_df["Preferred_Order_Category"] = merged_df["Preferred_Order_Category"].fillna("Unknown")
merged_df["Preferred_Order_Category_Name"] = merged_df["Preferred_Order_Category_Name"].fillna("Unknown")

###feature engineering-total orders this month###
orders_per_customer = (
    orders_df
    .groupby("Customer_ID")
    .size()
    .reset_index(name="Total_Orders_This_Month")
)

merged_df = merged_df.merge(
    orders_per_customer,
    on="Customer_ID",
    how="left"
)

merged_df["Total_Orders_This_Month"] = merged_df["Total_Orders_This_Month"].fillna(0).astype(int)

print("\nFinal merged dataset shape:", merged_df.shape)
print("\nFinal columns:")
print(list(merged_df.columns))

#formats customer_id
merged_df["Customer_ID"] = (
    merged_df["Customer_ID"]
    .astype(int)
    .astype(str)
    .str.zfill(4)
)

#converts float columns to integers where values have no decimals
for col in merged_df.columns:
    if merged_df[col].dtype == 'float64':
        if (merged_df[col].fillna(0) % 1 == 0).all():
            merged_df[col] = merged_df[col].astype('Int64')

print("\nFinal Data Types:")
print(merged_df.dtypes.to_string())

merged_df.to_csv("../Datasets/customer_info_final.csv", index=False)

