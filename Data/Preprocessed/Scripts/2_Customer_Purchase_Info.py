import pandas as pd

df = pd.read_csv("../../Raw/2_E-commerce Customer Data For Behavior Analysis/ecommerce_customer_data_custom_ratios.csv")

print("Initial shape:", df.shape)

#drops unwanted columns
df = df.drop(columns=["Purchase Date", "Product Price", "Quantity", "Payment Method", "Customer Name", "Age", "Gender", "Churn", "Product Category"])

print("Shape after dropping columns:", df.shape)

#renames columns
df = df.rename(columns={
    "Customer ID": "Customer_ID",
    "Total Purchase Amount": "Amount_Spent",
    "Customer Age": "Customer_Age",
    "Returns": "Order_Returns"
})

#drops rows with any nulls
df = df.dropna()
print("Shape after dropping rows with nulls:", df.shape)

#keeps only first 5000 unique customers
unique_customers = sorted(df["Customer_ID"].unique())[:5000]  #sorts ids first
df = df[df["Customer_ID"].isin(unique_customers)]

#maps customers to new ids
customer_id_mapping = {old_id: str(i+1).zfill(4) for i, old_id in enumerate(unique_customers)}
df["Customer_ID"] = df["Customer_ID"].map(customer_id_mapping)

#sorts df by new customer_id
df = df.sort_values(by="Customer_ID").reset_index(drop=True)

#converts order_return data type
df["Order_Returns"] = df["Order_Returns"].astype(int)

print("Final shape after keeping 5000 customers:", df.shape)
df.to_csv("../Datasets/2_customer_purchase_info_cleaned.csv", index=False)

