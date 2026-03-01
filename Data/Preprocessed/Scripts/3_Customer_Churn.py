import pandas as pd

df = pd.read_csv("../../Raw/3_Ecommerce Customer Churn Analysis and Prediction/E Commerce Dataset.csv")

print("Initial shape:", df.shape)

#drops unwanted columns
df = df.drop(columns=["Tenure", "WarehouseToHome", "NumberOfAddress", "PreferedOrderCat", "OrderCount", "Gender"])

print("Shape after dropping columns:", df.shape)

#renames columns
df = df.rename(columns={
    "CustomerID": "Customer_ID",
    "HourSpendOnApp": "Hours_Spend_On_Website",
    "NumberOfDeviceRegistered": "Total_Devices_Registered",
    "MaritalStatus": "Marital_Status",
    "Complain": "Complain_Count_In_Last_Month",
    "OrderAmountHikeFromlastYear": "Order_Amount_Hike_From_Last_Year",
    "CouponUsed": "Discount_Coupon_Used_Count",
    "PreferredPaymentMode": "Preferred_Payment_Mode",
    "PreferredLoginDevice": "Preferred_Login_Device",
    "CityTier": "City_Tier",
    "SatisfactionScore": "Satisfaction_Score",
    "DaySinceLastOrder": "Days_Since_Last_Order",
    "CashbackAmount": "Average_Cash_Back",
})

#cleans categories by merging redundant values
df["Preferred_Login_Device"] = df["Preferred_Login_Device"].replace("Phone", "Mobile Phone")

#cleans numeric values
#list of numeric columns to clean
numeric_cols = [
    "Hours_Spend_On_Website",
    "Order_Amount_Hike_From_Last_Year",
    "Discount_Coupon_Used_Count",
    "Days_Since_Last_Order",
    "Average_Cash_Back",
    "Total_Devices_Registered",
    "Satisfaction_Score"
]

for col in numeric_cols:
    #removes any non-numeric characters
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)

    df[col] = pd.to_numeric(df[col], errors="coerce")

    #fills nulls with median
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

#attribute type check
df["Average_Cash_Back"] = df["Average_Cash_Back"].astype(float)
int_cols = [c for c in numeric_cols if c != "Average_Cash_Back"]
df[int_cols] = df[int_cols].astype(int)

#data resizing and formatting
row_null_count = df.isnull().sum(axis=1)
df = df[row_null_count < 2]

#limits dataset to 5000 rows, prioritizing churned customers
total_rows = len(df)
rows_to_drop = total_rows - 5000
if rows_to_drop > 0:
    non_churn_rows = df[df["Churn"] == 0]
    if len(non_churn_rows) >= rows_to_drop:
        drop_indices = non_churn_rows.sample(n=rows_to_drop, random_state=42).index
        df = df.drop(drop_indices)

#maps customers to new ids and moves churn to end
df["Customer_ID"] = [str(i).zfill(4) for i in range(1, len(df) + 1)]
if "Churn" in df.columns:
    cols = [c for c in df.columns if c != "Churn"] + ["Churn"]
    df = df[cols]

print("Final shape:", df.shape)
df.to_csv("../Datasets/3_customer_churn_cleaned.csv", index=False)
