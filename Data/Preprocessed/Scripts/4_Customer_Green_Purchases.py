import pandas as pd

OUTPUT_PATH = "../Datasets/4_customer_green_purchases_cleaned.csv"


df = pd.read_csv("../../Raw/4_Green Consumption Behavior Dataset/Green_Consumption_Behavior_2024.csv")

print("Initial shape:", df.shape)
print(f"Original rows: {len(df)}")

#drops rows with any null value
df = df.dropna().copy()
print(f"Rows after dropping nulls: {len(df)}")

#drops unwanted columns
df = df.drop(columns=["Age", "Gender", "Session_Duration_Minutes", "Device_Type", "Gender"])

#renames columns
df = df.rename(columns={"User_ID": "Customer_ID"})

#renumbers customer_id
df["Customer_ID"] = (
    range(1, len(df) + 1)
)

#formats customer_id as 4 digit strings
df["Customer_ID"] = df["Customer_ID"].apply(lambda x: f"{x:04d}")

#rounds off to 2 decimal places
round_columns = [
    "Time_Spent_on_Green_Products",
    "Green_Consumption_Score"
]

for col in round_columns:
    if col in df.columns:
        df[col] = df[col].round(2)

print("Final shape:", df.shape)
print(f"Final rows: {len(df)}")
print(f"Final columns: {list(df.columns)}")

df.to_csv("../Datasets/4_customer_green_purchases_cleaned.csv", index=False)

