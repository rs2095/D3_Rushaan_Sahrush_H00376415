import pandas as pd

df = pd.read_csv("../../Raw/6_Predict CLTV of a Customer/train_BRCpofr.csv")

#drops rows with any null value
df = df.dropna()
  
#renames columns
df = df.rename(columns={
    "id": "Customer_ID",
    "area": "Customer_Residence_Type",
    "type_of_policy": "Membership_Type",
    "income": "Customer_Income",
    "cltv": "Customer_Live_Time_Value",
    "gender": "Customer_Gender",
    "qualification": "Customer_Qualification",
    "num_policies": "Total_Subscriptions",
    "claim_amount": "Loyalty_Engagement_Score"

})

#drops unwanted columns
df = df.drop(columns=["marital_status", "policy", "vintage"])


#replaces 'L' with 'K' in customer_income
df["Customer_Income"] = df["Customer_Income"].str.replace("L", "K")
  
#sorts by customer_id
df = df.sort_values(by="Customer_ID").reset_index(drop=True)

#reassigns customer_id and keeps first 5000
df = df.iloc[:5000]
df["Customer_ID"] = [f"{i:04d}" for i in range(1, len(df) + 1)]

df.to_csv("../Datasets/6_customer_cltv_cleaned.csv", index=False)
