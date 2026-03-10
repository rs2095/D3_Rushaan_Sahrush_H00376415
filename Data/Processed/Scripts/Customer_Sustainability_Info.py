import pandas as pd

df = pd.read_csv("../../Preprocessed/Datasets/4_customer_green_purchases_cleaned.csv")

print("Initial shape:", df.shape)

#drops income level
df = df.drop(columns=["Income_Level"])

#rounds time_spent_on_green_products to nearest integer
df["Time_Spent_on_Green_Products"] = (
    df["Time_Spent_on_Green_Products"]
    .round()
    .astype(int)
)

print("\nFinal shape:", df.shape)
print("\nFinal columns:")

df.to_csv("../Datasets/sustainable_customer_final.csv", index=False)

