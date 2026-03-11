import pandas as pd
from sklearn.model_selection import train_test_split

df_main = pd.read_csv("../Datasets/customer_info_final.csv")
df_sust = pd.read_csv("../Datasets/sustainable_customer_final.csv")
orders = pd.read_csv("../Datasets/mb_customer_orders_final.csv")
order_products = pd.read_csv("../Datasets/mb_order_product_map_final.csv")
products = pd.read_csv("../Datasets/mb_products_final.csv")
department = pd.read_csv("../Datasets/mb_departments_final.csv")

#splits dataset into 80/20 train test
train_main, test_main = train_test_split(
    df_main, test_size=0.20, stratify=df_main['Churn'], random_state=42
)

train_green = df_sust[df_sust['Customer_ID'].isin(train_main['Customer_ID'])]

test_final = test_main.merge(df_sust, on='Customer_ID', how='left')

#drops columns in test set
cols_to_drop = ['Churn', 'Customer_Live_Time_Value', 'Green_Consumption_Score', 'Green_Purchase_Made']
test_final= test_final.drop(columns=cols_to_drop, errors='ignore')

#filters and merges test customer orders with products and departments into one dataset
test_customer_ids = test_main['Customer_ID'].unique()
test_mb_orders = orders[orders['Customer_ID'].isin(test_customer_ids)]

test_order_ids = test_mb_orders['Order_ID'].unique()
test_mb_map_raw = order_products[order_products['Order_ID'].isin(test_order_ids)]

test_mb_final = test_mb_map_raw.merge(
    products[["Product_ID", "Department_ID"]], on="Product_ID", how="left"
).merge(
    department[["Department_ID", "Department_Name"]], on="Department_ID", how="left"
).merge(
    test_mb_orders[["Order_ID", "Customer_ID"]], on="Order_ID", how="left"
)

#saves final datasets
train_main.to_csv("../TrainTest/train_customer_info_final.csv", index=False)
train_green.to_csv("../TrainTest/train_sustainable_customer_final.csv", index=False)
test_final.to_csv("../TrainTest/test_customer_info.csv", index=False)
test_mb_final.to_csv("../TrainTest/test_market_basket.csv", index=False)
