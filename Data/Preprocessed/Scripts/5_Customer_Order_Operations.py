import pandas as pd

orders = pd.read_csv("../../Raw/5_Instacart Market Basket Analysis/orders.csv")
order_products = pd.read_csv("../../Raw/5_Instacart Market Basket Analysis/order_products__prior.csv")

#prints original orders and order_products row counts and unique order counts
print(f"Original orders rows: {len(orders)} | Unique orders: {orders['order_id'].nunique()}")
print(f"Original order_products rows: {len(order_products)} | Unique orders: {order_products['order_id'].nunique()}")

#drops column
if 'days_since_prior_order' in orders.columns:
    orders = orders.drop(columns=['days_since_prior_order'])

#drops rows with any null value
orders = orders.dropna()
order_products = order_products.dropna()

#filters orders to keep only those that have products
valid_order_ids = order_products["order_id"].unique()
orders = orders[orders["order_id"].isin(valid_order_ids)].copy()
print(f"\nOrders after filtering for valid product links: {len(orders)} | Unique orders: {orders['order_id'].nunique()}")

#counts orders per customer
orders_per_customer = orders.groupby("user_id")["order_id"].count().reset_index()
orders_per_customer.columns = ["user_id", "num_orders"]

#separates customers by order counts
customers_3plus = orders_per_customer[orders_per_customer["num_orders"] >= 3]
customers_2 = orders_per_customer[orders_per_customer["num_orders"] == 2]
customers_1 = orders_per_customer[orders_per_customer["num_orders"] == 1]

#filters 5000 customers by prioritizing 3+ orders, slightly more 2 order than 1 order
num_needed = 5000 - len(customers_3plus)
num_1_order = min(len(customers_1), 1000)  #ensure at least 1000 with 1 order
num_2_order = min(len(customers_2), num_needed - num_1_order)

#makes 2 order customers slightly more than 1 order
if num_2_order < num_1_order:
    extra = min(len(customers_2) - num_2_order, num_1_order - num_2_order + 100)
    num_2_order += extra
    num_1_order = num_needed - num_2_order

sampled_2 = customers_2.sample(num_2_order, random_state=42)
sampled_1 = customers_1.sample(num_1_order, random_state=42)
selected_customers = pd.concat([customers_3plus, sampled_2, sampled_1])

#filters orders dataset to selected customers
filtered_orders = orders[orders["user_id"].isin(selected_customers["user_id"])].copy()

#matches order_products_prior with filtered orders
order_products = order_products[order_products["order_id"].isin(filtered_orders["order_id"].unique())].copy()
filtered_orders = filtered_orders[filtered_orders["order_id"].isin(order_products["order_id"].unique())].copy()

#sequentially remaps order_ids
filtered_orders = filtered_orders.sort_values(by="order_id").reset_index(drop=True)
order_id_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(filtered_orders["order_id"].unique())}
filtered_orders["order_id_new"] = filtered_orders["order_id"].map(order_id_mapping)
order_products["order_id_new"] = order_products["order_id"].map(order_id_mapping)

#sequentially remaps customer_ids
filtered_orders = filtered_orders.sort_values(by="user_id").reset_index(drop=True)
user_id_mapping = {old_id: f"{new_id:04d}" for new_id, old_id in enumerate(filtered_orders["user_id"].unique(), start=1)}
filtered_orders["user_id_new"] = filtered_orders["user_id"].map(user_id_mapping)

#cleans and renames for orders dataset
final_orders = filtered_orders.drop(columns=[
    "order_id", "user_id", "eval_set", "order_number",
    "order_dow", "order_hour_of_day"  # Removed "Order_ID_New"
])
final_orders = final_orders.rename(columns={
    "order_id_new": "Order_ID",
    "user_id_new": "Customer_ID"
})

#cleans and renames for order_products dataset
final_order_products = order_products.drop(columns=[
    "order_id", "add_to_cart_order"
])
final_order_products = final_order_products.rename(columns={
    "product_id": "Product_ID",
    "order_id_new": "Order_ID"
})

print(f"Total customers: {final_orders['Customer_ID'].nunique()}")
print(f"Total orders: {final_orders['Order_ID'].nunique()}")
print("\nOrder count distribution per customer:")
print(final_orders.groupby("Customer_ID")["Order_ID"].count().value_counts().sort_index())

final_orders.to_csv("../Datasets/5_customer_orders_frequency_cleaned.csv", index=False)
final_order_products.to_csv("../Datasets/5_order_product_reorders_cleaned.csv", index=False)

