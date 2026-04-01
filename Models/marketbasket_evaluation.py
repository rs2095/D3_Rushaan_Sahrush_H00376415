import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpgrowth_marketbasket_model import MarketBasketEngine

plt.style.use('seaborn-v0_8-whitegrid')
PRIMARY_COLOR = "teal"
sns.set_palette([PRIMARY_COLOR])

orders = pd.read_csv("../Data/Processed/Datasets/mb_customer_orders_final.csv")
order_products = pd.read_csv("../Data/Processed/Datasets/mb_order_product_map_final.csv")
products = pd.read_csv("../Data/Processed/Datasets/mb_products_final.csv")
departments = pd.read_csv("../Data/Processed/Datasets/mb_departments_final.csv")

#merges order products with product and department names
order_products = order_products.merge(products[['Product_ID', 'Department_ID']], on='Product_ID', how='left')
order_products = order_products.merge(departments[['Department_ID', 'Department_Name']], on='Department_ID', how='left')
order_products = order_products.merge(orders[['Order_ID', 'Customer_ID']], on='Order_ID', how='left')
order_products = order_products[['Order_ID', 'Department_Name']].dropna()

print(f"Merged Dataset Shape: {order_products.shape}")

#initializes market basket engine and generates rules with timing
engine = MarketBasketEngine(min_support=0.05, min_confidence=0.4, min_lift=1.3)

start_mine = time.time()
rules, frequent_itemsets, basket = engine.generate_rules(order_products)
end_mine = time.time()

print(f"Basket Matrix Shape: {basket.shape}")
print(f"Mining and Rule Generation Time: {end_mine - start_mine:.2f}s")

#prints top association rules
print("\nTop Cross-Sell Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15))

#prints rule summary statistics
print(f"\nTotal Frequent Itemsets: {len(frequent_itemsets)}")
print(f"Total Rules After Filtering: {len(rules)}")
print(f"Support Range: {rules['support'].min():.4f} - {rules['support'].max():.4f}")
print(f"Confidence Range: {rules['confidence'].min():.4f} - {rules['confidence'].max():.4f}")
print(f"Lift Range: {rules['lift'].min():.4f} - {rules['lift'].max():.4f}")
print(f"Mean Lift: {rules['lift'].mean():.4f}")
print(f"Mean Confidence: {rules['confidence'].mean():.4f}")

#association rules scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=rules,
    x='support',
    y='confidence',
    size='lift',
    hue='lift',
    palette='magma',
    sizes=(100, 500)
)
plt.title("Department-Level Association Rules")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#exports rules to csv
rules.to_csv("Outputs/cross_selling_rules_refined.csv", index=False)