import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

df_main = pd.read_csv("../Data/Processed/TrainTest/train_customer_info_final.csv")
df_sust = pd.read_csv("Outputs/final_sustainability_profiles_full.csv")

#creates engineered features for segmentation
df_main['Purchase_Intensity'] = (df_main['Total_Orders_This_Month'] * df_main['Total_Amount_Spent']) / 1000
df_main['Recency'] = df_main['Days_Since_Last_Order']
df_main['Avg_Spend_Per_Order'] = df_main['Total_Amount_Spent'] / (df_main['Total_Orders_This_Month'] + 1)

features = ['Purchase_Intensity', 'Recency', 'Avg_Spend_Per_Order']
X = df_main[features].copy()

#captures and saves training medians for future use
train_medians = X.median()
X = X.fillna(train_medians)

#applies yeo-johnson transformation and feature weighting
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X)

X_transformed[:, 0] *= 1.0
X_transformed[:, 1] *= 1.5
X_transformed[:, 2] *= 3.0

# evaluates optimal k using silhouette score and davies-bouldin index
k_range = range(2, 9)
sil_scores = []
db_scores = []

for k in k_range:
    km_temp = KMeans(n_clusters=k, init='k-means++', n_init='auto', random_state=42)
    temp_labels = km_temp.fit_predict(X_transformed)
    sil_scores.append(silhouette_score(X_transformed, temp_labels))
    db_scores.append(davies_bouldin_score(X_transformed, temp_labels))

print("K | Silhouette | Davies-Bouldin")
for k, s, d in zip(k_range, sil_scores, db_scores):
    print(f"{k} | {s:.3f}      | {d:.3f}")

#trains kmeans clustering and measures training time
start_train = time.time()
kmeans = KMeans(n_clusters=4, init='k-means++', n_init='auto', random_state=42)
df_main['KMeans_Segment'] = kmeans.fit_predict(X_transformed)
end_train = time.time()

print(f"Training Time: {end_train - start_train:.2f}s")

#assigns segment labels based on cluster statistics
stats = df_main.groupby('KMeans_Segment').agg({
    'Total_Amount_Spent': 'mean',
    'Recency': 'mean',
    'Total_Orders_This_Month': 'mean'
})

labels = {}
available_segments = list(stats.index)

loyal_id = stats.loc[available_segments, 'Total_Orders_This_Month'].idxmax()
labels[loyal_id] = "Frequent Loyal Customers"
available_segments.remove(loyal_id)

lapsed_id = stats.loc[available_segments, 'Recency'].idxmax()
labels[lapsed_id] = "At-Risk Customers"
available_segments.remove(lapsed_id)

high_value_id = stats.loc[available_segments, 'Total_Amount_Spent'].idxmax()
labels[high_value_id] = "High-Value Customers"
available_segments.remove(high_value_id)

labels[available_segments[0]] = "Value Seeking Customers"

df_main['Segment_Label'] = df_main['KMeans_Segment'].map(labels)

#merges sustainability profiles
df_sust['Pred_Green_Score'] = df_sust['Pred_Green_Score'].round(2)
df_final = df_main.merge(df_sust, on='Customer_ID', how='left')

summary_main = df_final.groupby('Segment_Label').agg({
    'Customer_ID': 'count',
    'Total_Amount_Spent': 'mean',
    'Days_Since_Last_Order': 'mean',
    'Satisfaction_Score': 'mean',
    'Total_Orders_This_Month': 'mean'
}).rename(columns={'Customer_ID': 'Count'})

#calculates eco distribution per segment
eco_subset = df_final[df_final['Eco_Segment'].notna()].copy()
eco_subset['Eco_Segment'] = eco_subset['Eco_Segment'].str.replace('-', '_')

eco_dist = eco_subset.groupby(['Segment_Label', 'Eco_Segment'])['Customer_ID'].count().unstack(fill_value=0)

for col in ['Eco_Conscious', 'Eco_Warrior', 'Low_Interest']:
    if col not in eco_dist.columns:
        eco_dist[col] = 0

eco_dist['Total_Eco'] = eco_dist.sum(axis=1)
eco_dist['Eco_Strength'] = np.where(
    eco_dist['Total_Eco'] > 0,
    ((2 * eco_dist['Eco_Warrior']) +
     (1 * eco_dist['Eco_Conscious']) -
     (2 * eco_dist['Low_Interest'])) / eco_dist['Total_Eco'],
    0
)
eco_dist['Eco_Strength_Pct'] = ((eco_dist['Eco_Strength'] + 1) / 3) * 100

summary_combined = summary_main.merge(
    eco_dist[['Eco_Conscious', 'Eco_Warrior', 'Low_Interest', 'Eco_Strength_Pct']],
    left_index=True, right_index=True, how='left'
).fillna(0)

#prints segmentation and eco-strength report
sil = silhouette_score(X_transformed, df_main['KMeans_Segment'])
db = davies_bouldin_score(X_transformed, df_main['KMeans_Segment'])

print(f"\nSilhouette Score: {sil:.3f}")
print(f"Davies-Bouldin Index: {db:.3f}")

for label, row in summary_combined.iterrows():
    print(f"\n{label} ({int(row['Count'])} Customers):")
    print(f"Avg Spend: ${row['Total_Amount_Spent']:.2f} | Avg Orders: {row['Total_Orders_This_Month']:.2f}")
    print(f"Recency: {int(row['Days_Since_Last_Order'])} days | Satisfaction: {row['Satisfaction_Score']:.2f}/5")
    print(f"Eco-Strength: {row['Eco_Strength_Pct']:.1f}%")
    print(f"Warriors: {int(row['Eco_Warrior'])} | Conscious: {int(row['Eco_Conscious'])} | Low: {int(row['Low_Interest'])}")

#pca visualization of customer segments
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_transformed)
df_main['PCA1'], df_main['PCA2'] = X_pca[:, 0], X_pca[:, 1]

plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_main, x='PCA1', y='PCA2', hue='Segment_Label', palette='viridis', alpha=0.6, s=80)
plt.title("Customer Segments PCA Visualization (K=4)", fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#saves models, transformer and medians
joblib.dump(kmeans, 'Outputs/kmeans_model.pkl')
joblib.dump(pt, 'Outputs/power_transformer.pkl')
joblib.dump(train_medians, 'Outputs/segmentation_medians.pkl')

summary_combined.to_csv("Outputs/segment_eco_lookup_full.csv", index=True)
df_final.to_csv("Outputs/customer_segments_full.csv", index=False)