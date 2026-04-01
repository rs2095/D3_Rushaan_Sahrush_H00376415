import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, mean_absolute_error, r2_score, confusion_matrix, roc_auc_score
)

train_df = pd.read_csv("../Data/Processed/TrainTest/train_sustainable_customer_final.csv")
test_df = pd.read_csv("../Data/Processed/TrainTest/test_customer_info_eval.csv")

#defines categorical and numeric feature columns
cat_features = ['Referral_Source']
num_features = [
    'Cultural_Eco_Alignment', 'Emotional_Guilt_Score', 'Emotional_Pride_Score',
    'Social_Influence_Score', 'Green_Knowledge_Level', 'Time_Spent_on_Green_Products',
    'Clicks_on_Green_Products', 'Previous_Green_Purchases'
]

features = cat_features + num_features

#preprocesses with one-hot encoding and passthrough for numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
        ('num', 'passthrough', num_features)
    ]
)

#sets up classification and regression pipelines
clf_pipe = Pipeline([
    ('prepro', preprocessor),
    ('model', DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42))
])

reg_pipe = Pipeline([
    ('prepro', preprocessor),
    ('model', DecisionTreeRegressor(max_depth=5, random_state=42))
])

X_train = train_df[features]
y_class_train = train_df['Green_Purchase_Made']
y_reg_train = train_df['Green_Consumption_Score']

#drops rows with missing critical values in test set
test_df_clean = test_df.dropna(subset=num_features).copy()
X_test = test_df_clean[features]
y_class_test = test_df_clean['Green_Purchase_Made']
y_reg_test = test_df_clean['Green_Consumption_Score']

print(f"Training on {len(X_train)} rows")
print(f"Testing on {len(X_test)} customers with eco data")

#trains classifier and regressor and measures training time
start_train_clf = time.time()
clf_pipe.fit(X_train, y_class_train)
end_train_clf = time.time()

start_train_reg = time.time()
reg_pipe.fit(X_train, y_reg_train)
end_train_reg = time.time()

print(f"Classifier Training Time: {end_train_clf - start_train_clf:.2f}s")
print(f"Regressor Training Time: {end_train_reg - start_train_reg:.2f}s")

#evaluates classifier on test set and measures prediction time
start_pred_c = time.time()
y_pred_c = clf_pipe.predict(X_test)
y_prob_c = clf_pipe.predict_proba(X_test)[:, 1]
end_pred_c = time.time()

print(f"Classifier Prediction Time: {end_pred_c - start_pred_c:.2f}s")

print("\nClassification Test Set Performance:")
print(classification_report(y_class_test, y_pred_c))
print(f"ROC-AUC: {roc_auc_score(y_class_test, y_prob_c):.3f}")

#classification confusion matrix
cm = confusion_matrix(y_class_test, y_pred_c)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['No Green Purchase', 'Green Purchase'],
            yticklabels=['No Green Purchase', 'Green Purchase'])
plt.title('Sustainability Classifier - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

#evaluates regressor on test set and measures prediction time
start_pred_r = time.time()
y_pred_r = reg_pipe.predict(X_test)
end_pred_r = time.time()

print(f"Regressor Prediction Time: {end_pred_r - start_pred_r:.2f}s")

print("\nRegression Test Set Performance:")
print(f"R2 Score: {r2_score(y_reg_test, y_pred_r):.3f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_reg_test, y_pred_r):.3f}")

#actual vs predicted scatter plot for regression
plt.figure(figsize=(8, 6))
plt.scatter(y_reg_test, y_pred_r, alpha=0.4, s=20)
plt.plot([y_reg_test.min(), y_reg_test.max()],
         [y_reg_test.min(), y_reg_test.max()], 'r--', lw=1.5)
plt.title('Green Consumption Score - Actual vs Predicted')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.tight_layout()
plt.show()

#generates eco profiles and segments from training data
df_profiles = train_df.copy()
df_profiles['Pred_Green_Class'] = clf_pipe.predict(X_train)
df_profiles['Pred_Green_Score'] = reg_pipe.predict(X_train).clip(0, 1).round(2)

#assigns eco segment labels
def eco_label(row):
    if row['Pred_Green_Class'] == 1 and row['Pred_Green_Score'] >= 0.7:
        return "Eco_Warrior"
    elif row['Pred_Green_Class'] == 1 and 0.4 <= row['Pred_Green_Score'] < 0.7:
        return "Eco_Conscious"
    else:
        return "Low_Interest"

df_profiles['Eco_Segment'] = df_profiles.apply(eco_label, axis=1)

#saves eco profiles dataset
df_profiles.to_csv("Outputs/final_sustainability_profiles_full.csv", index=False)

#saves pipelines for reuse
joblib.dump(clf_pipe, 'Outputs/eco_classification_pipe.pkl')
joblib.dump(reg_pipe, 'Outputs/eco_regression_pipe.pkl')

#helper function to get feature names and transformed data for shap
def get_shap_ready_data(pipeline, X_input):
    prepro = pipeline.named_steps['prepro']
    ohe_names = prepro.named_transformers_['cat'].get_feature_names_out(cat_features).tolist()
    all_feature_names = ohe_names + num_features
    X_transformed = prepro.transform(X_input)
    return X_transformed, all_feature_names

#generates shap values for classification
X_test_trans_c, feature_names_c = get_shap_ready_data(clf_pipe, X_test)
explainer_c = shap.TreeExplainer(clf_pipe.named_steps['model'])
shap_values_c = explainer_c.shap_values(X_test_trans_c)

#handles binary classification shap output shape
if isinstance(shap_values_c, list):
    shap_to_plot_c = shap_values_c[1]
else:
    shap_to_plot_c = shap_values_c[:, :, 1] if len(shap_values_c.shape) == 3 else shap_values_c

#shap bar plot for classification
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_to_plot_c, X_test_trans_c, feature_names=feature_names_c,
                  plot_type="bar", show=False, color='#6A0DAD')
plt.title("Top 10 Drivers of Green Purchase (Classification)")
plt.xlabel("Mean Absolute SHAP Value (Impact on Prediction)")
plt.tight_layout()
plt.show()

#shap beeswarm plot for classification
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_to_plot_c,
    X_test_trans_c,
    feature_names=feature_names_c,
    show=False
)
plt.title("SHAP Beeswarm: Feature Impact on Green Purchase (Classification)")
plt.tight_layout()
plt.show()

#generates shap values for regression
X_test_trans_r, feature_names_r = get_shap_ready_data(reg_pipe, X_test)
explainer_r = shap.TreeExplainer(reg_pipe.named_steps['model'])
shap_values_r = explainer_r.shap_values(X_test_trans_r)

#shap bar plot for regression
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_r, X_test_trans_r, feature_names=feature_names_r,
                  plot_type="bar", show=False, color='#8A2BE2')
plt.title("Top 10 Drivers of Consumption Score (Regression)")
plt.xlabel("Mean Absolute SHAP Value (Impact on Prediction)")
plt.tight_layout()
plt.show()

#shap beeswarm plot for regression
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values_r,
    X_test_trans_r,
    feature_names=feature_names_r,
    show=False
)
plt.title("SHAP Beeswarm: Feature Impact on Consumption Score (Regression)")
plt.tight_layout()
plt.show()