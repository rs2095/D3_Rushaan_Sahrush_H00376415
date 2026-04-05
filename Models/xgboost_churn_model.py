import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import time

train_df = pd.read_csv("../Data/Processed/TrainTest/train_customer_info_final.csv")
test_df = pd.read_csv("../Data/Processed/TrainTest/test_customer_info_eval.csv")

target = "Churn"

keep_features = [
    'Preferred_Login_Device',
    'City_Tier',
    'Preferred_Payment_Mode',
    'Total_Devices_Registered',
    'Satisfaction_Score',
    'Marital_Status',
    'Complain_Count_In_Last_Month',
    'Days_Since_Last_Order',
    'Average_Cash_Back'
]

X_train = train_df[keep_features]
y_train = train_df[target]

X_test = test_df[keep_features]
y_test = test_df[target]

#defines categorical and numeric columns for preprocessing
categorical_cols = ['Preferred_Login_Device', 'Preferred_Payment_Mode', 'Marital_Status']
numeric_cols = [c for c in keep_features if c not in categorical_cols]

#preprocesses with one-hot encoding and robust scaling
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", RobustScaler(), numeric_cols)
    ]
)

#calculates dampened class weight ratio
neg, pos = sum(y_train == 0), sum(y_train == 1)
dampened_ratio = np.sqrt(neg / pos)

#sets up xgboost classifier
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=dampened_ratio,
    random_state=42
)

param_grid = {
    'model__n_estimators': [300, 500, 700],
    'model__max_depth': [4, 6, 8],
    'model__learning_rate': [0.01, 0.02, 0.05],
    'model__subsample': [0.7, 0.8, 0.9],
    'model__colsample_bytree': [0.7, 0.8]
}

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", xgb)
])

#trains baseline model before tuning
pipeline.fit(X_train, y_train)

y_pred_base = pipeline.predict(X_test)
y_prob_base = pipeline.predict_proba(X_test)[:, 1]

#baseline test set performance
print("\nBaseline Test Set Performance:")
print(classification_report(y_test, y_pred_base))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob_base):.4f}")

#baseline confusion matrix
cm_base = confusion_matrix(y_test, y_pred_base)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
plt.title('Baseline - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

#runs randomized search for hyperparameter tuning
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

cv_results = search.cv_results_
best_idx = search.best_index_
mean_score = cv_results['mean_test_score'][best_idx]
std_score = cv_results['std_test_score'][best_idx]
print(f"CV F1 Score: {mean_score:.4f} ± {std_score:.4f}")
best_pipeline = search.best_estimator_

#evaluates tuned model on test set
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

print("\nBest Parameters:")
print(search.best_params_)

print("\nTuned Model Test Set Performance:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

#tuned model confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'])
plt.title('Tuned Model - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

#trains model with hyperparameter tuning and measures training and prediction time
start_tune = time.time()
search.fit(X_train, y_train)
end_tune = time.time()
best_pipeline = search.best_estimator_

print(f"Training Time: {end_tune - start_tune:.2f}s")

start_pred = time.time()
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]
end_pred = time.time()

print(f"Prediction Time: {end_pred - start_pred:.2f}s")

#saves tuned model
joblib.dump(best_pipeline, 'Outputs/churn_model.pkl')

#extracts preprocessing and model for shap
preprocessor = best_pipeline.named_steps['preprocessing']
model = best_pipeline.named_steps['model']

#gets feature names for shap analysis
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
all_feature_names = cat_feature_names + numeric_cols

#transforms test set for shap
X_test_transformed = preprocessor.transform(X_test)

#generates shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transformed)

#shap summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_transformed, feature_names=all_feature_names)

#shap bar plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_transformed, feature_names=all_feature_names, plot_type="bar")