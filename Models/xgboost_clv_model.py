import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap
import time
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

train_df = pd.read_csv("../Data/Processed/TrainTest/train_customer_info_final.csv")
test_df = pd.read_csv("../Data/Processed/TrainTest/test_customer_info_eval.csv")

#drops customer ID column
for df in [train_df, test_df]:
    if 'Customer_ID' in df.columns:
        df.drop(columns=['Customer_ID'], inplace=True)

#applies log transform to target variable
train_df['cltv_log'] = np.log1p(train_df['Customer_Live_Time_Value'])
test_df['cltv_log'] = np.log1p(test_df['Customer_Live_Time_Value'])

X_train = train_df.drop(columns=['Customer_Live_Time_Value', 'cltv_log'])
y_train = train_df['cltv_log']

X_test = test_df.drop(columns=['Customer_Live_Time_Value', 'cltv_log'])
y_test = test_df['cltv_log']

#defines categorical and numeric columns
categorical_cols = [
    'Marital_Status', 'Customer_Gender', 'Customer_Residence_Type',
    'Customer_Qualification', 'Customer_Income', 'Total_Subscriptions',
    'Membership_Type'
]

numeric_cols = [
    'Loyalty_Engagement_Score',
]

#preprocesses with one-hot encoding and robust scaling
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', RobustScaler(), numeric_cols)
    ]
)

#sets up xgboost regressor
xgb_model = XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    n_jobs=-1,
    tree_method='hist'
)

param_grid = {
    'regressor__n_estimators': [500, 1000, 1500],
    'regressor__learning_rate': [0.01, 0.05, 0.1],
    'regressor__max_depth': [3, 5, 7],
    'regressor__subsample': [0.7, 0.8, 0.9],
    'regressor__colsample_bytree': [0.7, 0.8, 0.9]
}

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', xgb_model)
])

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=15,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

#trains baseline model before tuning
pipeline.fit(X_train, y_train)

y_pred_base_log = pipeline.predict(X_test)

#calculates baseline log-scale metrics
base_log_r2 = r2_score(y_test, y_pred_base_log)

#calculates baseline original-scale metrics
y_test_orig_base = np.expm1(y_test)
y_pred_orig_base = np.expm1(y_pred_base_log)
base_mae_orig = mean_absolute_error(y_test_orig_base, y_pred_orig_base)

print("\nBaseline Test Set Performance:")
print(f"Log-Scale R2 Score: {base_log_r2:.4f}")
print(f"Mean Absolute Error: ${base_mae_orig:.2f}")

#trains model with hyperparameter tuning and measures training and prediction time
start_tune = time.time()
search.fit(X_train, y_train)
end_tune = time.time()
best_model = search.best_estimator_

print(f"Training Time: {end_tune - start_tune:.2f}s")

start_pred = time.time()
y_pred_log = best_model.predict(X_test)
end_pred = time.time()

print(f"Prediction Time: {end_pred - start_pred:.2f}s")

#calculates tuned model log-scale metrics
log_r2 = r2_score(y_test, y_pred_log)

#calculates tuned model original-scale metrics
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred_log)
mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)

print("\nTuned Model Test Set Performance:")
print(f"Log-Scale R2 Score: {log_r2:.4f}")
print(f"Mean Absolute Error: ${mae_orig:.2f}")

#actual vs predicted scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_orig, y_pred_orig, alpha=0.4, s=20)
plt.plot([y_test_orig.min(), y_test_orig.max()],
         [y_test_orig.min(), y_test_orig.max()], 'r--', lw=1.5)
plt.title('CLV - Actual vs Predicted')
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.tight_layout()
plt.show()

#saves tuned model
joblib.dump(best_model, 'Outputs/cltv_model.pkl')

#extracts preprocessing and model for shap
preprocessor = best_model.named_steps['preprocessor']
model = best_model.named_steps['regressor']

#gets feature names for shap analysis
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
all_feature_names = ohe_feature_names + numeric_cols

#transforms test set for shap
X_test_transformed = preprocessor.transform(X_test)

#generates shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transformed)

#shap summary plot
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=all_feature_names,
    plot_type="dot",
    show=False
)
plt.title("SHAP Summary: Drivers of Customer Lifetime Value (Log-Scale)")
plt.tight_layout()
plt.show()

#shap bar plot
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_test_transformed,
    feature_names=all_feature_names,
    plot_type="bar",
    show=False
)
plt.title("Global Feature Importance (Mean Absolute SHAP)")
plt.show()