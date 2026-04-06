# D3_Rushaan_Sahrush_H00376415
AI-Driven Customer Analytics Platform
An interactive decision-support platform integrating predictive modelling, explainability, and scenario simulation for customer analytics.

=> Live Application: https://d3-rushaan-sahrush-h00376415.streamlit.app/

=> Getting Started 
- No local installation is needed - the platform runs entirely in the browser
- Navigate to the live application URL above to access the platform

=> Authentication
Sign Up
- Once you navigate to the home page, select the 'Sign Up' option
- Enter your email address and create a password
   - Select your role - 'Admin' or 'Analyst'
     - Admin: full access including the Scenario Simulator and audit management
     - Analyst: access to all model tabs, SHAP explanations, and data management
- Click 'Sign Up' to create your account
- No email verification is required - you can log in immediately after signing up

=> Log In
- Enter your registered email and password on the login page
- Click 'Login' to access the dashboard

=> Test Credentials
Pre-configured accounts are available for evaluation:

Role: Admin
- Email: admin@gmail.com
- Password: admin@ai

Role: Analyst
- Email: analyst@gmail.com
- Password: analyst@ai

=> Test Datasets

Test CSV files are provided in the project's Sample Datasets/ folder. Each platform tab requires a specific file:

customer_info_and_green_data.csv - Automatically runs segmentation, churn, CLV and sustainability profiling
market_basket_data.csv - Separate file containing order and product data 
Scenario Simulator (Requires the predictions CSV downloaded from the Prediction Dashboard)

- Scenario Simulator - dataset preparation:
  - Upload the customer_info_and_green_data.csv dataset in the 'Dashboard' tab
  - After the models run, download the predictions CSV from the dashboard
  - Upload that downloaded CSV into the 'Scenario Simulator' tab

=> Using the Platform
Dashboard
- Navigate to the 'Dashboard' tab
- Upload the customer_info_and_green_data.csv file and run analysis
- The platform automatically runs all applicable models: Churn, CLV, Segmentation, and Sustainability Profiling
- View predictions, segment distributions, and summary statistics
- Explore SHAP explanations to understand key drivers behind each prediction
- Download the predictions CSV for use in the Scenario Simulator or external analysis

Sustainability Analysis
- Navigate to the 'Sustainability' tab
- View predictions, segment distributions, and summary statistics
- Explore SHAP explanations to understand key drivers behind each prediction

Market Basket Analysis
- Navigate to the 'Market Basket Analysis' tab
- Upload the market_basket_data.csv file
- Adjust sliders to preference and run analysis
- View generated association rules, frequent itemsets, and cross-selling recommendations

Scenario Simulator (Admin only)
- Navigate to the 'Scenario Simulator' tab 
- Upload the predictions CSV downloaded from the Dashboard
- Adjust business levers using the interactive sliders:
  - Cashback: increase or decrease cashback incentives
  - Customer Support: simulate complaint reduction
  - Marketing / Pricing: adjust spending and pricing strategies
  - Subscription Push: simulate upselling campaigns
- View updated KPI cards showing projected churn rate, CLV changes, and revenue impact


