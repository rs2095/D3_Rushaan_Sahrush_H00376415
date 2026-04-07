import streamlit as st


def show():
    st.set_page_config(page_title="Customer AI Overview", layout="wide")

    st.title("Customer AI: Predicting Customer Dynamics and Business Outcomes")
    st.markdown("""
    **Project Overview:**  
    Customer AI is an intelligent platform designed to analyze customer behavior, predict churn, estimate customer lifetime value, and uncover purchasing patterns. Using advanced machine learning, it empowers organizations to make data-driven decisions that enhance engagement, retention, and profitability.
    """)

    st.markdown("---")
    st.subheader("Key Features")
    st.markdown("""
    - **Customer Segmentation and Profiling:** Identify distinct customer groups using clustering and behavioral analysis.  
    - **Churn Prediction and Retention:** Forecast churn risk and assign actionable retention scores.  
    - **Purchase Patterns and Cross-Selling:** Discover frequent product associations to drive targeted promotions.  
    - **Customer Lifetime Value Estimation :** Evaluate and rank customers by projected value with explainable insights.  
    - **Sustainability and Ethics Profiling:** Classify eco-conscious customers and compute Sustainability Impact Scores.  
    - **Scenario Testing and Simulation:** Run interactive simulations for churn, CLTV, and sales outcomes.  
    - **Integrated Dashboard:** Centralized visualizations combining all modules with drill-through functionality.  
    """)

    st.markdown("---")
    st.subheader("Technologies Used")
    st.markdown("""
    - **Python and Streamlit**: Interactive dashboard and web app  
    - **K-Means Clustering**: Customer segmentation  
    - **XGBoost (Classifier & Regressor)**: Churn prediction & CLV estimation  
    - **Decision Tree Classifier**: Rule-based insights  
    - **FP-Growth Algorithm**: Frequent pattern analysis  
    - **Predictive Simulation Models**: Scenario analysis and business outcome simulations  
    - **Firebase**: Authentication, role management, and feedback logging  
    """)

    st.markdown("---")
    st.info("Use the sidebar to navigate through dashboards, analytics, and simulation modules.")