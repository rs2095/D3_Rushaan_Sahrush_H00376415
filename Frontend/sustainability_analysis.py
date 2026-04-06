import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
from styles import inject_styles
from audit_logger import log_event


def show():
    st.set_page_config(page_title="Sustainability Intelligence", layout="wide")
    inject_styles()

    st.title("Sustainability Analysis Dashboard")
    st.markdown("Green Behavior Prediction and Eco Profiling")
    st.markdown("<br>", unsafe_allow_html=True)

    #checks if dataset exists from main dashboard
    if 'df' not in st.session_state or st.session_state.get('uploaded_file') is None:
        keys_to_clear = ['eco_clf', 'eco_reg', 'eco_models_loaded', 'eco_logged_hash',
                         'eco_user_comments']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]

        st.warning("Please upload a dataset from the Executive Dashboard first.")
        return

    df = st.session_state.df.copy()

    #logs page navigation event
    current_file = st.session_state.get("uploaded_file", None)
    if current_file is None:
        st.warning("Please upload a dataset from the Executive Dashboard first.")
        return

    current_hash = st.session_state.get("last_uploaded_hash")
    if st.session_state.get("last_file_hash") != current_hash:
        st.session_state.eco_logged_hash = None
        st.session_state.last_file_hash = current_hash

    if current_hash and st.session_state.get("eco_logged_hash") != current_hash:
        log_event(
            action_type="run_analysis",
            page_name="Sustainability Intelligence",
            file_name=current_file,
            details=f"User opened Sustainability page and eco models ran on {len(df)} records"
        )
        st.session_state.eco_logged_hash = current_hash

    #loads eco models once
    if 'eco_models_loaded' not in st.session_state:
        st.session_state.eco_clf = joblib.load("eco_classification_pipe.pkl")
        st.session_state.eco_reg = joblib.load("eco_regression_pipe.pkl")
        st.session_state.eco_models_loaded = True

    eco_clf = st.session_state.eco_clf
    eco_reg = st.session_state.eco_reg

    #extracts required feature columns from model pipeline
    cat_cols = eco_clf.named_steps['prepro'].transformers_[0][2]
    num_cols = eco_clf.named_steps['prepro'].transformers_[1][2]
    required_cols = cat_cols + num_cols

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Missing required eco features: {missing_cols}")
        return

    #filters valid rows for prediction
    valid_mask = df[required_cols].notnull().all(axis=1)
    num_invalid = (~valid_mask).sum()
    pred_df = df[valid_mask].copy()
    pred_df.reset_index(drop=True, inplace=True)

    #generates eco predictions
    X = pred_df[required_cols].copy()
    pred_df['Pred_Green_Class'] = eco_clf.predict(X)
    pred_df['Pred_Green_Score'] = eco_reg.predict(X).clip(0, 1)
    pred_df['Green_Score_%'] = (pred_df['Pred_Green_Score'] * 100).round(1)

    #assigns eco segment labels
    def eco_label(row):
        if row['Pred_Green_Class'] == 1 and row['Pred_Green_Score'] >= 0.7:
            return "Eco_Warrior"
        elif row['Pred_Green_Class'] == 1 and 0.4 <= row['Pred_Green_Score'] < 0.7:
            return "Eco_Conscious"
        else:
            return "Low_Interest"

    pred_df['Eco_Segment'] = pred_df.apply(eco_label, axis=1)

    #merges predictions back into main dataframe
    df.loc[valid_mask, 'Pred_Green_Class'] = pred_df['Pred_Green_Class']
    df.loc[valid_mask, 'Pred_Green_Score'] = pred_df['Pred_Green_Score']
    df.loc[valid_mask, 'Green_Score_%'] = pred_df['Green_Score_%']
    df.loc[valid_mask, 'Eco_Segment'] = pred_df['Eco_Segment']

    #renders kpi card row
    def render_kpis(kpi_list, border_color):
        cols = st.columns(len(kpi_list))
        for i, kpi in enumerate(kpi_list):
            with cols[i]:
                st.markdown(f"""
                <div class="sust-card {border_color}">
                    <p class="sust-kpi-title">{kpi['label']}</p>
                    <p class="sust-kpi-value">{kpi['value']}</p>
                    <p class="sust-kpi-sub">{kpi['unit']}</p>
                </div>
                """, unsafe_allow_html=True)

    #generates shap bar plot for a given model
    def shap_bar_plot(model, prepro, X, cat_cols, num_cols, color):
        X_trans = prepro.transform(X)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_trans)
        if isinstance(shap_vals, list):
            shap_impact = np.abs(shap_vals[1]).mean(axis=0)
        else:
            shap_impact = np.abs(shap_vals).mean(axis=0)
        shap_impact = np.ravel(shap_impact)
        ohe_names = prepro.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
        feature_names = ohe_names + num_cols
        top_features = sorted(zip(feature_names, shap_impact), key=lambda x: x[1], reverse=True)[:8]
        names = [x[0] for x in top_features]
        values = [x[1] for x in top_features]
        fig = px.bar(x=values, y=names, orientation='h', text=[f"{v:.3f}" for v in values])
        fig.update_traces(marker_color=color, textposition='auto')
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', height=400, yaxis={'categoryorder': 'total ascending'})
        return fig

    #displays green engagement kpis
    st.markdown("### Green Engagement Metrics")
    green_kpis = [
        {"label": "Avg Green Knowledge", "value": f"{pred_df['Green_Knowledge_Level'].mean():.1f}",
         "unit": "Customer knowledge on eco-products"},
        {"label": "Avg Time Browsing Green", "value": f"{pred_df['Time_Spent_on_Green_Products'].mean():.1f}",
         "unit": "Minutes spent exploring eco-products"},
        {"label": "Avg Clicks on Eco-Products", "value": f"{pred_df['Clicks_on_Green_Products'].mean():.1f}",
         "unit": "Number of eco-products interested in"},
        {"label": "Avg Predicted Green Score", "value": f"{pred_df['Pred_Green_Score'].mean()*100:.1f}",
         "unit": "Predicted green consumption %"}
    ]
    render_kpis(green_kpis, "sust-border-green-600")
    st.markdown("<br>", unsafe_allow_html=True)

    #displays emotional and social influence kpis
    st.markdown("### Emotional and Social Influence")
    emotional_kpis = [
        {"label": "Avg Eco Alignment", "value": f"{pred_df['Cultural_Eco_Alignment'].mean()/10:.2f}",
         "unit": "Customer eco-products preference"},
        {"label": "Avg Emotional Pride", "value": f"{pred_df['Emotional_Pride_Score'].mean():.1f}",
         "unit": "Positive eco driver (/10)"},
        {"label": "Avg Emotional Guilt", "value": f"{pred_df['Emotional_Guilt_Score'].mean():.1f}",
         "unit": "Negative eco driver (/10)"},
        {"label": "Avg Social Influence", "value": f"{pred_df['Social_Influence_Score'].mean():.1f}",
         "unit": "Community influence on behavior (/10)"}
    ]
    render_kpis(emotional_kpis, "sust-border-green-500")
    st.markdown("<br>", unsafe_allow_html=True)

    #displays eco segment, purchase, and score charts
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### Eco Segment Breakdown")
        fig_segment = px.pie(pred_df, names='Eco_Segment', color='Eco_Segment',
                             color_discrete_map={'Eco_Warrior': '#1B5E20',
                                                 'Eco_Conscious': '#66BB6A',
                                                 'Low_Interest': '#C8E6C9'})
        st.plotly_chart(fig_segment, use_container_width=True)

    with col2:
        st.markdown("### Green Purchase Distribution")
        counts = pred_df['Pred_Green_Class'].value_counts().sort_index()
        fig_purchase = go.Figure(data=[go.Bar(
            x=['No', 'Yes'],
            y=counts.values,
            marker_color=['#C8E6C9', '#2E7D32'],
            text=counts.values,
            textposition='auto'
        )])
        fig_purchase.update_layout(xaxis_title="Predicted Green Purchase",
                                   yaxis_title="Customers",
                                   plot_bgcolor='rgba(0,0,0,0)',
                                   height=400)
        st.plotly_chart(fig_purchase, use_container_width=True)

    with col3:
        st.markdown("### Green Score Distribution")
        fig_score = px.histogram(pred_df, x='Green_Score_%', nbins=30, color_discrete_sequence=['#43A047'])
        fig_score.update_layout(xaxis_title="Green Score (%)",
                                yaxis_title="Customers",
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=400)
        st.plotly_chart(fig_score, use_container_width=True)

    #displays shap analysis for classification and regression
    st.title("Top Green Drivers (SHAP)")
    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        st.markdown("### Drivers of Green Purchase")
        fig_shap_clf = shap_bar_plot(eco_clf.named_steps['model'], eco_clf.named_steps['prepro'],
                                     X, cat_cols, num_cols, '#2E7D32')
        st.plotly_chart(fig_shap_clf, use_container_width=True)

    with shap_col2:
        st.markdown("### Drivers of Green Consumption Score")
        fig_shap_reg = shap_bar_plot(eco_reg.named_steps['model'], eco_reg.named_steps['prepro'],
                                     X, cat_cols, num_cols, '#66BB6A')
        st.plotly_chart(fig_shap_reg, use_container_width=True)

    #displays record count notifications
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.success(f"{len(pred_df):,} records loaded from main dashboard")
    if num_invalid > 0:
        st.warning(f"{num_invalid:,} customer records are missing eco info and cannot be predicted.")

    #displays customer drill-through table
    st.title("Customer Drill-Through")
    display_cols = ['Customer_ID', 'Cultural_Eco_Alignment', 'Green_Knowledge_Level',
                    'Time_Spent_on_Green_Products', 'Clicks_on_Green_Products',
                    'Previous_Green_Purchases', 'Pred_Green_Score', 'Pred_Green_Class',
                    'Eco_Segment']
    st.dataframe(pred_df[display_cols].sort_values('Pred_Green_Score', ascending=False),
                 use_container_width=True)

    #saves updated dataframe to session state
    st.session_state.df = df

    #exports sustainability results with optional comments
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.title("Export Sustainability Results")

    export_df = pred_df.copy()

    if "eco_user_comments" not in st.session_state:
        st.session_state.eco_user_comments = []

    export_df['Comment'] = ""

    if st.session_state.eco_user_comments:
        comments_df = pd.DataFrame({
            "Customer_ID": [""] * len(st.session_state.eco_user_comments),
            "Comment": st.session_state.eco_user_comments
        })
        export_df = pd.concat([export_df, comments_df], ignore_index=True)

    csv_data = export_df.to_csv(index=False).encode("utf-8")

    #logs sustainability report download event
    def log_sustainability_download():
        current_file = st.session_state.get("uploaded_file", "unknown.csv")
        log_event(
            action_type="download",
            page_name="Sustainability Intelligence",
            file_name=current_file,
            details=f"Sustainability results exported ({len(export_df)} rows)"
        )

    st.download_button(
        label="Download Sustainability Report",
        data=csv_data,
        file_name="sustainability_report.csv",
        mime="text/csv",
        on_click=log_sustainability_download,
        use_container_width=True
    )