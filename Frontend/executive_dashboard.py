import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import shap
import hashlib
from styles import inject_styles
from audit_logger import log_event

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "Models", "Outputs")

#calculates eco strength percentage per segment
def calculate_eco_strength(df):

    if 'Eco_Segment' not in df.columns:
        return pd.DataFrame()

    eco_dist = (
        df.groupby(['Segment_Label', 'Eco_Segment'])['Customer_ID']
        .count()
        .unstack(fill_value=0)
    )

    for col in ['Eco_Conscious', 'Eco_Warrior', 'Low_Interest']:
        if col not in eco_dist.columns:
            eco_dist[col] = 0

    eco_dist['Total_Eco'] = eco_dist.sum(axis=1)

    eco_dist['Eco_Strength'] = np.where(
        eco_dist['Total_Eco'] > 0,
        ((2 * eco_dist['Eco_Warrior']) +
         (1 * eco_dist['Eco_Conscious']) -
         (2 * eco_dist['Low_Interest'])) /
        eco_dist['Total_Eco'],
        0
    )

    eco_dist['Eco_Strength_Pct'] = ((eco_dist['Eco_Strength'] + 1) / 3) * 100

    return eco_dist


REQUIRED_COLUMNS = [
    'Customer_ID',
    'Total_Orders_This_Month',
    'Total_Amount_Spent',
    'Days_Since_Last_Order',
    'Satisfaction_Score',
]


#validates uploaded dataset has required columns
def validate_dataset(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return len(missing) == 0, missing


def show():

    st.set_page_config(page_title="Customer Intelligence Platform", layout="wide")
    inject_styles()

    st.title("Customer Prediction Dashboard")
    st.markdown("360° View of Customer Value, Risk and Segmentation")
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Customer Dataset (CSV)", type=["csv"])

    if not uploaded_file:
        st.info("Upload a dataset to begin customer analysis.")
        st.stop()

    #computes file hash and logs upload event
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if st.session_state.get("last_uploaded_hash") != file_hash:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the uploaded file. Please ensure it is a valid CSV.\n\nError: {e}")
            st.stop()

        #validates required columns exist
        is_valid, missing_cols = validate_dataset(df)
        if not is_valid:
            st.error(
                "The uploaded CSV is missing required columns and cannot be processed.\n\n"
                f"**Missing columns:** {', '.join(missing_cols)}\n\n"
                f"**Your columns:** {', '.join(df.columns.tolist())}\n\n"
                "Please upload a dataset that contains at least: "
                f"{', '.join(REQUIRED_COLUMNS)}"
            )
            st.stop()

        st.session_state.df = df
        st.session_state.uploaded_file = uploaded_file.name
        st.session_state.last_uploaded_hash = file_hash

        #resets session artifacts for fresh analysis
        for key in ['df_churn', 'available_cols', 'segmentation_done', 'models_loaded', 'eco_models_loaded']:
            st.session_state.pop(key, None)

        log_event(
            action_type="upload",
            page_name="Customer Intelligence Platform",
            file_name=uploaded_file.name
        )

        st.rerun()
    else:
        df = st.session_state.df

        #loads core models once
        if 'models_loaded' not in st.session_state:
            try:
                st.session_state.kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
                st.session_state.pt = joblib.load(os.path.join(MODEL_DIR, "power_transformer.pkl"))
                st.session_state.medians = joblib.load(os.path.join(MODEL_DIR, "segmentation_medians.pkl"))
                st.session_state.churn_model = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
                st.session_state.cltv_model = joblib.load(os.path.join(MODEL_DIR, "cltv_model.pkl"))
                st.session_state.models_loaded = True
            except FileNotFoundError as e:
                st.error(f"One or more model files could not be found. Please check the Models/Outputs directory.\n\nMissing: {e}")
                st.stop()
            except Exception as e:
                st.error(f"Failed to load models: {e}")
                st.stop()

        #loads eco models once
        if 'eco_models_loaded' not in st.session_state:
            try:
                st.session_state.eco_clf = joblib.load(os.path.join(MODEL_DIR, "eco_classification_pipe.pkl"))
                st.session_state.eco_reg = joblib.load(os.path.join(MODEL_DIR, "eco_regression_pipe.pkl"))
                st.session_state.eco_models_loaded = True
            except FileNotFoundError:
                st.warning("Eco models not found — eco segmentation will be skipped.")
                st.session_state.eco_clf = None
                st.session_state.eco_reg = None
                st.session_state.eco_models_loaded = True
            except Exception as e:
                st.warning(f"Eco models failed to load: {e}. Eco segmentation will be skipped.")
                st.session_state.eco_clf = None
                st.session_state.eco_reg = None
                st.session_state.eco_models_loaded = True

        eco_clf = st.session_state.eco_clf
        eco_reg = st.session_state.eco_reg

        kmeans = st.session_state.kmeans
        pt = st.session_state.pt
        medians = st.session_state.medians
        churn_model = st.session_state.churn_model

        run_segmentation = st.button("Run Segmentation Analysis")

        if run_segmentation or st.session_state.get('segmentation_done', False):

            #logs segmentation run event
            if run_segmentation:
                log_event(
                    action_type="run_analysis",
                    page_name="Customer Intelligence Platform",
                    details=f"Segmentation run initiated with {len(df)} records"
                )

            #creates engineered features and runs kmeans prediction
            try:
                df['Purchase_Intensity'] = df['Total_Orders_This_Month'] * df['Total_Amount_Spent'] / 1000
                df['Recency'] = df['Days_Since_Last_Order']
                df['Avg_Spend_Per_Order'] = df['Total_Amount_Spent'] / (df['Total_Orders_This_Month'] + 1)

                features = ['Purchase_Intensity', 'Recency', 'Avg_Spend_Per_Order']
                X = df[features].copy().fillna(medians)

                X_transformed = pt.transform(X)
                X_transformed[:, 0] *= 1.0
                X_transformed[:, 1] *= 1.5
                X_transformed[:, 2] *= 3.0

                df['KMeans_Segment'] = kmeans.predict(X_transformed)
            except Exception as e:
                st.error(
                    f"Segmentation failed during feature engineering or KMeans prediction.\n\n"
                    f"This usually means the uploaded CSV has incompatible data types or values.\n\nError: {e}"
                )
                st.stop()

            #assigns segment labels based on cluster statistics
            try:
                stats = df.groupby('KMeans_Segment').agg({
                    'Total_Amount_Spent': 'mean',
                    'Recency': 'mean',
                    'Total_Orders_This_Month': 'mean'
                })

                labels = {}
                available = list(stats.index)

                loyal_id = stats.loc[available, 'Total_Orders_This_Month'].idxmax()
                labels[loyal_id] = "Value Seeking Customers"
                available.remove(loyal_id)

                at_risk_id = stats.loc[available, 'Recency'].idxmax()
                labels[at_risk_id] = "At-Risk Customers"
                available.remove(at_risk_id)

                high_value_id = stats.loc[available, 'Total_Amount_Spent'].idxmax()
                labels[high_value_id] = "High-Value Customers"
                available.remove(high_value_id)

                if available:
                    labels[available[0]] = "Frequent Loyal Customers"

                df['Segment_Label'] = df['KMeans_Segment'].map(labels)
            except Exception as e:
                st.error(f"Segment labeling failed: {e}")
                st.stop()

            st.session_state['segmentation_done'] = True

            #runs eco prediction if models and columns are available
            if eco_clf is not None and eco_reg is not None:
                try:
                    eco_cat_cols = eco_clf.named_steps['prepro'].transformers_[0][2]
                    eco_num_cols = eco_clf.named_steps['prepro'].transformers_[1][2]
                    eco_required_cols = eco_cat_cols + eco_num_cols

                    missing_eco_cols = [c for c in eco_required_cols if c not in df.columns]

                    if len(missing_eco_cols) == 0:
                        X_eco = df[eco_required_cols].copy()

                        df['Pred_Green_Class'] = eco_clf.predict(X_eco)
                        df['Pred_Green_Score'] = eco_reg.predict(X_eco).clip(0, 1)

                        def eco_label(row):
                            if row['Pred_Green_Class'] == 1 and row['Pred_Green_Score'] >= 0.7:
                                return "Eco_Warrior"
                            elif row['Pred_Green_Class'] == 1 and 0.4 <= row['Pred_Green_Score'] < 0.7:
                                return "Eco_Conscious"
                            else:
                                return "Low_Interest"

                        df['Eco_Segment'] = df.apply(eco_label, axis=1)
                    else:
                        df['Eco_Segment'] = np.nan
                        st.warning(f"Eco prediction skipped. Missing columns: {missing_eco_cols}")
                except Exception as e:
                    df['Eco_Segment'] = np.nan
                    st.warning(f"Eco prediction failed: {e}")
            else:
                df['Eco_Segment'] = np.nan

            st.session_state["df_final"] = df.copy()
            st.session_state["segmentation_done"] = True

            #runs churn prediction if required columns are available
            try:
                required_cols = churn_model.named_steps['preprocessing'].transformers_[0][2] + \
                                churn_model.named_steps['preprocessing'].transformers_[1][2]
                available_cols = [c for c in required_cols if c in df.columns]
                st.session_state.available_cols = available_cols

                df_churn = df.copy()

                if len(available_cols) == len(required_cols):
                    X_churn = df_churn[available_cols].copy()
                    df_churn['Predicted_Churn'] = churn_model.predict(X_churn)
                else:
                    missing_churn = [c for c in required_cols if c not in df.columns]
                    df_churn['Predicted_Churn'] = np.nan
                    st.warning(f"Churn prediction skipped. Missing columns: {missing_churn}")
            except Exception as e:
                df_churn = df.copy()
                df_churn['Predicted_Churn'] = np.nan
                st.session_state.available_cols = []
                st.warning(f"Churn prediction failed: {e}")

            st.session_state.df = df
            st.session_state.df_churn = df_churn
            st.session_state.segmentation_done = True

        #runs cltv prediction
        cltv_model = st.session_state.cltv_model

        try:
            cltv_required_cols = (
                    cltv_model.named_steps['preprocessor'].transformers_[0][2] +
                    cltv_model.named_steps['preprocessor'].transformers_[1][2]
            )

            cltv_available_cols = [c for c in cltv_required_cols if c in df.columns]

            if len(cltv_available_cols) == len(cltv_required_cols):
                X_cltv = df[cltv_required_cols].copy()
                cltv_log_pred = cltv_model.predict(X_cltv)
                df['Predicted_CLTV'] = np.expm1(cltv_log_pred)
            else:
                missing_cltv = [c for c in cltv_required_cols if c not in df.columns]
                df['Predicted_CLTV'] = np.nan
                st.warning(f"CLTV prediction skipped. Missing columns: {missing_cltv}")

        except Exception as e:
            df['Predicted_CLTV'] = np.nan
            st.warning(f"CLTV prediction failed: {e}")

        #displays dashboard if segmentation is complete
        if st.session_state.get('segmentation_done', False):
            df = st.session_state.df
            df_churn = st.session_state.get('df_churn', df.copy())
            available_cols = st.session_state.get('available_cols', [])

            if 'Predicted_Churn' not in df_churn.columns:
                df_churn['Predicted_Churn'] = np.nan

            #displays financial and growth health kpis
            st.markdown("### Financial and Growth Health")
            row1_cols = st.columns(4)
            row1_data = [
                ("Avg Predicted CLTV", f"${df['Predicted_CLTV'].mean():,.0f}" if 'Predicted_CLTV' in df.columns and df['Predicted_CLTV'].notna().any() else "N/A", "Model Output", "border-purple-600"),
                ("Total Revenue", f"${df['Total_Amount_Spent'].sum():,.0f}", "All customers", "border-purple-600"),
                ("Avg Orders / Month", f"{df['Total_Orders_This_Month'].mean():.2f}", "Calculated", "border-purple-500"),
                ("Avg Recency (Days)", f"{df['Days_Since_Last_Order'].mean():.1f}", "Calculated", "border-purple-400"),
            ]
            for col, (title, value, sub, border) in zip(row1_cols, row1_data):
                col.markdown(f"""
                    <div class="card {border}">
                        <p class="kpi-title">{title}</p>
                        <p class="kpi-value">{value}</p>
                        <p class="kpi-sub">{sub}</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            #displays engagement and risk health kpis
            st.markdown("### Engagement and Risk Health")
            row2_cols = st.columns(4)
            row2_data = [
                ("Avg Satisfaction Score", f"{df['Satisfaction_Score'].mean():.2f}/5", "Calculated", "border-purple-600"),
                ("Avg Orders / Customer", f"{df['Total_Orders_This_Month'].mean():.2f}", "Calculated", "border-purple-500"),
                ("Active Customer Segments", df['Segment_Label'].nunique(), "4 segments", "border-purple-400"),
                ("Total Customers", len(df), "Count", "border-purple-300"),
            ]
            for col, (title, value, sub, border) in zip(row2_cols, row2_data):
                col.markdown(f"""
                    <div class="card {border}">
                        <p class="kpi-title">{title}</p>
                        <p class="kpi-value">{value}</p>
                        <p class="kpi-sub">{sub}</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)

            #displays segmentation charts and customer segment cards
            st.title("Segmentation Intelligence")
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1], gap="large")

            with col1:
                st.markdown("### Segment Distribution")
                segments = df['Segment_Label'].unique()
                colors = ['#6A0DAD', '#8A2BE2', '#c9a0dc', '#e8d5f2']
                color_map = dict(zip(segments, colors))
                seg_counts = df['Segment_Label'].value_counts().reset_index()
                fig = go.Figure(data=[go.Pie(
                    labels=seg_counts['Segment_Label'],
                    values=seg_counts['count'],
                    hole=0.65,
                    marker=dict(colors=[color_map.get(x, '#cccccc') for x in seg_counts['Segment_Label']])
                )])
                fig.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("<br><br>", unsafe_allow_html=True)

                st.markdown("### Average Spend per Segment")
                avg_spend = df.groupby('Segment_Label')['Total_Amount_Spent'].mean().reset_index()
                spend_chart = px.bar(
                    avg_spend,
                    x='Segment_Label',
                    y='Total_Amount_Spent',
                    color='Segment_Label',
                    color_discrete_map=color_map
                )
                spend_chart.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                                          xaxis_title="", yaxis_title="Average Spend ($)",
                                          margin=dict(t=20, b=20, l=0, r=0))
                st.plotly_chart(spend_chart, use_container_width=True)

                #renders customer segment detail cards
                with col2:
                    churn_per_segment = df_churn.groupby('Segment_Label')['Predicted_Churn'].mean().to_dict()
                    eco_dist = calculate_eco_strength(df)

                    segments_list = []
                    colors_copy = ['#6A0DAD', '#8A2BE2', '#c9a0dc', '#e8d5f2']
                    for label, row in df.groupby('Segment_Label'):
                        churn_rate = churn_per_segment.get(label, np.nan)
                        eco_strength_pct = eco_dist.loc[label, 'Eco_Strength_Pct'] if label in eco_dist.index else 0

                        segments_list.append({
                            'name': label,
                            'base': f"{len(row) / len(df) * 100:.0f}%",
                            'avg_spend': f"${row['Total_Amount_Spent'].mean():,.0f}",
                            'avg_orders': f"{row['Total_Orders_This_Month'].mean():.2f}",
                            'recency': f"{row['Days_Since_Last_Order'].mean():.1f} days",
                            'satisfaction': f"{row['Satisfaction_Score'].mean():.2f}/5",
                            'eco_strength': f"{eco_strength_pct:.0f}%",
                            'churn_risk': f"{churn_rate * 100:.0f}%" if not np.isnan(churn_rate) else "N/A",
                            'color': colors_copy.pop(0) if colors_copy else '#8A2BE2'
                        })

                    for seg in segments_list:
                        st.markdown(f"#### {seg['name']} ({seg['base']} of base)")
                        kpi_cols = st.columns(3)
                        kpi_cols[0].metric("Avg Spend", seg['avg_spend'])
                        kpi_cols[1].metric("Avg Orders", seg['avg_orders'])
                        kpi_cols[2].metric("Recency", seg['recency'])

                        kpi_cols2 = st.columns(3)
                        kpi_cols2[0].metric("Satisfaction", seg['satisfaction'])
                        kpi_cols2[1].metric("Eco Strength", seg['eco_strength'])
                        kpi_cols2[2].markdown(
                            f"<div style='background:{seg['color']}33; padding:6px; border-radius:6px; text-align:center;'>"
                            f"<span style='color:{seg['color']}; font-weight:600'>Churn Risk: {seg['churn_risk']}</span></div>",
                            unsafe_allow_html=True
                        )
                        st.markdown("---")

            #displays churn analysis section
            st.markdown("<br>", unsafe_allow_html=True)
            st.title("Churn Analysis")

            if df_churn['Predicted_Churn'].notna().any():
                churn_col1, churn_col2 = st.columns([1, 1], gap="large")

                with churn_col1:
                    st.markdown("### Churn Distribution (0 = No Churn, 1 = Churn)")

                    churn_counts = df_churn['Predicted_Churn'].value_counts().sort_index()

                    fig_churn = go.Figure(data=[go.Bar(
                        x=['No', 'Yes'],
                        y=churn_counts.values,
                        marker_color=['#6A0DAD', '#c9a0dc'],
                        text=churn_counts.values,
                        textposition='auto',
                    )])

                    fig_churn.update_layout(
                        xaxis_title="Churn Status",
                        yaxis_title="Number of Customers",
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=20, b=20, l=0, r=0),
                        height=400
                    )

                    st.plotly_chart(fig_churn, use_container_width=True)

                #displays churn shap analysis
                with churn_col2:
                    st.markdown('<div style="margin-top: -75px;"><h2>Top Churn Drivers (SHAP)</h2></div>',
                                unsafe_allow_html=True)

                    try:
                        model = churn_model.named_steps['model']
                        preprocessor = churn_model.named_steps['preprocessing']

                        numeric_cols = preprocessor.transformers_[1][2]
                        cat_cols = preprocessor.transformers_[0][2]

                        cat_feature_names = preprocessor.named_transformers_['cat'] \
                            .get_feature_names_out(cat_cols).tolist()

                        all_feature_names = cat_feature_names + numeric_cols

                        X_shap = preprocessor.transform(df_churn[available_cols])
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_shap)

                        if isinstance(shap_values, list):
                            shap_impact = np.abs(shap_values[1]).mean(axis=0)
                        else:
                            shap_impact = np.abs(shap_values).mean(axis=0)

                        feature_impacts = sorted(
                            zip(all_feature_names, shap_impact),
                            key=lambda x: x[1],
                            reverse=True
                        )[:8]

                        feat_names = [f[0] for f in feature_impacts]
                        impact_vals = [f[1] for f in feature_impacts]

                        fig_shap_churn = px.bar(
                            x=impact_vals,
                            y=feat_names,
                            orientation='h',
                            text=[f"{v:.3f}" for v in impact_vals]
                        )

                        fig_shap_churn.update_traces(
                            marker_color='#EBEBEB',
                            textposition='auto',
                            textfont_color='#8A2BE2',
                            insidetextanchor='end'
                        )

                        fig_shap_churn.update_layout(
                            xaxis_title="Average Impact (SHAP Value)",
                            yaxis_title="",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=0, b=0, l=0, r=10),
                            height=400,
                            yaxis={'categoryorder': 'total ascending'}
                        )

                        fig_shap_churn.update_layout(uniformtext_minsize=11, uniformtext_mode='show')

                        st.plotly_chart(fig_shap_churn, use_container_width=True)

                    except Exception as e:
                        st.error(f"SHAP analysis could not be generated: {e}")
            else:
                st.warning("Churn predictions are not available for this dataset. Required columns may be missing.")

            #displays cltv analysis section
            st.markdown("<br>", unsafe_allow_html=True)
            st.title("Customer Lifetime Value Analysis")

            if 'Predicted_CLTV' in df.columns and df['Predicted_CLTV'].notna().sum() > 0:
                cltv_col1, cltv_col2 = st.columns([1, 1], gap="large")

                with cltv_col1:
                    st.markdown("### Predicted CLTV Distribution")

                    fig_cltv = px.histogram(
                        df,
                        x='Predicted_CLTV',
                        nbins=40,
                        color_discrete_sequence=['#8A2BE2']
                    )

                    fig_cltv.update_traces(
                        marker=dict(
                            color=df['Predicted_CLTV'],
                            colorscale=['#e8d5f2', '#8A2BE2', '#6A0DAD'],
                            showscale=False
                        )
                    )

                    fig_cltv.update_layout(
                        xaxis_title="Predicted CLTV ($)",
                        yaxis_title="Number of Customers",
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(t=20, b=20, l=0, r=0),
                        height=400,
                        bargap=0.05
                    )

                    st.plotly_chart(fig_cltv, use_container_width=True)

                #displays cltv shap analysis
                with cltv_col2:
                    st.markdown('<div style="margin-top: -75px;"><h2>Top CLTV Drivers (SHAP)</h2></div>',
                                unsafe_allow_html=True)

                    try:
                        cltv_model = st.session_state.cltv_model
                        preprocessor = cltv_model.named_steps['preprocessor']
                        model = cltv_model.named_steps['regressor']

                        cat_cols = preprocessor.transformers_[0][2]
                        num_cols = preprocessor.transformers_[1][2]

                        cat_feature_names = preprocessor.named_transformers_['cat'] \
                            .get_feature_names_out(cat_cols).tolist()

                        all_feature_names = cat_feature_names + num_cols

                        X_shap = preprocessor.transform(df[cat_cols + num_cols])
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(X_shap)

                        shap_impact = np.abs(shap_values).mean(axis=0)

                        feature_impacts = sorted(
                            zip(all_feature_names, shap_impact),
                            key=lambda x: x[1],
                            reverse=True
                        )[:8]

                        feat_names = [f[0] for f in feature_impacts]
                        impact_vals = [f[1] for f in feature_impacts]

                        fig_shap = px.bar(
                            x=impact_vals,
                            y=feat_names,
                            orientation='h',
                            text=[f"{v:.3f}" for v in impact_vals]
                        )

                        fig_shap.update_traces(
                            marker_color='#EBEBEB',
                            textposition='auto',
                            textfont_color='#BA56E9',
                            insidetextanchor='end'
                        )

                        fig_shap.update_layout(
                            xaxis_title="Average Impact (SHAP Value)",
                            yaxis_title="",
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            margin=dict(t=0, b=0, l=0, r=10),
                            height=400,
                            yaxis={'categoryorder': 'total ascending'}
                        )

                        fig_shap.update_layout(uniformtext_minsize=11, uniformtext_mode='show')

                        st.plotly_chart(fig_shap, use_container_width=True)

                    except Exception as e:
                        st.error(f"CLTV SHAP could not be generated: {e}")
            else:
                st.warning("CLTV predictions are not available for this dataset. Required columns may be missing.")

            #displays customer drill-through table
            st.title("Customer Drill-Through")
            df_final = df_churn.copy()

            if 'Predicted_CLTV' in df.columns:
                df_final['Predicted_CLTV'] = df['Predicted_CLTV']

            st.dataframe(df_final, use_container_width=True, height=400)

        #user comments and annotations section
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Add Your Comments or Insights")

        if "user_comments" not in st.session_state:
            st.session_state.user_comments = []

        col1, col2 = st.columns([4, 1], vertical_alignment="bottom")

        with col1:
            comment_input = st.text_area(
                "Enter your comment/annotation here:",
                key="comment_input",
                height=80,
                label_visibility="collapsed"
            )

        with col2:
            if st.button("Save Comment", use_container_width=True):
                if comment_input.strip() != "":
                    st.session_state.user_comments.append(comment_input.strip())
                    st.success("Saved!")
                    st.rerun()

        #exports dashboard report with comments
        if "df_final" in st.session_state:
            export_df = st.session_state["df_final"].copy()
        else:
            st.info("Run segmentation first to enable export.")
            st.stop()

        export_df['Comment'] = ""

        if st.session_state.user_comments:
            comments_df = pd.DataFrame({
                "Customer_ID": [""] * len(st.session_state.user_comments),
                "Comment": st.session_state.user_comments
            })
            export_df = pd.concat([export_df, comments_df], ignore_index=True)

        csv_data = export_df.to_csv(index=False).encode('utf-8')

        #logs download event
        def log_download():
            log_event(
                action_type="download",
                page_name="Customer Intelligence Platform",
                file_name="dashboard_report.csv"
            )

        st.download_button(
            label="Download Dashboard Report with Comments",
            data=csv_data,
            file_name="dashboard_report.csv",
            mime="text/csv",
            on_click=log_download,
            use_container_width=True
        )