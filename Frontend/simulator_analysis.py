import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from Managers.SimulatorManager import SimulatorManager
from styles import inject_styles
from audit_logger import log_event

REQUIRED_FEATURES = [
    'Preferred_Login_Device', 'City_Tier', 'Preferred_Payment_Mode',
    'Hours_Spend_On_Website', 'Total_Devices_Registered', 'Satisfaction_Score',
    'Marital_Status', 'Complain_Count_In_Last_Month', 'Order_Amount_Hike_From_Last_Year',
    'Days_Since_Last_Order', 'Average_Cash_Back',
    'Customer_Gender', 'Customer_Residence_Type', 'Customer_Qualification',
    'Customer_Income', 'Total_Subscriptions', 'Membership_Type',
    'Loyalty_Engagement_Score', 'Customer_Age', 'Total_Amount_Spent',
    'Total_Returns', 'Total_Orders_This_Month'
]

BASELINE_COLUMNS = ['Predicted_Churn', 'Predicted_CLTV']


def _reset_sliders():
    st.session_state["sl_cashback"] = 0
    st.session_state["sl_marketing"] = 0
    st.session_state["sl_support"] = 0
    st.session_state["sl_subscription"] = 0


#generates rule-based business insight bullets from simulation results
def _generate_insights(df, sim_churn_arr, sim_cltv_arr, is_baseline,
                       cashback_val, marketing_val, support_val, subscription_val):

    if is_baseline:
        return [("neutral", "All scenario levers are at their default position. Adjust the sliders above to simulate business strategies and view projected impacts.")]

    insights = []
    total_customers = len(df)

    base_churn_count = int(df['Predicted_Churn'].sum())
    sim_churn_count = int(sim_churn_arr.sum())
    churn_diff = sim_churn_count - base_churn_count

    base_cltv_total = float(df['Predicted_CLTV'].sum())
    sim_cltv_total = float(sim_cltv_arr.sum())
    revenue_diff = sim_cltv_total - base_cltv_total

    base_churn_rate = float(df['Predicted_Churn'].mean()) * 100
    sim_churn_rate = float(sim_churn_arr.mean()) * 100
    churn_rate_diff = sim_churn_rate - base_churn_rate

    base_avg_cltv = float(df['Predicted_CLTV'].mean())
    sim_avg_cltv = float(sim_cltv_arr.mean())
    avg_cltv_diff = sim_avg_cltv - base_avg_cltv

    improved_count = int((sim_cltv_arr > df['Predicted_CLTV'].values).sum())
    worsened_count = int((sim_cltv_arr < df['Predicted_CLTV'].values).sum())

    #churn impact insights
    if churn_diff < 0:
        insights.append((
            "positive",
            f"Churn is projected to decrease by {abs(churn_diff)} customers "
            f"(from {base_churn_count} to {sim_churn_count}), reducing the churn rate "
            f"by {abs(churn_rate_diff):.2f}%. This suggests the applied strategies are "
            f"effective at retaining at-risk customers."
        ))
    elif churn_diff > 0:
        insights.append((
            "negative",
            f"Churn is projected to increase by {churn_diff} customers "
            f"(from {base_churn_count} to {sim_churn_count}), raising the churn rate "
            f"by {churn_rate_diff:.2f}%. This indicates the current scenario may be "
            f"negatively impacting customer retention and should be reconsidered."
        ))
    else:
        insights.append((
            "neutral",
            "The projected churn count remains unchanged from baseline, "
            "indicating the applied adjustments have a neutral effect on customer retention."
        ))

    #revenue impact insights
    if revenue_diff > 0:
        insights.append((
            "positive",
            f"Total projected revenue increased by ${revenue_diff:,.0f}, "
            f"with {improved_count} out of {total_customers} customers showing improved CLV. "
            f"This represents a {(revenue_diff / base_cltv_total * 100):.2f}% uplift in total revenue."
        ))
    elif revenue_diff < 0:
        insights.append((
            "negative",
            f"Total projected revenue decreased by ${abs(revenue_diff):,.0f}, "
            f"with {worsened_count} out of {total_customers} customers showing reduced CLV. "
            f"This represents a {(abs(revenue_diff) / base_cltv_total * 100):.2f}% decline in total revenue."
        ))
    else:
        insights.append((
            "neutral",
            "Total projected revenue remains unchanged from baseline."
        ))

    #average clv change insight
    if abs(avg_cltv_diff) > 0.01:
        direction = "increased" if avg_cltv_diff > 0 else "decreased"
        severity = "positive" if avg_cltv_diff > 0 else "negative"
        insights.append((
            severity,
            f"Average customer lifetime value {direction} by ${abs(avg_cltv_diff):,.2f} per customer "
            f"(from ${base_avg_cltv:,.2f} to ${sim_avg_cltv:,.2f})."
        ))

    #cashback slider insights
    if cashback_val > 0 and churn_diff < 0:
        insights.append((
            "note",
            f"Increasing cashback by {cashback_val}% contributed to lower churn, consistent with "
            f"SHAP analysis identifying Average_Cash_Back as the strongest churn-reducing factor."
        ))
    elif cashback_val < 0:
        insights.append((
            "note",
            f"Reducing cashback by {abs(cashback_val)}% may weaken a key retention lever. "
            f"Consider balancing cost savings against potential churn increases."
        ))

    #support quality slider insights
    if support_val > 0 and churn_diff <= 0:
        insights.append((
            "note",
            f"Improving customer support quality by {support_val}% is projected to reduce complaints, "
            f"which SHAP analysis identified as the second strongest driver of churn."
        ))

    #subscription push slider insights
    if subscription_val > 0:
        subs_str = df['Total_Subscriptions'].astype(str).str.strip()
        original_single = int((subs_str == '1').sum())
        converted = min(int(original_single * (subscription_val / 100)), original_single)
        if converted > 0:
            insights.append((
                "note",
                f"The subscription push converted approximately {converted} single-subscription "
                f"customers to multi-subscription status. SHAP analysis identified subscription type "
                f"as the dominant driver of CLV, so this conversion directly targets the highest-impact lever."
            ))

    #marketing slider insights
    if marketing_val > 0 and revenue_diff > 0:
        insights.append((
            "note",
            f"Increasing marketing incentives by {marketing_val}% is projected to boost spending "
            f"and reduce customer inactivity, contributing to the overall revenue improvement."
        ))
    elif marketing_val < 0:
        insights.append((
            "note",
            f"Reducing marketing incentives by {abs(marketing_val)}% may increase customer inactivity "
            f"and reduce order frequency. Monitor Days_Since_Last_Order for early warning signs."
        ))

    #overall scenario recommendation
    if revenue_diff > 0 and churn_diff <= 0:
        insights.append((
            "positive",
            "Overall, the simulated scenario projects both improved revenue and reduced or stable churn. "
            "This combination suggests a viable strategy worth considering for implementation."
        ))
    elif revenue_diff < 0 and churn_diff > 0:
        insights.append((
            "negative",
            "Overall, the simulated scenario projects both reduced revenue and increased churn. "
            "This combination suggests the strategy may be counterproductive and should be revised."
        ))
    elif revenue_diff > 0 and churn_diff > 0:
        insights.append((
            "mixed",
            "The scenario projects improved revenue but increased churn. While short-term revenue "
            "gains are positive, the rising churn rate could erode long-term value. Consider adjusting "
            "the strategy to balance both outcomes."
        ))
    elif revenue_diff < 0 and churn_diff < 0:
        insights.append((
            "mixed",
            "The scenario projects reduced revenue but lower churn. While customer retention is improving, "
            "the revenue decline suggests the cost of retention strategies may outweigh immediate gains. "
            "Consider optimising the balance between retention investment and revenue impact."
        ))

    return insights


#renders a single insight card with colored border
def _render_insight(severity, text):
    border_colors = {
        "positive": "#2ecc71",
        "negative": "#e74c3c",
        "mixed": "#f39c12",
        "neutral": "#3498db",
        "note": "#8e44ad",
    }
    border_color = border_colors.get(severity, "#3498db")

    st.markdown(f"""
        <div class="sim-card" style="
            border-left: 4px solid {border_color};
            text-align: left;
        ">
            <div style="font-size: 14px; line-height: 1.6;">
                {text}
            </div>
        </div>
    """, unsafe_allow_html=True)


def show():
    if "user" not in st.session_state:
        st.warning("You must log in first to view this page.")
        st.stop()

    user = st.session_state["user"]
    role = user.get("role", "user")

    if role != "admin":
        st.warning("Access Denied: Only admins can use the simulator.")
        st.stop()

    inject_styles()
    st.title("Strategy and Growth Simulator")

    uploaded_file = st.file_uploader(
        "Upload CSV with baseline predictions",
        type=["csv"],
        key="sim_file_uploader"
    )
    if uploaded_file is None:
        st.info("Upload a dataset containing 'Predicted_Churn' and 'Predicted_CLTV'.")
        return

    #computes file hash and logs upload event
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    if st.session_state.get("upload_hash") != file_hash:
        log_event(
            action_type="upload",
            page_name="Strategy & Growth Simulator",
            file_name=uploaded_file.name
        )
        st.session_state["upload_hash"] = file_hash
        st.session_state.pop("sim_base_df", None)

    #loads and validates uploaded dataset
    if "sim_base_df" not in st.session_state:
        df = pd.read_csv(uploaded_file)

        missing_features = [c for c in REQUIRED_FEATURES if c not in df.columns]
        missing_baseline = [c for c in BASELINE_COLUMNS if c not in df.columns]

        if missing_features:
            st.error(f"Dataset missing required features: {', '.join(missing_features)}")
            return
        if missing_baseline:
            st.error(f"Dataset missing baseline columns: {', '.join(missing_baseline)}")
            return

        st.session_state["sim_base_df"] = df.copy()

    df = st.session_state["sim_base_df"]
    st.success(f"{len(df)} records loaded.")

    with st.expander("Preview Uploaded Data"):
        st.dataframe(df.head(), use_container_width=True)

    #business scenario slider controls
    st.divider()
    st.subheader("Business Scenario Levers")

    st.button("Reset Sliders", key="reset_sim", on_click=_reset_sliders)

    c1, c2 = st.columns(2)
    with c1:
        cashback_val = st.slider(
            "Increase Cashback Opportunities (%)",
            min_value=-20, max_value=50, value=0, step=1,
            key="sl_cashback"
        )
    with c2:
        marketing_val = st.slider(
            "More Marketing & Customer Incentives (%)",
            min_value=-10, max_value=30, value=0, step=1,
            key="sl_marketing"
        )

    c3, c4 = st.columns(2)
    with c3:
        support_val = st.slider(
            "Boost Customer Support Quality (%)",
            min_value=0, max_value=50, value=0, step=1,
            key="sl_support"
        )
    with c4:
        subscription_val = st.slider(
            "Subscription Push (%)",
            min_value=0, max_value=50, value=0, step=1,
            key="sl_subscription"
        )

    #runs simulation with current slider values
    sim_manager = SimulatorManager(df)
    sim_df_result, sim_churn_probs, sim_cltv_dollars = sim_manager.simulate(
        cashback=cashback_val,
        complaints_red=support_val,
        price_hike=marketing_val,
        subscription_push=subscription_val
    )

    is_baseline = sim_df_result is None

    if is_baseline:
        sim_churn_arr = df['Predicted_Churn'].values.astype(float)
        sim_cltv_arr = df['Predicted_CLTV'].values.astype(float)
        display_df = df.copy()
        display_df['Simulated_Churn'] = df['Predicted_Churn']
        display_df['Simulated_CLTV'] = df['Predicted_CLTV']
    else:
        sim_churn_arr = (sim_churn_probs > 0.5).astype(float)
        sim_cltv_arr = sim_cltv_dollars.astype(float)
        display_df = sim_df_result.copy()
        display_df['Simulated_Churn'] = sim_churn_arr
        display_df['Simulated_CLTV'] = sim_cltv_arr

    st.session_state["simulated_df"] = display_df.copy()

    #calculates baseline and simulated kpi values
    base_cltv = float(df['Predicted_CLTV'].mean())
    base_churn = float(df['Predicted_Churn'].mean()) * 100
    base_revenue = float(df['Predicted_CLTV'].sum())

    if is_baseline:
        sim_cltv_mean = base_cltv
        sim_churn_pct = base_churn
        sim_revenue = base_revenue
        improved_pct = 0.0
        churned_pct = base_churn
    else:
        sim_cltv_mean = float(sim_cltv_arr.mean())
        sim_churn_pct = float(sim_churn_arr.mean()) * 100
        sim_revenue = float(sim_cltv_arr.sum())
        improved_pct = float((sim_cltv_arr > df['Predicted_CLTV'].values).mean()) * 100
        churned_pct = float(sim_churn_arr.mean()) * 100

    avg_cltv_change = sim_cltv_mean - base_cltv

    #displays simulation result kpi cards
    st.divider()
    st.subheader("Simulation Results")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        _money_card("Recalculated CLTV", sim_cltv_mean, base_cltv)
    with r1c2:
        _churn_card("Projected Churn Rate", sim_churn_pct, base_churn)
    with r1c3:
        _money_card("Net Revenue Impact", sim_revenue, base_revenue)

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        _pct_card("% Customers Improving CLTV", improved_pct)
    with r2c2:
        _pct_card("% Customers Churned", churned_pct)
    with r2c3:
        _val_card("Avg CLTV Change Per Customer", f"${avg_cltv_change:,.2f}")

    #displays business insights
    st.divider()
    st.subheader("Business Insights")

    insights = _generate_insights(
        df, sim_churn_arr, sim_cltv_arr, is_baseline,
        cashback_val, marketing_val, support_val, subscription_val
    )

    for severity, text in insights:
        _render_insight(severity, text)

    #displays active scenario summary table
    st.divider()
    st.subheader("Active Scenario Summary")

    st.dataframe(pd.DataFrame({
        "Business Action": [
            "Increase Cashback Opportunities",
            "More Marketing & Customer Incentives",
            "Boost Customer Support Quality",
            "Subscription Push"
        ],
        "Adjustment (%)": [
            cashback_val, marketing_val, support_val, subscription_val
        ],
        "Primary Feature": [
            "Average_Cash_Back",
            "Total_Amount_Spent, Order_Amount_Hike_From_Last_Year",
            "Complain_Count_In_Last_Month",
            "Total_Subscriptions"
        ],
        "Correlated Features": [
            "Satisfaction_Score, Loyalty_Engagement_Score",
            "Days_Since_Last_Order",
            "Satisfaction_Score",
            "Loyalty_Engagement_Score"
        ]
    }), use_container_width=True, hide_index=True)

    #exports simulation results and business actions
    st.divider()
    st.subheader("Export Simulation")

    export_df = display_df.copy()
    export_df['Baseline_Churn'] = df['Predicted_Churn']
    export_df['Baseline_CLTV'] = df['Predicted_CLTV']

    st.download_button(
        label="Download Simulated Dataset",
        data=export_df.to_csv(index=False).encode('utf-8'),
        file_name="simulated_dataset.csv",
        mime="text/csv",
        use_container_width=True,
        on_click=lambda: log_event(
            action_type="download",
            page_name="Strategy & Growth Simulator",
            file_name="simulated_dataset.csv"
        ),
        key="dl_sim"
    )

    actions_df = pd.DataFrame([
        {"Action": "Increase Cashback Opportunities", "Slider Value": cashback_val,
         "Primary Feature": "Average_Cash_Back",
         "Correlated Features": "Satisfaction_Score, Loyalty_Engagement_Score"},
        {"Action": "More Marketing & Customer Incentives", "Slider Value": marketing_val,
         "Primary Feature": "Total_Amount_Spent",
         "Correlated Features": "Days_Since_Last_Order, Order_Amount_Hike_From_Last_Year"},
        {"Action": "Boost Customer Support Quality", "Slider Value": support_val,
         "Primary Feature": "Complain_Count_In_Last_Month",
         "Correlated Features": "Satisfaction_Score"},
        {"Action": "Subscription Push", "Slider Value": subscription_val,
         "Primary Feature": "Total_Subscriptions",
         "Correlated Features": "Loyalty_Engagement_Score"},
    ])

    st.download_button(
        label="Download Business Actions Log",
        data=actions_df.to_csv(index=False).encode('utf-8'),
        file_name="business_actions_log.csv",
        mime="text/csv",
        use_container_width=True,
        on_click=lambda: log_event(
            action_type="download",
            page_name="Strategy & Growth Simulator",
            file_name="business_actions_log.csv"
        ),
        key="dl_actions"
    )


#renders churn rate kpi card with baseline comparison
def _churn_card(title, value, base):
    d = value - base
    if abs(d) < 0.001:
        sub = '<span style="color:#888;">No change from baseline</span>'
    else:
        arrow = "▲" if d > 0 else "▼"
        color = "#e74c3c" if d > 0 else "#2ecc71"
        sub = f'<span style="color:{color};">{arrow} {abs(d):.2f}% from baseline</span>'
    st.markdown(f"""
        <div class="sim-card">
            <div class="sim-title">{title}</div>
            <div class="sim-value">{value:.2f}%</div>
            <div class="sim-sub">{sub}</div>
        </div>
    """, unsafe_allow_html=True)


#renders monetary kpi card with baseline comparison
def _money_card(title, value, base):
    d = ((value - base) / base * 100) if base != 0 else 0
    if abs(d) < 0.001:
        sub = '<span style="color:#888;">No change from baseline</span>'
    else:
        arrow = "▲" if d > 0 else "▼"
        color = "#2ecc71" if d > 0 else "#e74c3c"
        sub = f'<span style="color:{color};">{arrow} {abs(d):.2f}% vs baseline</span>'
    st.markdown(f"""
        <div class="sim-card">
            <div class="sim-title">{title}</div>
            <div class="sim-value">${value:,.2f}</div>
            <div class="sim-sub">{sub}</div>
        </div>
    """, unsafe_allow_html=True)


#renders percentage kpi card
def _pct_card(title, value):
    st.markdown(f"""
        <div class="sim-card">
            <div class="sim-title">{title}</div>
            <div class="sim-value">{value:.2f}%</div>
        </div>
    """, unsafe_allow_html=True)


#renders generic value kpi card
def _val_card(title, value_str):
    st.markdown(f"""
        <div class="sim-card">
            <div class="sim-title">{title}</div>
            <div class="sim-value">{value_str}</div>
        </div>
    """, unsafe_allow_html=True)