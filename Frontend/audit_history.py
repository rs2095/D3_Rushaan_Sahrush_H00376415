import streamlit as st
from firebase_init import db
import pandas as pd


def show():

    if st.session_state["user"]["role"].lower() != "admin":
        st.error("Access Denied. Admins Only.")
        st.stop()

    st.title("Audit Logs")
    st.markdown("Real-time tracking of uploads, downloads, and analytical executions.")

    #pulls logs from firebase filtered by organization
    org_id = st.session_state["user"].get("organization_id", "")
    logs = db.collection("audit_logs").where("organization_id", "==", org_id).stream()

    log_list = []
    for log in logs:
        log_list.append(log.to_dict())

    if not log_list:
        st.info("No audit records found for your organization.")
        return

    df_logs = pd.DataFrame(log_list)

    df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
    df_logs = df_logs.sort_values(by="timestamp", ascending=False)

    #displays audit overview metrics
    st.markdown("### Audit Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Events", len(df_logs))
    col2.metric("Uploads", len(df_logs[df_logs["action_type"] == "upload"]))
    col3.metric("Downloads", len(df_logs[df_logs["action_type"] == "download"]))
    col4.metric("Analyses Run", len(df_logs[df_logs["action_type"] == "run_analysis"]))

    st.markdown("---")

    #filter controls for user, action type, and page
    st.markdown("### Filters")

    col1, col2, col3 = st.columns(3)

    with col1:
        user_filter = st.selectbox(
            "User",
            ["All"] + sorted(df_logs["user_email"].dropna().unique().tolist())
        )

    with col2:
        action_filter = st.selectbox(
            "Action Type",
            ["All"] + sorted(df_logs["action_type"].dropna().unique().tolist())
        )

    with col3:
        page_filter = st.selectbox(
            "Page",
            ["All"] + sorted(df_logs["page_name"].dropna().unique().tolist())
        )

    #applies selected filters to log data
    filtered_df = df_logs.copy()

    if user_filter != "All":
        filtered_df = filtered_df[filtered_df["user_email"] == user_filter]

    if action_filter != "All":
        filtered_df = filtered_df[filtered_df["action_type"] == action_filter]

    if page_filter != "All":
        filtered_df = filtered_df[filtered_df["page_name"] == page_filter]

    #displays filtered audit records
    st.markdown("### Audit Records")

    st.dataframe(
        filtered_df[
            [
                "timestamp",
                "user_email",
                "role",
                "action_type",
                "page_name",
                "file_name",
                "details"
            ]
        ],
        use_container_width=True,
        height=500
    )

    #exports filtered logs as csv
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Export Filtered Logs",
        data=csv,
        file_name="audit_logs_export.csv",
        mime="text/csv"
    )