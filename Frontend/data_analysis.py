import streamlit as st
import pandas as pd
import hashlib
from Managers.DataIngestionManager import DataIngestionManager
from styles import inject_styles
from audit_logger import log_event


def show():
    inject_styles()

    st.title("Data Management and Integrity")
    st.caption("Data Cleaning, Validation, and Structural Integrity Checks")
    st.markdown("<br>", unsafe_allow_html=True)

    #data upload section
    st.markdown("### Data Upload")
    uploaded_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "json"],
        key="data_upload"
    )

    #resets session if no file uploaded
    if uploaded_file is None:
        st.session_state.pop("cleaned_df", None)
        st.session_state.pop("audit", None)
        st.session_state.pop("data_upload_hash", None)
        return

    #computes file hash to avoid duplicate logging
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if "data_upload_hash" not in st.session_state:
        st.session_state.data_upload_hash = None

    if st.session_state.data_upload_hash != file_hash:
        log_event(
            action_type="upload",
            page_name="Data Management & Integrity",
            file_name=uploaded_file.name
        )
        st.session_state.data_upload_hash = file_hash

    #reads uploaded file based on extension
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_json(uploaded_file)

    st.success(f"{len(df):,} records loaded")

    #runs audit and preprocessing
    manager = DataIngestionManager(df)
    manager.smart_quality_audit()
    df_cleaned = manager.auto_preprocess()
    audit_report = manager.generate_audit_report()

    st.session_state["cleaned_df"] = df_cleaned
    st.session_state["audit"] = audit_report

    #displays audit results
    if "audit" in st.session_state:
        audit = st.session_state["audit"]
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("## Format Integrity Check")
        if audit["format_issues"]:
            for col, msg in audit["format_issues"].items():
                st.warning(f"{col}: {msg}")
        else:
            st.success("No format issues detected")
        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Missing Values Report")
            for col, percent in audit["missing_summary"].items():
                st.markdown(f"**{col}** — {100 - percent:.2f}% Complete")
                st.progress((100 - percent) / 100)

        with col2:
            st.markdown("### Outlier Detection")
            if audit["outliers"]:
                for col, count in audit["outliers"].items():
                    st.warning(f"{count} outliers detected in {col}")
            else:
                st.success("No significant outliers detected")

    #displays cleaned data preview and download
    if "cleaned_df" in st.session_state:
        st.markdown("## Cleaned Dataset Preview")
        st.dataframe(st.session_state["cleaned_df"], use_container_width=True)

        csv_data = st.session_state["cleaned_df"].to_csv(index=False).encode("utf-8")

        #logs download event
        def log_download():
            log_event(
                action_type="download",
                page_name="Data Management & Integrity",
                file_name="cleaned_dataset.csv"
            )

        st.download_button(
            label="Download Cleaned Dataset",
            data=csv_data,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
            use_container_width=True,
            on_click=log_download
        )

        #displays processing report
        st.markdown("## Processing Report")
        audit = st.session_state["audit"]

        for action in audit["actions_taken"]:
            st.info(action)

        for note in audit["audit_notes"]:
            st.warning(note)