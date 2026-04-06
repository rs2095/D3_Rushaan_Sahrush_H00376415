import streamlit as st
from firebase_init import db
from datetime import datetime


#logs user action to firebase audit collection scoped by organization
def log_event(action_type, page_name, file_name=None, details=None):
    user = st.session_state.get("user", {})

    log_entry = {
        "user_email": user.get("email", "Unknown"),
        "role": user.get("role", "Unknown"),
        "organization_id": user.get("organization_id", ""),
        "action_type": action_type,
        "page_name": page_name,
        "file_name": file_name,
        "details": details,
        "timestamp": datetime.utcnow().isoformat()
    }

    try:
        db.collection("audit_logs").add(log_entry)
    except Exception as e:
        print("Audit log failed:", e)