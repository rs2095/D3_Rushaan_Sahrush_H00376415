import streamlit as st
from firebase_init import db
from datetime import datetime


def show():
    st.set_page_config(page_title="Feedback and Help", layout="centered")
    st.title("Feedback and Help")

    st.markdown("""
    We value your input! Please share your feedback on usability, performance, or feature requests.
    Your suggestions help us continuously improve Customer AI.
    """)

    with st.form("feedback_form"):
        feedback_type = st.selectbox("Feedback Type", ["Bug Report", "Feature Request", "General Feedback"])
        comments = st.text_area("Your Comments", placeholder="Describe your feedback here...")
        submit = st.form_submit_button("Submit Feedback")

        if submit:
            if not comments.strip():
                st.warning("Please enter some feedback before submitting.")
            else:
                user = st.session_state.get("user", {"email": "Anonymous"})
                feedback_entry = {
                    "user_email": user["email"],
                    "feedback_type": feedback_type,
                    "comments": comments.strip(),
                    "timestamp": datetime.utcnow().isoformat()
                }

                #saves feedback to firebase
                try:
                    db.collection("feedback").add(feedback_entry)
                    st.success("Thank you! Your feedback has been recorded.")
                except Exception as e:
                    st.error(f"Failed to save feedback: {e}")