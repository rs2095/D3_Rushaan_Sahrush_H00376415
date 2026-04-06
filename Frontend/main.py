import streamlit as st
import requests
import uuid
from firebase_init import db
import os
from dotenv import load_dotenv
import home_page
import feedback_page
import executive_dashboard
import sustainability_analysis
import market_basket_analysis
import simulator_analysis
import data_analysis
import audit_history

try:
    API_KEY = st.secrets["FIREBASE_API_KEY"]
except:
    load_dotenv()
    API_KEY = os.getenv("FIREBASE_API_KEY")

#generates a unique organization code for admin accounts
def generate_org_code():
    return uuid.uuid4().hex[:8].upper()


#signs up user and saves role and org to firestore
def signup(email, password, role, org_id):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload).json()
    if "error" not in r:
        uid = r["localId"]
        db.collection("users").document(uid).set({
            "email": email,
            "role": role,
            "organization_id": org_id
        })
    return r


#logs in user via firebase auth api
def login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    return requests.post(url, json=payload).json()


#validates organization code exists in firestore
def validate_org_code(org_code):
    orgs = db.collection("organizations").where("org_code", "==", org_code).get()
    if orgs:
        return orgs[0].to_dict()
    return None


#retrieves organization code for admin from firestore
def get_org_code(org_id):
    if not org_id:
        return None
    org_doc = db.collection("organizations").document(org_id).get()
    if org_doc.exists:
        return org_doc.to_dict().get("org_code")
    return None


#login and signup flow
if "user" not in st.session_state:
    st.set_page_config(page_title="Login", layout="centered")
    st.title("Customer AI: Organizational Intelligence")
    st.markdown("Access your analytics dashboard with personalized insights.")

    menu = ["Login", "Sign Up"]
    choice = st.selectbox("Select Action", menu)

    if choice == "Sign Up":
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Create an Account")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_pass")
        role = st.selectbox(
            "Role",
            ["Analyst", "Admin"],
            help="Analysts can access analytics dashboards. Admins can access all features including the Scenario Simulator and manage their organization."
        )

        if role == "Analyst":
            org_code_input = st.text_input(
                "Organization Code",
                key="org_code_input",
                help="Enter the organization code provided by your admin to join their workspace."
            )

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Sign Up"):
            if role == "Admin":
                org_code = generate_org_code()
                org_id = uuid.uuid4().hex

                db.collection("organizations").document(org_id).set({
                    "org_code": org_code,
                    "admin_email": email,
                    "created_by": email
                })

                res = signup(email, password, "admin", org_id)
                if "error" in res:
                    st.error(res["error"]["message"])
                else:
                    st.success(f"Admin account created! Your organization code is:")
                    st.code(org_code)
                    st.info("Share this code with your team members so they can join your organization as Analysts.")

            elif role == "Analyst":
                if not org_code_input or not org_code_input.strip():
                    st.error("Please enter an organization code to sign up as an Analyst.")
                else:
                    org_data = validate_org_code(org_code_input.strip().upper())
                    if org_data is None:
                        st.error("Invalid organization code. Please check with your admin and try again.")
                    else:
                        org_docs = db.collection("organizations").where("org_code", "==", org_code_input.strip().upper()).get()
                        org_id = org_docs[0].id

                        res = signup(email, password, "analyst", org_id)
                        if "error" in res:
                            st.error(res["error"]["message"])
                        else:
                            st.success(f"Analyst account created under organization managed by {org_data['admin_email']}. Please log in.")

    elif choice == "Login":
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Log In")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_pass")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Login"):
            res = login(email, password)
            if "error" in res:
                st.error(res["error"]["message"])
            else:
                uid = res["localId"]
                user_doc = db.collection("users").document(uid).get()
                user_data = user_doc.to_dict() if user_doc.exists else {}

                st.session_state["user"] = {
                    "localId": uid,
                    "email": user_data.get("email", email),
                    "role": user_data.get("role", "analyst"),
                    "organization_id": user_data.get("organization_id", "")
                }
                st.success(f"Logged in successfully as {st.session_state['user']['role'].title()}!")
                st.rerun()

    st.stop()

st.set_page_config(page_title="Customer AI Platform", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "home_page"

#sidebar navigation and user info
st.sidebar.markdown("### Customer AI\n*Predicting Customer Dynamics and Business Outcomes*")
st.sidebar.markdown("---")

user = st.session_state["user"]
st.sidebar.markdown(
    f"**Account:**\n{user['email']}"
    f"\n\n**Role:** {user['role'].title()}"
)

#displays organization code for admin users
if user["role"].lower() == "admin":
    org_code = get_org_code(user.get("organization_id", ""))
    if org_code:
        st.sidebar.markdown(f"**Org Code:** `{org_code}`")

st.sidebar.markdown("---")
st.sidebar.markdown("### Strategic Insights")


#handles sidebar navigation button with optional admin restriction
def nav_button(label, target_page, admin_only=False):
    if st.sidebar.button(label, use_container_width=True):
        if admin_only and st.session_state["user"]["role"].lower() != "admin":
            st.warning("Only admins can access this page.")
        else:
            st.session_state.page = target_page
            st.rerun()


nav_button("Home", "home_page")
nav_button("Dashboard", "executive_dashboard")
nav_button("Sustainability Hub", "sustainability_analysis")
nav_button("Market Insights", "market_basket_analysis")
nav_button("Scenario Simulator", "simulator", admin_only=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### Data Integrity and Logs")
nav_button("Data Analysis", "data_analysis")
nav_button("Audit History", "audit_history", admin_only=True)
st.sidebar.markdown("---")
st.sidebar.markdown("### Customer Support")
nav_button("Feedback and Help", "feedback_page")

#handles logout and clears session state
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("Logout"):
    st.session_state.pop("user", None)
    st.session_state.pop("page", None)
    st.session_state.pop("df", None)
    st.session_state.pop("uploaded_file", None)
    st.session_state.pop("last_uploaded_hash", None)
    st.session_state.pop("last_file_hash", None)

    eco_keys = ["eco_clf", "eco_reg", "eco_models_loaded", "eco_logged_hash", "eco_user_comments"]
    for key in eco_keys:
        st.session_state.pop(key, None)

    st.rerun()

#routes to selected page
page_map = {
    "home_page": home_page.show,
    "feedback_page": feedback_page.show,
    "executive_dashboard": executive_dashboard.show,
    "sustainability_analysis": sustainability_analysis.show,
    "market_basket_analysis": market_basket_analysis.show,
    "simulator": simulator_analysis.show,
    "data_analysis": data_analysis.show,
    "audit_history": audit_history.show,
}

page_func = page_map.get(st.session_state.page)
if page_func:
    page_func()