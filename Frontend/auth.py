import streamlit as st
import requests
import firebase_admin
from firebase_admin import credentials, firestore
import os
from dotenv import load_dotenv
import json

if not firebase_admin._apps:
    try:
        firebase_creds = json.loads(st.secrets["FIREBASE_CREDENTIALS"])
        cred = credentials.Certificate(firebase_creds)
    except:
        cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)
db = firestore.client()

try:
    API_KEY = st.secrets["FIREBASE_API_KEY"]
except:
    load_dotenv()
    API_KEY = os.getenv("FIREBASE_API_KEY")

st.title("Firebase Auth Test")
menu = ["Login", "Sign Up"]
choice = st.sidebar.selectbox("Menu", menu)


#signs up user via firebase auth api
def signup(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload)
    return r.json()


#logs in user via firebase auth api
def login(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    r = requests.post(url, json=payload)
    return r.json()


if choice == "Sign Up":
    email = st.text_input("Email", key="signup_email")
    password = st.text_input("Password", type="password", key="signup_pass")
    role = st.selectbox("Select Role", ["user", "admin"], key="signup_role")
    if st.button("Sign Up"):
        res = signup(email, password)
        if "error" in res:
            st.error(res["error"]["message"])
        else:
            uid = res["localId"]
            db.collection("users").document(uid).set({"email": email, "role": role})
            st.success(f"Account created as '{role}'! Please log in.")

elif choice == "Login":
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        res = login(email, password)
        if "error" in res:
            st.error(res["error"]["message"])
        else:
            st.session_state["user"] = res
            st.success("Logged in successfully!")