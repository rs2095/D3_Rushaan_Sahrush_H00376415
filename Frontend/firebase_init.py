import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import json

#initializes firebase app and firestore client
if not firebase_admin._apps:
    try:
        firebase_creds = json.loads(st.secrets["FIREBASE_CREDENTIALS"])
        cred = credentials.Certificate(firebase_creds)
    except:
        cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()