import streamlit as st

st.set_page_config(page_title="Tasks visualizations", page_icon="🧠", layout="wide")
st.title("Accuracy and RT analyses")
st.markdown("""
Welcome! Use the sidebar to navigate between tasks:
- **Visual Recognition** — RT and accuracy across training conditions
- **Association Task** — Matching/mismatch breakdown and learning curves
""")
