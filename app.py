import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("AI Cybersecurity Threat Detection Dashboard")

df = pd.read_csv("outputs/alerts_log.csv")

st.subheader("Alerts Log")
st.dataframe(df)

st.subheader("Alert Count")
st.write(len(df))

if st.button("Show Alerts Chart"):
    df["type"].value_counts().plot(kind="bar")
    st.pyplot()