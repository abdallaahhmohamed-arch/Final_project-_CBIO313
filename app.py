import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.title("Diabetes Readmission Prediction")
st.write("CBIO313 Final Project")

df = pd.read_csv("diabetic_data.csv")

st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Simple Prediction App")

# Basic cleaning
df = df.replace("?", pd.NA)
df = df.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"], errors="ignore")
df = df.dropna()

# Target
y = df["readmitted"].apply(lambda x: 1 if x == "<30" else 0)
X = df.drop("readmitted", axis=1)

# Convert categorical columns
X = pd.get_dummies(X)

# Train simple model
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X, y)

st.success("Model trained successfully!")

uploaded_file = st.file_uploader("Upload diabetic CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview")
    st.dataframe(new_data.head())

    new_data = new_data.replace("?", pd.NA)
    new_data = new_data.drop(columns=["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty", "readmitted"], errors="ignore")
    new_data = pd.get_dummies(new_data)

    # Match training columns
    new_data = new_data.reindex(columns=X.columns, fill_value=0)

    predictions = model.predict(new_data)

    result = ["Readmitted within 30 days" if p == 1 else "Not readmitted within 30 days" for p in predictions]

    st.subheader("Prediction Results")
    st.write(result)