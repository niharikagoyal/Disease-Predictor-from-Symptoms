import streamlit as st
import pickle
import pandas as pd

# Load model and vectorizer
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load original dataset to retrieve treatments
df = pd.read_csv("data/Diseases_Symptoms.csv")

st.set_page_config(page_title="🩺 Disease Predictor from Symptoms")
st.title("🩺 Disease Prediction from Symptoms")
st.write("Enter symptoms (comma-separated) and get predicted disease and treatment.")

# Input
symptoms_input = st.text_area("📝 Symptoms", placeholder="e.g. headache, fever, sore throat")

# Predict button
if st.button("Predict Disease"):
    if symptoms_input.strip() == "":
        st.warning("⚠️ Please enter some symptoms.")
    else:
        # Predict disease
        input_transformed = vectorizer.transform([symptoms_input])
        prediction = model.predict(input_transformed)[0]

        # Fetch treatment for the predicted disease
        treatment_row = df[df["Name"] == prediction]
        treatment = treatment_row["Treatments"].values[0] if not treatment_row.empty else "Treatment not available"

        # Show result
        st.success(f"🔍 Predicted Disease: **{prediction}**")
        st.info(f"💊 Recommended Treatment: **{treatment}**")
