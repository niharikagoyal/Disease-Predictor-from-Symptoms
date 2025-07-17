import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load the dataset
df = pd.read_csv("data/Diseases_Symptoms.csv")

# Check for nulls
df.dropna(subset=["Symptoms", "Name"], inplace=True)

# Features and labels
X_raw = df["Symptoms"]
y = df["Name"]

# Vectorize symptoms text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_raw)

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save the model and vectorizer
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully!")
