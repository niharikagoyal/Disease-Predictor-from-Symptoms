# 🧠 Disease Prediction & Treatment Recommender System

This project predicts diseases based on user-typed symptoms using a trained Machine Learning model. Once a disease is predicted, the app provides relevant treatments associated with that disease.

## 🔍 Problem Statement

In areas where access to professional diagnosis is limited or delayed, users often resort to self-diagnosis. This project helps bridge that gap by predicting the most likely disease based on user-input symptoms and suggesting appropriate treatments.


## 🚀 Features

- ✅ Select symptoms using a user-friendly Streamlit interface
- ✅ Predict disease using a trained ML model (Decision Tree / Random Forest)
- ✅ Display recommended treatments after prediction
- ✅ Data-driven backend using a real-world Kaggle dataset
- ✅ Clean, interactive UI powered by Streamlit


## 📊 Dataset Used

- **Source**: [Kaggle - Disease Symptoms and Treatments Dataset](https://www.kaggle.com/datasets/snmahsa/disease-symptoms-and-treatments-dataset)
- **Columns**:
  - Name – Name of the disease
  - Symptoms – Comma-separated symptoms
  - Treatments – Common treatments
  - Disease_Code, Contagious, Chronic – Additional disease metadata


## 🧠 Machine Learning

- **Model**: Random Forest / Decision Tree Classifier
- **Training**: Dataset is preprocessed to map symptoms to disease labels
- **Serialization**: Model is saved as a `.pkl` file for reuse
- **Prediction**: Based on input symptoms, top disease is predicted


## 🛠 Tech Stack

| Tool            | Purpose                             |
|-----------------|-------------------------------------|
| Python          | Core programming language           |
| Pandas / NumPy  | Data manipulation                   |
| Scikit-learn    | ML model training and prediction    |
| Streamlit       | Web interface                       |
| Pickle          | Model serialization                 |


## 🧠 How It Works

1. User enters symptoms as free-text input (e.g., `"fever, cough, headache"`).
2. The app preprocesses the input and compares it with trained data.
3. A machine learning model predicts the most likely disease.
4. The corresponding treatment(s) are displayed from the dataset.



## 📁 Project Structure

.
├── app.py 
├── model/ # Trained ML model
├── data/ Diseases_Symptoms.csv # Dataset
├── train_model.py
└── README.md

## 🛠 How to Run Locally

# 1. Clone the repository
git clone https://github.com/niharikagoyal/Disease-Predictor-from-Symptoms.git
cd disease-prediction-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (this will save a .pkl file)
python train_model.py

# 4. Run the app
streamlit run app.py

## 🔮 Future Enhancements

OpenAI API for auto-generated disease descriptions

Intelligent auto-suggest or autocomplete for symptoms

Deployable on Streamlit Cloud, Hugging Face, or Render

## 📜 License

MIT License

## 👩‍💻 Author

Niharika Goyal
