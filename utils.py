import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def load_data(path='data/Diseases_Symptoms.csv'):
    df = pd.read_csv(path)
    df['symptoms_list'] = df['Symptoms'].str.split(',').apply(lambda x: [s.strip() for s in x])
    return df

def prepare_features(df):
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['symptoms_list'])
    y = df['Disease']
    return X, y, mlb
