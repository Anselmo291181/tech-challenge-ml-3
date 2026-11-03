import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def train_and_save_model_task():
    data_path = "data/laudos_medicos.csv"
    output_model_path = "models/model.joblib"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Ficheiro {data_path} não encontrado. Execute o script de geração de dados primeiro.")
    
    df = pd.read_csv(data_path)
    X = df["laudo_medico"]
    y = df["urgencia"]
    
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    model_pipeline.fit(X, y)
    
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(model_pipeline, output_model_path)
    print(f"Modelo treinado e salvo com sucesso em: {output_model_path}")

if __name__ == "__main__":
    train_and_save_model_task()