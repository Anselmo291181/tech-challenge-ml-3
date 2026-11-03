import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train():
    print("Carregando dados...")
    # Lê o CSV gerado no passo anterior
    df = pd.read_csv("data/laudos_medicos.csv")
    
    # Assume que a 1ª coluna é o texto e a 2ª é a categoria (urgência)
    X = df.iloc[:, 0]
    y = df.iloc[:, 1]

    print("Treinando modelo Baseline (TF-IDF + Random Forest)...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000)),
        ('clf', RandomForestClassifier(n_estimators=50, random_state=42))
    ])
    pipeline.fit(X, y)

    # Cria a pasta models se não existir e salva o arquivo .joblib
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/model.joblib")
    print("Treinamento concluído. Modelo salvo em models/model.joblib")

if __name__ == "__main__":
    train()