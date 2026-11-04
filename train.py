import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from feature_engineering import prepare_features

def train_model():
    X, y, splits = prepare_features()
    
    # Divisão temporal: treina com 2022/2023 e valida em 2024
    train_idx = splits['season'] < 2024
    test_idx = splits['season'] == 2024
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Modelo XGBoost com peso para classes desbalanceadas (apenas 3 de 20 sobem ao pódio)
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Avaliação
    
    probs = model.predict_proba(X_test)[:, 1]
    print(f"Treinamento concluído. AUC-ROC: {roc_auc_score(y_test, probs):.4f}")
    
    joblib.dump(model, 'f1_model.pkl')
    print("modelo salvo em f1_model.pkl")

if __name__ == "__main__":
    train_model()