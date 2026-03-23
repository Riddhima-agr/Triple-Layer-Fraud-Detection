import pandas as pd
import numpy as np
from preprocessing import load_data, preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


# -----------------------------
# Load dataset
# -----------------------------
# def load_data():
#     df = pd.read_csv("data/paysim.csv")
#     return df


# -----------------------------
# Preprocessing
# -----------------------------
# def preprocess_data(df):

#     df = df.drop(['nameOrig', 'nameDest'], axis=1)

#     df = pd.get_dummies(df, columns=['type'], drop_first=True)

#     df = df.fillna(0)

#     return df


# -----------------------------
# Feature Engineering
# -----------------------------
def create_features(df):

    df['rule_score'] = 0

    df.loc[df['amount'] > 200000, 'rule_score'] += 1
    df.loc[df['oldbalanceOrg'] == 0, 'rule_score'] += 1
    df.loc[df['newbalanceOrig'] == 0, 'rule_score'] += 1

    return df


# -----------------------------
# Train model
# -----------------------------

def train_model(df):
    df = df.sample(frac=0.2, random_state=42)

    y = df['isFraud']

    X = df.drop([
        'isFraud',
        'isFlaggedFraud'
    ], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # imbalance handling
    scale = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )

    model.fit(X_train, y_train)

    # 🔥 Probability predictions
    y_probs = model.predict_proba(X_test)[:, 1]

    # 🔥 Threshold tuning
    threshold = 0.95
    y_pred = (y_probs > threshold).astype(int)

    print("\nModel Performance (Threshold =", threshold, "):\n")
    print(classification_report(y_test, y_pred))

    return model


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    print("Loading data...")
    df = load_data()

    print("Preprocessing...")
    df = preprocess_data(df)

    print("Creating rule-based features...")
    df = create_features(df)

    print("Training model...")
    model = train_model(df)

    print("\nDone ✅")