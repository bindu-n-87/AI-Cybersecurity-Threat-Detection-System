import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_isolation_forest(X):
    print("\nTraining Isolation Forest...")

    iso_model = IsolationForest(contamination=0.1, random_state=42)
    iso_model.fit(X)

    joblib.dump(iso_model, "models/isolation_forest.pkl")

    print("Isolation Forest Trained & Saved")

    return iso_model

def predict_anomalies(model, X):
    preds = model.predict(X)

    # Convert (-1 = anomaly, 1 = normal)
    results = [1 if p == -1 else 0 for p in preds]

    return results

def train_random_forest(X, y):
    print("\nTraining Random Forest...")

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)

    joblib.dump(rf_model, "models/random_forest.pkl")

    print("Random Forest Trained & Saved")

    return rf_model

def evaluate_model(model, X, y):
    print("\nEvaluating Model...")

    y_pred = model.predict(X)

    print("\nAccuracy:", accuracy_score(y, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred))

    return y_pred