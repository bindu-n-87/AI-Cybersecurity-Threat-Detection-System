from src.preprocess import load_data, clean_data, encode_data, scale_features
from src.model import train_isolation_forest, predict_anomalies, train_random_forest, evaluate_model
from src.visualize import plot_confusion_matrix, plot_attack_distribution, plot_anomalies
from src.alert_system import generate_alerts, save_alerts, print_sample_alerts
from src.final_report import generate_report

df = load_data("data/UNSW_NB15_training-set.csv")

df = clean_data(df)
df, encoders = encode_data(df)
X, y, scaler = scale_features(df)

iso_model = train_isolation_forest(X)
anomaly_preds = predict_anomalies(iso_model, X)

print("\nSample Anomaly Predictions:")
print(anomaly_preds[:20])

rf_model = train_random_forest(X, y)
y_pred = evaluate_model(rf_model, X, y)

# Visualization
plot_confusion_matrix(y, y_pred)
plot_attack_distribution(y)
plot_anomalies(anomaly_preds)

# Alert System
alerts = generate_alerts(anomaly_preds, y)
print_sample_alerts(alerts)
alerts_df = save_alerts(alerts)

generate_report(y, anomaly_preds, alerts)