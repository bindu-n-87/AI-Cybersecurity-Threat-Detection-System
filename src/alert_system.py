import pandas as pd
import datetime
import os

os.makedirs("outputs", exist_ok=True)

def generate_alerts(anomaly_preds, y_true):
    import datetime

    y_true = y_true.reset_index(drop=True)

    alerts = []

    for i, val in enumerate(anomaly_preds):
        timestamp = datetime.datetime.now()

        if val == 1:
            alert = {
                "timestamp": timestamp,
                "index": i,
                "type": "ANOMALY DETECTED",
                "severity": "HIGH",
                "actual_label": int(y_true.iloc[i])
            }
            alerts.append(alert)

    return alerts

def save_alerts(alerts):
    df = pd.DataFrame(alerts)

    file_path = "outputs/alerts_log.csv"
    df.to_csv(file_path, index=False)

    print(f"Alerts saved to {file_path}")

    return df

def print_sample_alerts(alerts, n=10):
    print("\nSAMPLE ALERTS:")
    for alert in alerts[:n]:
        print(alert)
