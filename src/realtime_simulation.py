import time
import pandas as pd
from src.preprocess import load_data, clean_data, encode_data, scale_features
from src.model import train_isolation_forest, predict_anomalies
from src.alert_system import generate_alerts


def simulate_realtime(data_path, batch_size=500, delay=2):
    print("\nStarting Real-Time Simulation...\n")

    # Load and preprocess once
    df = load_data(data_path)
    df = clean_data(df)
    df, encoders = encode_data(df)
    X, y, scaler = scale_features(df)

    # Train model
    model = train_isolation_forest(X)

    total_batches = len(X) // batch_size

    for i in range(total_batches):
        start = i * batch_size
        end = start + batch_size

        X_batch = X[start:end]
        y_batch = y.iloc[start:end].reset_index(drop=True)

        print(f"\nProcessing Batch {i+1}/{total_batches}")

        # Predict anomalies
        preds = predict_anomalies(model, X_batch)

        # Generate alerts
        alerts = generate_alerts(preds, y_batch)

        print(f"Alerts in this batch: {len(alerts)}")

        if len(alerts) > 0:
            print("Sample Alert:", alerts[0])

        # Simulate real-time delay
        time.sleep(delay)

    print("\nReal-Time Simulation Completed!")

if __name__ == "__main__":
    simulate_realtime("data/UNSW_NB15_training-set.csv")