import pandas as pd
import os

def generate_report(y, y_pred, alerts):
    print("\nFINAL SYSTEM REPORT")
    print("-" * 40)

    total = len(y)
    attacks = sum(y)
    detected = sum(y_pred)
    alerts_count = len(alerts)

    print(f"Total Records: {total}")
    print(f"Actual Attacks: {attacks}")
    print(f"Detected Anomalies: {detected}")
    print(f"Generated Alerts: {alerts_count}")

    print("\nSystem Performance Summary Complete")
    
    # Save report
    os.makedirs("outputs", exist_ok=True)

    report = {
        "Total Records": [total],
        "Actual Attacks": [attacks],
        "Detected Anomalies": [detected],
        "Alerts Generated": [alerts_count]
    }

    df = pd.DataFrame(report)
    df.to_csv("outputs/final_report.csv", index=False)

    print("Report saved to outputs/final_report.csv")
