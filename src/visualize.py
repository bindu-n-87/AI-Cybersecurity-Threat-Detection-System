import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

os.makedirs("outputs", exist_ok=True)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig("outputs/confusion_matrix.png")
    plt.show()

    print("Confusion matrix saved!")


def plot_attack_distribution(y):
    plt.figure()
    y.value_counts().plot(kind='bar')
    plt.title("Attack vs Normal Distribution")
    plt.xlabel("Class (0=Normal, 1=Attack)")
    plt.ylabel("Count")

    plt.savefig("outputs/attack_distribution.png")
    plt.show()

    print("Attack distribution saved!")


def plot_anomalies(anomalies):
    plt.figure()
    plt.plot(anomalies)
    plt.title("Anomaly Detection Output")
    plt.xlabel("Data Points")
    plt.ylabel("0=Normal, 1=Anomaly")

    plt.savefig("outputs/anomaly_graph.png")
    plt.show()

    print("Anomaly graph saved!")