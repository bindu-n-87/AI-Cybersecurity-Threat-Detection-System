import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    print("Data Loaded:", df.shape)
    print("Columns:", df.columns)
    return df

def clean_data(df):
    print("\nCleaning Data...")

    df = df.drop_duplicates()
    df = df.dropna()

    # Drop irrelevant columns
    df = df.drop(columns=['srcip', 'dstip'], errors='ignore')

    print("After Cleaning:", df.shape)
    return df

def encode_data(df):
    
    print("\nEncoding Categorical Features...")

    from sklearn.preprocessing import LabelEncoder

    label_encoders = {}

    # Select ONLY object columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    print("Categorical Columns:", categorical_cols)

    for col in categorical_cols:
        if col != 'attack_cat':  # keep attack_cat separate
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    print("Encoding Completed")
    return df, label_encoders

def scale_features(df):
    
    print("\nChecking for non-numeric columns:")
    print(df.select_dtypes(include=['object']).columns)

    print("\nScaling Features...")

    # Drop unnecessary columns
    df = df.drop(columns=['attack_cat'], errors='ignore')

    # Check remaining object columns
    print("Remaining object columns:",
          df.select_dtypes(include=['object']).columns)

    X = df.drop(columns=['label'])
    y = df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Scaling Done")

    return X_scaled, y, scaler

if __name__ == "__main__":
    path = "../data/UNSW_NB15_training-set.csv"

    df = load_data(path)
    df = clean_data(df)
    df, encoders = encode_data(df)
    X, y, scaler = scale_features(df)

    print("\nFINAL SHAPE:")
    print("X:", X.shape)
    print("y:", y.shape)