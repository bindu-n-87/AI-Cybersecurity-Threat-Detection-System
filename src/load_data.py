import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    print("\nDataset Loaded Successfully!")
    print("Shape:", df.shape)
    return df

if __name__ == "__main__":
    train_path = "../data/UNSW_NB15_training-set.csv"
    
    df = load_dataset(train_path)

    # Show first 5 rows
    print("\nFirst 5 rows:")
    print(df.head())

    # Column info
    print("\nDataset Info:")
    print(df.info())

    # Check attack types
    print("\nAttack Types:")
    print(df['attack_cat'].value_counts())