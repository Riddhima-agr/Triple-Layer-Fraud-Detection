import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load dataset
# -----------------------------
def load_data():
    df = pd.read_csv("data/paysim.csv")
    return df


# -----------------------------
# Basic preprocessing
# -----------------------------
def preprocess_data(df):

    # Drop unnecessary columns
    df = df.drop(['nameOrig', 'nameDest'], axis=1)

    # Convert categorical to numeric
    df = pd.get_dummies(df, columns=['type'], drop_first=True)

    # Handle missing values (if any)
    df = df.fillna(0)
    return df
   

# -----------------------------
# Run file directly
# -----------------------------
if __name__ == "__main__":
    df = load_data()
    print("Original Data Shape:", df.shape)

    df = preprocess_data(df)
    print("After Preprocessing Shape:", df.shape)

    print("\nSample Data:")
    print(df.head())