import pandas as pd

# -----------------------------
# Load dataset
# -----------------------------
def load_data():
    df = pd.read_csv("data/paysim.csv")
    return df


# -----------------------------
# Create custom fraud labels
# -----------------------------
def create_custom_labels(df):

    df['fraud_custom'] = 0

    # Rule 1: Very high amount + very low balance
    df.loc[
        (df['amount'] > 300000) & 
        (df['oldbalanceOrg'] < 2000),
        'fraud_custom'
    ] = 1

    # Rule 2: Large transaction wipes balance to zero
    df.loc[
        (df['oldbalanceOrg'] > 10000) &
        (df['newbalanceOrig'] == 0) &
        (df['amount'] > 200000),
        'fraud_custom'
    ] = 1

    # Rule 3: High-value transfer/cash-out to empty destination
    df.loc[
        (df['type'].isin(['TRANSFER', 'CASH_OUT'])) &
        (df['oldbalanceDest'] == 0) &
        (df['amount'] > 150000),
        'fraud_custom'
    ] = 1

    # Optional Rule 4 (extra strict filtering)
    df.loc[
        (df['step'] > 100) & 
        (df['fraud_custom'] == 1),
        'fraud_custom'
    ] = 1

    return df


# -----------------------------
# Run file directly
# -----------------------------
if __name__ == "__main__":
    df = load_data()

    print("Original Data Shape:", df.shape)

    df = create_custom_labels(df)

    print("\nSample Data:")
    print(df[['amount', 'type', 'fraud_custom']].head())

    print("\nFraud Distribution:")
    print(df['fraud_custom'].value_counts())