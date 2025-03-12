import pandas as pd
from sklearn.preprocessing import StandardScaler # type: ignore

def preprocess_data(df):
    df = df.copy()

    # Encode categorical variables
    df = pd.get_dummies(df, columns=["Contract", "PaymentMethod"], drop_first=True)

    # Normalize numerical columns
    scaler = StandardScaler()
    df[["MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(df[["MonthlyCharges", "TotalCharges"]])

    return df

