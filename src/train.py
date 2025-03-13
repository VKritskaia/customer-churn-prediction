import pandas as pd
import pickle
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report # type: ignore

from preprocess import preprocess_data

# Load dataset
df = pd.read_csv("../data/raw/Telco-Customer-Churn.csv")
df = preprocess_data(df)

# Split data
X = df.drop(columns=["Churn"])
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("../api/model.pkl", "wb") as f:
    pickle.dump(model, f)

