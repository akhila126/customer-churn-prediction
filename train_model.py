import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# Step 1: Load the Dataset
# -------------------------------
df = pd.read_csv("dataset/Customer-Churn.csv")

# Display original dataset size
print("Original Shape:", df.shape)

# -------------------------------
# Step 2: Remove Unnecessary Column
# -------------------------------
df.drop("customerID", axis=1, inplace=True)

# -------------------------------
# Step 3: Convert TotalCharges to Numeric
# -------------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# -------------------------------
# Step 4: Fill Missing Values
# -------------------------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# -------------------------------
# Step 5: Convert Text Columns into Numbers
# -------------------------------
encoder = LabelEncoder()

for column in df.select_dtypes(include="object").columns:
    df[column] = encoder.fit_transform(df[column])

print("\nProcessed Data:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# Step 6: Separate Features (X) and Target (y)
# -------------------------------
X = df[
    [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]
]

y = df["Churn"]

# -------------------------------
# Step 7: Split the Dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# -------------------------------
# Step 8: Train the Model
# -------------------------------
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# -------------------------------
# Step 9: Make Predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# Step 10: Calculate Accuracy
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
# Save the trained model
joblib.dump(model, "model/churn_model.pkl")

print("Model saved successfully!")