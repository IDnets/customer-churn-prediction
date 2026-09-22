import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Generate Synthetic Customer Dataset
np.random.seed(42)
n_samples = 1000

data = {
    'Tenure_Months': np.random.randint(1, 72, size=n_samples),
    'Monthly_Charges': np.random.uniform(20.0, 120.0, size=n_samples),
    'Total_Charges': np.random.uniform(100.0, 8000.0, size=n_samples),
    'Tech_Support_Calls': np.random.randint(0, 10, size=n_samples),
    'Churn': np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
}

df = pd.DataFrame(data)

# 2. Features and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# 3. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training (Random Forest & Logistic Regression)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# 5. Predictions & Evaluation
rf_preds = rf_model.predict(X_test_scaled)
print("--- Random Forest Performance ---")
print(f"Accuracy: {accuracy_score(y_test, rf_preds):.2f}")
print(classification_report(y_test, rf_preds))
