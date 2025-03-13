import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load Dataset (Download from Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud)
df = pd.read_csv('creditcard.csv')

# Step 2: Data Preprocessing
X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Target (1 = Fraud, 0 = Legit)

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Step 5: Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
