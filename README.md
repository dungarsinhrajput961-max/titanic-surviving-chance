# titanic-surviving-chance
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = sns.load_dataset('titanic')

# Display the first few rows and basic info
print(df.head())
print(df.info())
# Fill missing age values with the mean
df['age'].fillna(df['age'].mean(), inplace=True)

# Fill missing embarked and embark_town with the mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

# Drop the deck column
df.drop('deck', axis=1, inplace=True)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, columns=['sex', 'embarked', 'class', 'who', 'embark_town'], drop_first=True)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Define features (X) and target (y)
X = df.drop('survived', axis=1)
y = df['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Drop columns that are not useful for prediction
df.drop(['alive', 'adult_male', 'pclass'], axis=1, inplace=True)
