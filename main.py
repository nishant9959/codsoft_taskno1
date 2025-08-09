import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the Dataset ---
try:
    df = pd.read_csv('Titanic-Dataset.csv')
    print("Dataset loaded successfully.")
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\n" + "="*50 + "\n")
except FileNotFoundError:
    print("Error: 'Titanic-Dataset.csv' not found. Please make sure the file is in the correct directory.")
    exit()


# --- 2. Data Preprocessing and Feature Engineering ---
print("Starting data preprocessing...")

# Handle missing 'Age' values by filling them with the median age.
# The median is less sensitive to outliers than the mean.
df['Age'].fillna(df['Age'].median(), inplace=True)

# Handle missing 'Embarked' values by filling them with the most common port (mode).
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# The 'Cabin' column has too many missing values to be useful, so we'll drop it.
df.drop('Cabin', axis=1, inplace=True)

# Convert categorical 'Sex' and 'Embarked' columns into numerical format using one-hot encoding.
# drop_first=True avoids multicollinearity (dummy variable trap).
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Drop columns that are unlikely to be useful for prediction.
# 'PassengerId', 'Name', and 'Ticket' are unique identifiers and don't provide general patterns.
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

print("Data after preprocessing:")
print(df.head())
print("\n" + "="*50 + "\n")


# --- 3. Define Features (X) and Target (y) ---
# We'll separate our data into features (the input variables) and the target (what we want to predict).
# 'X' contains all columns except 'Survived'.
# 'y' contains only the 'Survived' column.
X = df.drop('Survived', axis=1)
y = df['Survived']


# --- 4. Split Data into Training and Testing Sets ---
# We split the dataset into two parts:
# - A training set to train the model.
# - A testing set to evaluate the model's performance on unseen data.
# test_size=0.2 means 20% of the data will be used for testing.
# random_state ensures that the splits are the same every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} passengers")
print(f"Testing set size: {X_test.shape[0]} passengers")
print("\n" + "="*50 + "\n")


# --- 5. Train the Machine Learning Model ---
# We'll use a RandomForestClassifier. It's an ensemble model that is powerful,
# handles non-linear relationships well, and is less prone to overfitting than a single decision tree.
print("Training the RandomForestClassifier model...")

# n_estimators=100 means the forest will be built from 100 decision trees.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model training complete.")
print("\n" + "="*50 + "\n")


# --- 6. Evaluate the Model ---
# Now we'll use the trained model to make predictions on the test set and see how well it performs.
print("Evaluating the model...")
predictions = model.predict(X_test)

# Calculate the accuracy of the model.
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.4f}")

# The classification report gives us more detailed metrics like precision, recall, and f1-score.
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# A confusion matrix is a great way to visualize the performance.
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

# Plotting the confusion matrix for better visualization.
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Did not Survive', 'Survived'], yticklabels=['Did not Survive', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# --- 7. Feature Importance ---
# Let's see which features the model found most important for making predictions.
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importance_df)

# Plotting feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importances in Titanic Survival Prediction')
plt.show()


