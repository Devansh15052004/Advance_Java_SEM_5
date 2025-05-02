# Advance_Java_SEM_5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Load dataset
df = pd.read_csv("your_dataset.csv")  # Replace with your file

# Basic info
print(df.head())
print(df.info())

# Drop rows with missing values for simplicity
df = df.dropna()

# ------------------------
# Regression
# ------------------------
features = ['study_hours_per_day', 'attendance_percentage', 'sleep_hours',
            'exercise_frequency', 'stress_level', 'exam_anxiety_score',
            'motivation_level', 'time_management_score', 'previous_gpa']
target = 'exam_score'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Linear Regression
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

# Evaluate regression
print("Regression MSE:", mean_squared_error(y_test, y_pred))

# Plot actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Exam Score")
plt.ylabel("Predicted Exam Score")
plt.title("Actual vs Predicted Exam Score")
plt.show()

# ------------------------
# Classification on regression output
# ------------------------

# Define function to classify scores
def classify(score):
    if score <= 40:
        return 0  # Low
    elif score <= 70:
        return 1  # Medium
    else:
        return 2  # High

# Apply to actual and predicted scores
y_test_class = y_test.apply(classify)
y_pred_class = pd.Series(y_pred).apply(classify)

# Logistic Regression Classifier (for reference)
clf_model = LogisticRegression(max_iter=200)
clf_model.fit(X_train, y_train.apply(classify))

y_class_pred = clf_model.predict(X_test)

# Evaluate classification
print("Classifier Accuracy:", accuracy_score(y_test_class, y_class_pred))
print("Classification Report:\n", classification_report(y_test_class, y_class_pred))

# Confusion Matrix
cm = confusion_matrix(y_test_class, y_class_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Med', 'High'], yticklabels=['Low', 'Med', 'High'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Exam Score Classes")
plt.show()

# Optional: Correlation heatmap
corr = df[features + [target]].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()
