
# ============================================
# STUDENT PERFORMANCE PREDICTION SYSTEM
# Basic Machine Learning Project
# ============================================

# STEP 1: Import libraries

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ============================================
# STEP 2: Create sample student dataset
# ============================================

data = {
    "Study_Hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 3,
                    5, 7, 1, 4, 6, 8, 2, 5, 7, 3],

    "Attendance": [50, 55, 60, 65, 70, 75, 80, 90, 52, 62,
                   72, 85, 45, 68, 78, 95, 58, 73, 88, 64],

    "Previous_Marks": [35, 40, 45, 50, 55, 60, 65, 80, 38, 47,
                       58, 72, 30, 52, 68, 85, 42, 61, 78, 49],

    "Assignment_Marks": [40, 42, 48, 55, 60, 65, 70, 85, 35, 50,
                          62, 75, 30, 58, 72, 90, 45, 66, 82, 52],

    "Internal_Marks": [35, 40, 45, 50, 55, 62, 68, 82, 38, 48,
                       60, 73, 28, 54, 70, 88, 40, 63, 80, 50],

    "Result": [
        "Fail", "Fail", "Fail", "Pass", "Pass",
        "Pass", "Pass", "Pass", "Fail", "Fail",
        "Pass", "Pass", "Fail", "Pass", "Pass",
        "Pass", "Fail", "Pass", "Pass", "Pass"
    ]
}


# Convert dictionary into DataFrame

df = pd.DataFrame(data)


# ============================================
# STEP 3: Display the dataset
# ============================================

print("Student Dataset:")
print(df)


# ============================================
# STEP 4: Check dataset information
# ============================================

print("\nDataset Information:")
# noinspection PyNoneFunctionAssignment
print(df.info())


# ============================================
# STEP 5: Check for missing values
# ============================================

print("\nMissing Values:")
print(df.isnull().sum())


# ============================================
# STEP 6: Convert Result into numbers
# Fail = 0
# Pass = 1
# ============================================

df["Result"] = df["Result"].map({
    "Fail": 0,
    "Pass": 1
})


print("\nAfter converting Result:")
print(df)


# ============================================
# STEP 7: Select input features
# ============================================

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Previous_Marks",
        "Assignment_Marks",
        "Internal_Marks"
    ]
]


# Target variable

y = df["Result"]


# ============================================
# STEP 8: Split data into training and testing
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining data:")
print(X_train)

print("\nTesting data:")
print(X_test)


# ============================================
# STEP 9: Create Machine Learning model
# ============================================

model = LogisticRegression()


# ============================================
# STEP 10: Train the model
# ============================================

model.fit(X_train, y_train)


print("\nModel training completed!")


# ============================================
# STEP 11: Make predictions
# ============================================

y_pred = model.predict(X_test)


print("\nPredicted values:")
print(y_pred)

print("\nActual values:")
print(y_test.values)


# ============================================
# STEP 12: Check accuracy
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)


# ============================================
# STEP 13: Classification report
# ============================================

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ============================================
# STEP 14: Confusion Matrix
# ============================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================
# STEP 15: Predict a NEW student
# ============================================

new_student = [[
    5,      # Study Hours
    85,     # Attendance
    70,     # Previous Marks
    75,     # Assignment Marks
    72      # Internal Marks
]]


prediction = model.predict(new_student)


# ============================================
# STEP 16: Display final prediction
# ============================================

if prediction[0] != 1:
    print("\n================================")
    print("PREDICTION: FAIL")
    print("================================")
else:
    print("\n================================")
    print("PREDICTION: PASS")
    print("================================")

