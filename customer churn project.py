# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('customer_churn.csv')

# Data Cleaning
df = df.dropna()

# Feature Engineering
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Convert categorical variables to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Preparing Data for Modeling
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data with a test_size of 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Building and Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)
    print('Classification Report:')
    print(classification_report(y_test, y_pred, zero_division=0))
    return precision, recall, f1

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
print('Logistic Regression Performance:')
precision_lr, recall_lr, f1_lr = evaluate_model(log_reg, X_test, y_test)

# Random Forest Classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
print('Random Forest Performance:')
precision_rf, recall_rf, f1_rf = evaluate_model(rf_clf, X_test, y_test)

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)
print('Gradient Boosting Performance:')
precision_gb, recall_gb, f1_gb = evaluate_model(gb_clf, X_test, y_test)

# Plotting the results
model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
precisions = [precision_lr, precision_rf, precision_gb]
recalls = [recall_lr, recall_rf, recall_gb]
f1_scores = [f1_lr, f1_rf, f1_gb]

plt.figure(figsize=(10, 6))
bar_width = 0.2
index = np.arange(len(model_names))

plt.bar(index, precisions, bar_width, label='Precision', color='b')
plt.bar(index + bar_width, recalls, bar_width, label='Recall', color='g')
plt.bar(index + 2 * bar_width, f1_scores, bar_width, label='F1 Score', color='r')

plt.xlabel('Model')
plt.ylabel('Scores')
plt.title('Model Performance')
plt.xticks(index + bar_width, model_names)
plt.legend()

plt.tight_layout()
plt.show()
