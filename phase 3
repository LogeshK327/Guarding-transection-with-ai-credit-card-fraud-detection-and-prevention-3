# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Create a synthetic dataset (for demonstration)
data = {
    'Time': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Amount': [100.0, 50.0, 2000.0, 75.0, 150.0, 3000.0, 25.0, 500.0, 10000.0, 80.0],
    'V1': [-1.359, 1.191, -5.0, 0.966, -0.185, -7.0, 1.792, -0.418, -10.0, 1.257],
    'V2': [0.072, -0.173, 4.0, -0.287, 0.669, 5.5, -0.863, 0.403, 7.0, -0.211],
    'V3': [2.536, 0.405, -6.0, 1.798, 1.974, -8.0, 0.095, 0.762, -12.0, 0.988],
    'V4': [1.378, -0.338, 3.5, -0.094, 0.456, 4.0, -0.631, 0.175, 6.0, -0.403],
    'Class': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

# Step 2: Data Preprocessing
X = df.drop('Class', axis=1)
y = df['Class']
scaler = StandardScaler()
X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])

# Step 3: Train a Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Model Evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print("\nROC AUC Score:", roc_auc)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(recall, precision, label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()

# Save the model and scaler
joblib.dump(rf_model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
