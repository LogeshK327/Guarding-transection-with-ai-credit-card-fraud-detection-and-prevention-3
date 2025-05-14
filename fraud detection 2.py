import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import os
import zipfile
import kaggle

# Set random seed for reproducibility
np.random.seed(42)

# Streamlit app title
st.title("Fraud Detection with Random Forest")

# Display the updated date and time
st.write("**Date and Time:** 03:17 PM IST on Wednesday, May 14, 2025")

# Sidebar for user interaction
st.sidebar.header("Model Controls")
train_button = st.sidebar.button("Train Model")

# Step 1: Authenticate with Kaggle API and download the dataset
st.header("Dataset from Kaggle")

# Access the Kaggle API key from Streamlit secrets
try:
    kaggle_api_key = st.secrets["KAGGLE_API_KEY"]
except KeyError:
    st.error("KAGGLE_API_KEY not found in Streamlit secrets. Please set it in .streamlit/secrets.toml or Streamlit Cloud settings.")
    st.stop()

# Set Kaggle API credentials
os.environ["KAGGLE_USERNAME"] = "your_kaggle_username"  # Replace with your Kaggle username
os.environ["KAGGLE_KEY"] = kaggle_api_key

# Initialize Kaggle API
try:
    kaggle.api.authenticate()
except Exception as e:
    st.error(f"Error authenticating with Kaggle API: {e}")
    st.stop()

# Download the dataset (e.g., creditcardfraud dataset)
dataset = "creditcardfraud/creditcard.csv"  # Kaggle dataset path
download_path = "./dataset"

try:
    # Download the dataset as a ZIP file
    kaggle.api.dataset_download_files(dataset, path=download_path, unzip=False)
    
    # Extract the ZIP file
    zip_path = os.path.join(download_path, "creditcard.csv.zip")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    
    # Load the dataset into a Pandas DataFrame
    csv_path = os.path.join(download_path, "creditcard.csv")
    df = pd.read_csv(csv_path)
    st.dataframe(df)
    
    # Clean up downloaded files (optional)
    os.remove(zip_path)
except Exception as e:
    st.error(f"Error downloading or loading dataset from Kaggle: {e}")
    st.stop()

# Step 2: Data Preprocessing and Model Training
if train_button:
    st.header("Model Training and Evaluation")
    
    # Preprocessing
    X = df.drop('Class', axis=1)
    y = df['Class']
    scaler = StandardScaler()
    X[['Time', 'Amount']] = scaler.fit_transform(X[['Time', 'Amount']])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    # Model Evaluation
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))
    
    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader("Accuracy")
    st.write(f"{accuracy:.4f}")
    
    roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
    st.subheader("ROC AUC Score")
    st.write(f"{roc_auc:.4f}")
    
    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    st.pyplot(fig)
    
    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label='Precision-Recall Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    st.pyplot(fig)
    
    # Save model and scaler
    joblib.dump(rf_model, 'fraud_detection_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    st.success("Model and scaler saved successfully!")
