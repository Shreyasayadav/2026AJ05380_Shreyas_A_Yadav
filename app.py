
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Page configuration
st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification")
st.write(
    "Upload your test dataset and select a trained machine learning model "
    "to evaluate its classification performance."
)

# Load saved models and scaler
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic_regression.joblib"),
        "Decision Tree": joblib.load("model/decision_tree.joblib"),
        "K-Nearest Neighbors": joblib.load("model/knn.joblib"),
        "Gaussian Naive Bayes": joblib.load("model/naive_bayes.joblib"),
        "Support Vector Machine": joblib.load("model/svm.joblib")
    }

    scaler = joblib.load("model/scaler.joblib")

    return models, scaler

models, scaler = load_models()

# Dataset upload
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.dataframe(data.head())

    target_column = "Class"

    if target_column not in data.columns:
        st.error("The uploaded dataset must contain the 'Class' target column.")

    else:
        X_test = data.drop(columns=[target_column])
        y_test = data[target_column]

        expected_features = list(scaler.feature_names_in_)

        if list(X_test.columns) != expected_features:
            st.error(
                "The uploaded dataset must contain the required input features "
                "in the same order as the training dataset."
            )

        else:
            # Apply the existing fitted scaler
            X_test_scaled = scaler.transform(X_test)

            # Model selection
            selected_model_name = st.selectbox(
                "Select Classification Model",
                list(models.keys())
            )

            selected_model = models[selected_model_name]

            # Predictions
            y_pred = selected_model.predict(X_test_scaled)
            y_prob = selected_model.predict_proba(X_test_scaled)

            # Evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)

            auc = roc_auc_score(
                y_test,
                y_prob,
                multi_class="ovr",
                average="weighted"
            )

            precision = precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            )

            recall = recall_score(
                y_test, y_pred, average="weighted", zero_division=0
            )

            f1 = f1_score(
                y_test, y_pred, average="weighted", zero_division=0
            )

            mcc = matthews_corrcoef(y_test, y_pred)

            # Display metrics
            st.subheader("Evaluation Metrics")

            col1, col2, col3 = st.columns(3)

            col1.metric("Accuracy", f"{accuracy:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{precision:.4f}")

            col4, col5, col6 = st.columns(3)

            col4.metric("Recall", f"{recall:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            # Confusion matrix
            st.subheader("Confusion Matrix")

            labels = selected_model.classes_

            cm = confusion_matrix(
                y_test,
                y_pred,
                labels=labels
            )

            fig, ax = plt.subplots(figsize=(9, 7))

            display = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=labels
            )

            display.plot(
                ax=ax,
                cmap="Blues",
                xticks_rotation=45
            )

            st.pyplot(fig)
            plt.close(fig)

            # Classification report
            st.subheader("Classification Report")

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True,
                zero_division=0
            )

            report_df = pd.DataFrame(report).transpose()

            st.dataframe(report_df)

else:
    st.info("Please upload the test_data.csv file to begin evaluation.")
