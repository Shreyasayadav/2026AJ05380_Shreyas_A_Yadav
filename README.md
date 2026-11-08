
# Dry Bean Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and evaluate machine learning classification models to identify different varieties of dry beans based on their morphological characteristics. Five classification algorithms are implemented and compared using multiple evaluation metrics.

## 2. Dataset Description

The Dry Beans Dataset contains numerical features describing the shape and characteristics of dry bean samples.

- Number of input features: 16
- Number of classes: 7
- Dataset instances after duplicate removal: 13,543
- Training samples: 10,834
- Testing samples: 2,709
- Target variable: Class

The dataset is used to train and evaluate five classification models.

## 3. GitHub Repository Link

Repository Link: To be added after creating the GitHub repository.

## 4. Models Used and Evaluation

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9199 | 0.9935 | 0.9205 | 0.9199 | 0.9200 | 0.9032 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| KNN | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.7630 | 0.9644 | 0.7647 | 0.7630 | 0.7607 | 0.7143 |
| SVM | 0.9210 | 0.9885 | 0.9214 | 0.9210 | 0.9211 | 0.9045 |

### Model Performance Observations

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Achieved 91.99% accuracy and the highest AUC score of 0.9935, indicating excellent classification performance. |
| Decision Tree | Achieved 89.55% accuracy, which was lower than the other models in this experiment. |
| KNN | Achieved 91.55% accuracy and an AUC score of 0.9811, showing good classification performance. |
| Gaussian Naive Bayes | Achieved the lowest accuracy of 76.30%, although its AUC score was 0.9644. |
| Support Vector Machine | Achieved the highest accuracy, precision, recall, F1 score, and MCC among the evaluated models. |

### Overall Winner

The Support Vector Machine (SVM) was the overall winner based on accuracy, precision, recall, F1 score, and MCC. It achieved an accuracy of 92.10% and an MCC score of 0.9045. Logistic Regression achieved the highest AUC score of 0.9935.

## 5. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## 6. Application Features

- Upload test data in CSV format
- Select a machine learning classification model
- Display evaluation metrics
- Visualize the confusion matrix
- Display the classification report

## 7. Project Files

- `app.py` - Streamlit application
- `requirements.txt` - Required Python packages
- `README.md` - Project documentation
- `test_data.csv` - Test dataset
- `model/` - Saved trained models and scaler
