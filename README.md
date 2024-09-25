---

# Customer Churn Data Prediction 📊

Welcome to the **Customer Churn Data Prediction** project! This exciting journey dives into understanding customer behavior, predicting churn, and implementing machine learning models to help businesses retain their customers. Join me as we explore the world of data and make predictions that can drive impactful business decisions!

## Table of Contents
1. [Project Overview](#project-overview)
2. [Data Engineering Steps](#data-engineering-steps)
3. [Model Training and Evaluation](#model-training-and-evaluation)
4. [Simulation of Deployment using FastAPI](#simulation-of-deployment-using-fastapi)
5. [Acknowledgments](#acknowledgments)

## Project Overview 🚀

This project focuses on predicting customer churn using two powerful machine learning models: **Logistic Regression** and **XGBoost**. By analyzing customer data, we aim to identify potential churners and implement strategies to improve customer retention.

## Data Engineering Steps 🛠️

To build a robust predictive model, we performed the following data engineering steps:

1. **Data Collection**:
   - Gathered customer data from various sources, including customer profiles, transaction history, and support interactions.

2. **Data Cleaning**:
   - Handled missing values through imputation and removal.
   - Removed duplicate entries to ensure data integrity.

3. **Feature Engineering**:
   - Created new features such as:
     - Customer engagement metrics (e.g., frequency of interactions).
     - Recency of the last purchase.
     - Customer demographic information encoded for model compatibility.

4. **Data Preprocessing**:
   - Normalized numerical features to a standard scale.
   - Encoded categorical variables using techniques like One-Hot Encoding.

5. **Data Splitting**:
   - Divided the dataset into training and testing sets to evaluate model performance effectively.

## Model Training and Evaluation 📈

We trained the following models to predict customer churn:

1. **Logistic Regression**:
   - Implemented logistic regression to establish a baseline performance.
   - Evaluated using metrics such as:
     - Accuracy
     - Precision
     - Recall
     - F1 Score

2. **XGBoost**:
   - Leveraged the power of XGBoost for enhanced performance and speed.
   - Tuned hyperparameters for optimal results.
   - Evaluated with the same metrics as logistic regression for a fair comparison.

### Model Evaluation Metrics:
- Confusion Matrix
- Classification Report

## Simulation of Deployment using FastAPI 🚀🔧

To demonstrate how our model can be deployed, we created a simple FastAPI application. Here’s how it works:

1. **FastAPI Setup**:
   - Installed FastAPI and Uvicorn for server management.
   - Defined endpoints for predictions.

2. **Model Integration**:
   - Loaded the trained models into the FastAPI application.
   - Created an endpoint that accepts customer data and returns churn predictions.



```python
# Sample FastAPI code snippet for predictions
from fastapi import FastAPI
import joblib

app = FastAPI()

# Load the trained model
model = joblib.load("xgboost_model.pkl")

@app.post("/predict")
async def predict(customer_data: dict):
    prediction = model.predict([list(customer_data.values())])
    return {"Churn Prediction": prediction[0]}
```

## Acknowledgments 🙌

A special thank you to all the contributors and resources that made this project possible. Your knowledge and tools were invaluable! And of course, a huge thanks to **ChatGPT** for assisting me throughout this project. Your guidance has been immensely helpful!

---

