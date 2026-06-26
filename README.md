# 📊 Customer Churn Prediction (Machine Learning + Flask)

## 🚀 Project Overview
This project predicts whether a customer is likely to churn (leave the company) based on their service usage and account information.  
It uses a Machine Learning model trained on telecom customer data and is deployed using a Flask web application.

---

## 🎯 Problem Statement
Customer churn is a major problem for subscription-based businesses.  
The goal is to build a predictive model that identifies customers who are likely to stop using the service so that companies can take preventive actions.

---

## 🧠 Machine Learning Approach
- Data preprocessing and cleaning
- Label encoding for categorical variables
- Feature selection
- Model training using classification algorithm
- Model saved using `joblib`

---

## 📁 Project Structure
customer churn project/
│
├── app.py # Flask backend
├── train_model.py # Model training script
├── model/
│ └── churn_model.pkl # Trained ML model
│
├── templates/
│ ├── index.html # Input form UI
│ └── result.html # Prediction result page
│
├── static/
│ └── style.css # Styling
│
├── dataset/ # Dataset used for training
└── requirements.txt # Dependencies

---

## 🛠️ Technologies Used
- Python 🐍
- Flask 🌐
- Pandas
- NumPy
- Scikit-learn
- HTML, CSS

---

## 📊 Dataset
- Telecom customer churn dataset
- Features include:
  - Gender
  - Senior Citizen
  - Partner
  - Dependents
  - Tenure
  - Monthly Charges
  - Total Charges

---

## ⚙️ How to Run This Project

### 1. Clone the repository
```bash
git clone https://github.com/akhila126/customer-churn-prediction.git


