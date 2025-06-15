# Introduction To Data Science Project
## Bank Term Deposit Subscription Predictor
 
This is my project for the **Introduction to Data Science** course.  
In it, built a machine learning model and an interactive Streamlit app that predicts whether a bank client will subscribe to a term deposit.

## Project Overview

The project uses a real-world marketing dataset from a Portuguese bank.  
The goal is to help the bank **predict which clients are likely to subscribe** to a term deposit — saving time and effort in marketing campaigns.

## 📊 About the Project

-  Exploratory Data Analysis (EDA)
-  Data Preprocessing (encoding, scaling, etc.)
-  Machine Learning using Logistic Regression
-  An interactive Streamlit web app
-  Insights on which features impact the prediction

## 📚 Dataset Details

-  **Source**: [Bank Marketing Dataset on Kaggle](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
-  **Target column**: `deposit` (yes/no)
-  16 Features include: age, job, education, balance, contact method, etc.

### 📊 EDA (Exploratory Data Analysis)
- Summary statistics
- Histograms, boxplots, and count plots
- Grouped analysis (e.g., deposit rate by marital status)
- Outlier detection using IQR
- Feature distribution and correlation matrix

### 🔄 Preprocessing
- Encoded categorical features using `LabelEncoder` and `OrdinalEncoder`
- Scaled numeric features with `StandardScaler`
- Removed the `duration` column to avoid target leakage 
- Saved encoders and the model using `joblib`

### 🧠 Model
- Used **Logistic Regression** for its simplicity and interpretability
- Evaluated using:
  - Accuracy
  - Confusion matrix
  - Classification report

## 🌐 The Streamlit App

I created a simple and interactive app using **Streamlit** where:
- You can enter client details from the sidebar
- The app predicts whether the client is likely to subscribe
- It also includes an Introduction, EDA insights, Model Summary, and Conclusion

## 🚀 How to Run the App

If you want to try it out:

1. **Clone this repository**:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app_model.py
