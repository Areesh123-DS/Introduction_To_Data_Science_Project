import streamlit as st
import pandas as pd
import matplotlib.pyplot as mp
import joblib

# Load saved objects
model = joblib.load('lg_model.pkl')
scaler = joblib.load('scaler_model.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')
label_encoder = joblib.load('binary_label_encoders.pkl')
model_cols = joblib.load('model_cols.pkl')
label_en_y = joblib.load('label_encoder_y.pkl')

st.set_page_config(page_title="Bank Deposit Prediction")
st.title(" Bank Deposit Subscription Predictor")

st.sidebar.header("Input Client Details")

def user_input():
    age = st.sidebar.slider("Age", 18, 95, 35)
    balance = st.sidebar.number_input("Account Balance", value=1000)
    
    campaign = st.sidebar.slider("Contacts During Campaign", 1, 50, 2)
    pdays = st.sidebar.slider("Days Since Last Contact", -1, 999, 999)
    previous = st.sidebar.slider("Previous Campaign Contacts", 0, 50, 0)

    job = st.sidebar.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                                       'retired', 'self-employed', 'services', 'student', 'technician',
                                       'unemployed', 'unknown'])
    marital = st.sidebar.selectbox("Marital Status", ['married', 'single', 'divorced'])
    education = st.sidebar.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
    contact = st.sidebar.selectbox("Contact Type", ['cellular', 'telephone'])
    month = st.sidebar.selectbox("Contact Month", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul',
                                                        'aug', 'sep', 'oct', 'nov', 'dec'])
    day = st.sidebar.slider("Day of Month", 1, 31, 15)
    poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", ['failure', 'other', 'success', 'unknown'])

    housing = st.sidebar.selectbox("Housing Loan", ['yes', 'no'])
    loan = st.sidebar.selectbox("Personal Loan", ['yes', 'no'])
    default = st.sidebar.selectbox("Has Credit in Default?", ['yes', 'no'])

    # Create DataFrame
    data = pd.DataFrame({
        'age': [age],
        'balance': [balance],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'contact': [contact],
        'month': [month],
        'day': [day],
        'poutcome': [poutcome],
        'housing': [housing],
        'loan': [loan],
        'default': [default]
    })
    return data

input_data = user_input()

# Label encode binary columns
for col in ['housing', 'loan', 'default']:
    input_data[col] = label_encoder[col].transform(input_data[col])

# Ordinal encode multi-category columns
input_data = ordinal_encoder.transform(input_data)

# Scale numeric features
numeric_cols = ['age', 'balance', 'campaign', 'pdays', 'previous']
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

# Reorder columns to match training
input_data = input_data[model_cols]

prediction = model.predict(input_data)[0]           
predicted_label = label_en_y.inverse_transform([prediction])[0]  # 'yes' or 'no'
pred_proba = model.predict_proba(input_data)[0][1]   # Probability for class '1'

st.subheader("Prediction Result")

if prediction == 1:
    st.success("✅ The client is LIKELY to subscribe.")
else:
    st.warning("⚠️ The client is UNLIKELY to subscribe.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Sections**")
section = st.sidebar.radio("Navigate to", ['Introduction', 'EDA', 'Model Summary', 'Conclusion'])

if section == 'Introduction':
    st.header("📌 Introduction")
    st.markdown("""
    This Streamlit app predicts whether a bank client will subscribe to a term deposit, 
    based on information collected during a marketing campaign.  
    It uses a logistic regression model trained on the **Bank Marketing** dataset from Kaggle.
    """)

elif section == 'EDA':
    st.header("📊 Exploratory Data Analysis")
    st.markdown("I have done exploratory data analysis by pre-processing data and visualizing it using various plots.")

    data = pd.read_csv("bank.csv") 
    num_cols = ['age', 'balance', 'campaign', 'pdays', 'previous']  
    data[num_cols].hist(figsize=(14, 10), bins=30, color='pink', edgecolor='purple')
    mp.suptitle('Histogram of Features', fontsize=14)
    mp.tight_layout()

    st.pyplot(mp.gcf())

elif section == 'Model Summary':
    st.header("🧠 Model Details")
    st.markdown("""
    **Model Used:** I have used Logistic Regression model because my problem was classifying deposit on the basis of features. Decision Tree Classifier can also used for this dataset.
    **Preprocessing:**  
    - Label Encoding (binary columns)  
    - Ordinal Encoding (categorical columns)  
    - Feature Scaling (StandardScaler on numeric columns)  
      
    **Evaluation Metrics:**  
    - Accuracy  
    - Precision, Recall  
    - F1-Score  
    - Confusion Matrix  
    """)

elif section == 'Conclusion':
    st.header("📌 Conclusion")
    st.markdown("""
    - The logistic regression model effectively predicts likelihood to deposit money.This app takes on input data and make run-time prediction about the likelihood of a client subscribing to a term deposit.
    - I have done preprocessing and transformations to ensure consistemcy and accuracy.
    """)

