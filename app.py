# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Loan Prediction using Logistic Regression",
    page_icon="💰",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM STYLE
# -------------------------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7ff;
}

h1 {
    color: #6C63FF;
    text-align: center;
}

h2, h3 {
    color: #4A47A3;
}

.stButton>button {
    background-color: #6C63FF;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px;
}

.explain-box {
    background-color: #E8F0FE;
    padding: 15px;
    border-radius: 10px;
    color: black;
    margin-bottom: 10px;
}

.result-box {
    background-color: #D1FFD6;
    padding: 15px;
    border-radius: 10px;
    color: black;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🏦 Loan Prediction using Logistic Regression")

st.markdown("""
This project predicts **Loan Approval (0 or 1)** using **Logistic Regression**.

### Prediction Meaning:
- **1 → Loan Approved**
- **0 → Loan Rejected**
""")

# -------------------------------------------------
# LOAD DATASET
# -------------------------------------------------
df = pd.read_csv("train_dataset1.csv")

st.header("1️⃣ Load Dataset")

st.write("### Dataset Preview")
st.dataframe(df.head())

st.write("### Dataset Shape")
st.write(df.shape)

st.write("### Dataset Columns")
st.write(df.columns.tolist())

# -------------------------------------------------
# UNDERSTAND DATASET
# -------------------------------------------------
st.header("2️⃣ Understand Dataset")

st.write("### Dataset Information")
st.write(df.dtypes)

st.write("### Statistical Summary")
st.dataframe(df.describe())

# -------------------------------------------------
# NULL VALUES
# -------------------------------------------------
st.header("3️⃣ Find and Handle Null Values")

st.markdown("""
<div class="explain-box">
<b>Why are we checking null values?</b><br>
Machine learning models cannot work properly if missing values exist.
So we fill missing values using:
<ul>
<li><b>Median</b> → for numerical columns</li>
<li><b>Mode</b> → for categorical columns</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.write("### Null Values Before Handling")
st.write(df.isnull().sum())

fig, ax = plt.subplots(figsize=(10,5))
sns.heatmap(df.isnull(), cmap="viridis", cbar=False)
plt.title("Missing Values Heatmap")
st.pyplot(fig)

# Fill null values
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])

df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])

st.write("### Null Values After Handling")
st.write(df.isnull().sum())

# -------------------------------------------------
# CHARACTER TO NUMERIC
# -------------------------------------------------
# Convert categorical values into numeric

df['Gender'] = df['Gender'].map({
    'Male': 1,
    'Female': 0
})

df['Married'] = df['Married'].map({
    'Yes': 1,
    'No': 0
})

df['Education'] = df['Education'].map({
    'Graduate': 1,
    'Not Graduate': 0
})

df['Self_Employed'] = df['Self_Employed'].map({
    'Yes': 1,
    'No': 0
})

df['Property_Area'] = df['Property_Area'].map({
    'Rural': 0,
    'Semiurban': 1,
    'Urban': 2
})

# Fix Dependents
df['Dependents'] = (
    df['Dependents']
    .astype(str)
    .replace('3+', '3')
)

df['Dependents'] = pd.to_numeric(
    df['Dependents'],
    errors='coerce'
)

# Fix Loan_Status safely
df['Loan_Status'] = (
    df['Loan_Status']
    .astype(str)
    .str.strip()
    .str.upper()
)

df['Loan_Status'] = df['Loan_Status'].map({
    'Y': 1,
    'N': 0
})

# Remove invalid target rows
df = df.dropna(subset=['Loan_Status'])

# Convert to int
df['Loan_Status'] = df['Loan_Status'].astype(int)

# Drop Loan_ID
if 'Loan_ID' in df.columns:
    df.drop("Loan_ID", axis=1, inplace=True)

# Final cleanup
df.dropna(inplace=True)

st.write(df.head())

# -------------------------------------------------
# OUTLIERS
# -------------------------------------------------
st.header("5️⃣ Find and Handle Outliers")

st.markdown("""
<div class="explain-box">
<b>Why check outliers?</b><br>
Extreme values can reduce model accuracy.
We use the <b>IQR method</b> to remove extreme outliers.
</div>
""", unsafe_allow_html=True)

numeric_cols = df.select_dtypes(include=np.number).columns

st.write("### Before Removing Outliers")

fig, ax = plt.subplots(figsize=(12,5))
df.boxplot(ax=ax)
plt.xticks(rotation=90)
plt.title("Before Outlier Removal")
st.pyplot(fig)

df_outlier_removed = df.copy()

for col in numeric_cols:
    if col != 'Loan_Status':
        Q1 = df_outlier_removed[col].quantile(0.25)
        Q3 = df_outlier_removed[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - (1.5 * IQR)
        upper = Q3 + (1.5 * IQR)

        df_outlier_removed = df_outlier_removed[
            (df_outlier_removed[col] >= lower) &
            (df_outlier_removed[col] <= upper)
        ]

st.write("### After Removing Outliers")

fig, ax = plt.subplots(figsize=(12,5))
df_outlier_removed.boxplot(ax=ax)
plt.xticks(rotation=90)
plt.title("After Outlier Removal")
st.pyplot(fig)

df = df_outlier_removed

# -------------------------------------------------
# FEATURE SCALING
# -------------------------------------------------
st.header("6️⃣ Feature Scaling")

st.markdown("""
<div class="explain-box">
<b>Why Feature Scaling?</b><br>
Some columns have bigger numbers than others.
Scaling makes all features balanced so Logistic Regression performs better.
</div>
""", unsafe_allow_html=True)

scaler = StandardScaler()

feature_cols = df.drop("Loan_Status", axis=1).columns
df[feature_cols] = scaler.fit_transform(df[feature_cols])

st.success("Feature Scaling Completed")

# -------------------------------------------------
# CORRELATION
# -------------------------------------------------
st.header("7️⃣ Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation")
st.pyplot(fig)

# -------------------------------------------------
# TRAIN TEST SPLIT
# -------------------------------------------------
st.header("8️⃣ Train Test Split")

X = df.drop("Loan_Status", axis=1)

# Ensure target is integer
y = df["Loan_Status"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

st.write("Training Shape:", X_train.shape)
st.write("Testing Shape:", X_test.shape)

# -------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------
st.header("9️⃣ Logistic Regression Model")

st.markdown("""
<div class="explain-box">
<b>Why Logistic Regression?</b><br>
Because Loan_Status contains only:
<ul>
<li>0 → Rejected</li>
<li>1 → Approved</li>
</ul>

Logistic Regression is used for <b>binary classification</b>.
</div>
""", unsafe_allow_html=True)

model = LogisticRegression()
model.fit(X_train, y_train)

st.success("Model Trained Successfully")

# -------------------------------------------------
# TEST MODEL
# -------------------------------------------------
st.header("🔟 Model Testing")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.metric("Model Accuracy", f"{accuracy:.2f}")

# -------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------
st.header("1️⃣1️⃣ Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

st.pyplot(fig)

st.text(classification_report(y_test, y_pred))

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
st.header("1️⃣2️⃣ Predict Loan Approval")

st.markdown("""
<div class="explain-box">
Enter applicant details below.

The model predicts:

✅ <b>1 → Loan Approved</b><br>
❌ <b>0 → Loan Rejected</b>

We show readable options like Male/Female
instead of numbers to make the project
easy to understand.
</div>
""", unsafe_allow_html=True)

# USER INPUTS
gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

married = st.selectbox(
    "Marital Status",
    ["Yes", "No"]
)

dependents = st.slider(
    "Dependents",
    0,
    3
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_emp = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income = st.number_input(
    "Applicant Income",
    min_value=0
)

co_income = st.number_input(
    "Coapplicant Income",
    min_value=0
)

loan_amt = st.number_input(
    "Loan Amount",
    min_value=0
)

loan_term = st.number_input(
    "Loan Amount Term",
    min_value=0
)

credit = st.selectbox(
    "Credit History",
    ["Good", "Bad"]
)

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

# Encoding explanation
st.info("""
Encoding Used Internally:

• Male = 1, Female = 0  
• Yes = 1, No = 0  
• Graduate = 1, Not Graduate = 0  
• Good Credit = 1, Bad Credit = 0  
• Rural = 0, Semiurban = 1, Urban = 2
""")

if st.button("Predict Loan Status"):

    # Convert text inputs into numbers
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    education = 1 if education == "Graduate" else 0
    self_emp = 1 if self_emp == "Yes" else 0
    credit = 1 if credit == "Good" else 0

    property_area = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }[property_area]

    input_data = np.array([[
        gender,
        married,
        dependents,
        education,
        self_emp,
        income,
        co_income,
        loan_amt,
        loan_term,
        credit,
        property_area
    ]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:

        st.success("✅ Loan Approved")

        st.metric(
            "Approval Probability",
            f"{probability[0][1]*100:.2f}%"
        )

        st.markdown("""
        <div class="result-box">
        <h4>Why Approved?</h4>

        ✔ Good financial profile<br>
        ✔ Strong approval pattern<br>
        ✔ Model predicts higher approval chance
        </div>
        """, unsafe_allow_html=True)

    else:

        st.error("❌ Loan Rejected")

        st.metric(
            "Rejection Probability",
            f"{probability[0][0]*100:.2f}%"
        )

        st.markdown("""
        <div class="result-box">
        <h4>Why Rejected?</h4>

        ❌ Risky loan profile<br>
        ❌ Lower approval pattern match<br>
        ❌ Model predicts rejection chance
        </div>
        """, unsafe_allow_html=True)
