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

# -------------------------
# USER INPUTS
# -------------------------

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

# -------------------------
# ENCODING EXPLANATION
# -------------------------
st.info("""
Encoding Used Internally:

• Male = 1, Female = 0  
• Yes = 1, No = 0  
• Graduate = 1, Not Graduate = 0  
• Good Credit = 1, Bad Credit = 0  
• Rural = 0, Semiurban = 1, Urban = 2
""")

# -------------------------
# PREDICT BUTTON
# -------------------------
if st.button("Predict Loan Status"):

    # Convert categorical inputs to numeric
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

    # Create input array
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

    # Prediction Probability
    probability = model.predict_proba(input_scaled)

    # Result
    if prediction[0] == 1:

        st.success("✅ Loan Approved")

        st.metric(
            "Approval Probability",
            f"{probability[0][1]*100:.2f}%"
        )

        st.markdown("""
        <div class="result-box">
        <h4>Why Approved?</h4>

        ✔ Good financial pattern<br>
        ✔ Suitable loan profile<br>
        ✔ Model found approval-like characteristics
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
        ❌ Low approval pattern match<br>
        ❌ Model found rejection-like characteristics
        </div>
        """, unsafe_allow_html=True)
