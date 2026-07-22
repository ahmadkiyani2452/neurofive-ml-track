import streamlit as st
import pandas as pd
import joblib


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "titanic_random_forest_model.joblib"
)


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)


# ==========================================
# APP TITLE
# ==========================================

st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger information below to predict "
    "whether the passenger is likely to survive."
)


# ==========================================
# USER INPUTS
# ==========================================

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)


sex = st.selectbox(
    "Sex",
    ["female", "male"]
)


age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=30.0
)


sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)


parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)


fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=32.0
)


embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)


# ==========================================
# FEATURE ENGINEERING
# ==========================================

family_size = sibsp + parch + 1

is_alone = 1 if family_size == 1 else 0


# ==========================================
# CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame({
    "Pclass": [pclass],
    "Sex_male": [1 if sex == "male" else 0],
    "Age": [age],
    "SibSp": [sibsp],
    "Parch": [parch],
    "Fare": [fare],
    "Embarked_Q": [1 if embarked == "Q" else 0],
    "Embarked_S": [1 if embarked == "S" else 0],
    "FamilySize": [family_size],
    "IsAlone": [is_alone]
})


# ==========================================
# MATCH MODEL TRAINING FEATURES
# ==========================================

if hasattr(model, "feature_names_in_"):

    input_data = input_data.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )


# ==========================================
# PREDICTION
# ==========================================

if st.button("Predict Survival"):

    prediction = model.predict(
        input_data
    )[0]


    probability = model.predict_proba(
        input_data
    )[0]


    survival_probability = probability[1] * 100


    # ======================================
    # DISPLAY RESULT
    # ======================================

    if prediction == 1:

        st.success(
            "Prediction: Passenger is likely to SURVIVE 🚢"
        )

    else:

        st.error(
            "Prediction: Passenger is likely NOT to survive."
        )


    st.info(
        f"Estimated Survival Probability: "
        f"{survival_probability:.2f}%"
    )


    # ======================================
    # SHOW INPUT DATA
    # ======================================

    st.subheader("Passenger Information")

    st.dataframe(
        input_data
    )
