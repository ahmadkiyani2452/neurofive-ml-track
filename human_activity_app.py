import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Human Activity Recognition",
    page_icon="🏃",
    layout="wide"
)

# Load model and feature names
@st.cache_resource
def load_model():
    model = joblib.load("human_activity_logistic_regression_model.pkl")
    feature_names = joblib.load("human_activity_feature_names.pkl")
    return model, feature_names


model, feature_names = load_model()


# Activity mapping
activity_map = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING"
}


# App title
st.title("🏃 Human Activity Recognition System")

st.write(
    "Upload a sensor data CSV file to predict the human activity "
    "using a trained Machine Learning model."
)

st.info(
    "The model was trained using the UCI Human Activity Recognition dataset."
)


# File uploader
uploaded_file = st.file_uploader(
    "Upload Sensor Data CSV",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        # Read uploaded CSV
        input_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.write("Rows:", input_data.shape[0])
        st.write("Columns:", input_data.shape[1])

        st.dataframe(
            input_data.head()
        )


        # Check required features
        missing_features = [
            feature
            for feature in feature_names
            if feature not in input_data.columns
        ]


        if len(missing_features) > 0:

            st.error(
                f"The uploaded file is missing {len(missing_features)} "
                "required features."
            )

            st.warning(
                "Please upload a CSV file containing the same 563 "
                "features used during model training."
            )

            with st.expander("View missing features"):
                st.write(missing_features)


        else:

            # Select features in correct order
            input_data = input_data[
                feature_names
            ]


            # Make prediction
            prediction = model.predict(
                input_data
            )


            # Convert prediction to activity name
            predicted_activity = activity_map.get(
                int(prediction[0]),
                "Unknown Activity"
            )


            # Display result
            st.success(
                f"Predicted Activity: {predicted_activity}"
            )


            # Prediction probability
            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                confidence = np.max(
                    probabilities
                ) * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )


            # Display activity description
            descriptions = {
                "WALKING":
                    "The person is walking normally.",

                "WALKING_UPSTAIRS":
                    "The person is walking upstairs.",

                "WALKING_DOWNSTAIRS":
                    "The person is walking downstairs.",

                "SITTING":
                    "The person is sitting.",

                "STANDING":
                    "The person is standing.",

                "LAYING":
                    "The person is lying down."
            }


            st.write(
                descriptions.get(
                    predicted_activity,
                    "Activity description unavailable."
                )
            )


    except Exception as e:

        st.error(
            "An error occurred while processing the uploaded file."
        )

        st.exception(e)


else:

    st.warning(
        "Please upload a CSV file to make a prediction."
    )


# Project information
st.divider()

st.subheader("About This Project")

st.write(
    "This Human Activity Recognition project uses smartphone "
    "sensor data to classify six different physical activities. "
    "Three machine learning models were compared: Logistic "
    "Regression, Random Forest, and XGBoost. Logistic Regression "
    "achieved the best performance with approximately 96.13% "
    "accuracy and was selected as the final model."
)
