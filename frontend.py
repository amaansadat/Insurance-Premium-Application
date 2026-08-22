import streamlit as st
import requests
import time
import pandas as pd

st.set_page_config(
    page_title="Insurance Premium Predictor",
    page_icon="💊",
    layout="centered"
)

API_URL = "https://insurance-premium-api-befz.onrender.com/predict"

st.title("💊 Insurance Premium Predictor")

st.markdown(
    "Predict the insurance premium category using a Machine Learning model based on demographic, lifestyle, and financial information."
)

st.divider()

# Input fields
age = st.number_input("Age", min_value=1, max_value=119, value=30)
weight = st.number_input("Weight (kg)", min_value=1.0, value=65.0)
height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.7)
income_lpa = st.number_input("Annual Income (LPA)", min_value=0.1, value=10.0)
smoker = st.selectbox("Are you a smoker?", options=[True, False])
city = st.selectbox(
    "City",
    [
        "Mumbai",
        "Delhi",
        "Bangalore",
        "Hyderabad",
        "Chennai",
        "Pune",
        "Lucknow",
        "Kolkata"
    ]
)
occupation = st.selectbox(
    "Occupation",
    ['retired', 'freelancer', 'student', 'government_job',
     'business_owner', 'unemployed', 'private_job']
)

# Button click
if st.button("🔍 Predict Premium Category", use_container_width=True):

    input_data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city,
        "occupation": occupation
    }

    response = None

    with st.spinner("⏳ Connecting to backend... Please wait."):

        for attempt in range(8):
            try:
                response = requests.post(
                    API_URL,
                    json=input_data,
                    timeout=30
                )

                if response.status_code == 200:
                    break

            except requests.exceptions.RequestException:
                response = None

            time.sleep(5)

    if response is None:
        st.error("❌ Unable to contact the backend service.")
        st.info(
        "If this is the first request, the Render Free Tier backend may be waking up. "
        "Please wait about 30–60 seconds and try again."
    )
        st.stop()


    if response.status_code != 200:
        st.warning("⚠️ Backend is currently starting.")

        st.info(
        "The backend is deployed on the Render Free Tier. "
        "After being inactive, it may take 30–60 seconds to wake up. "
        "Please wait a moment and try again."
    )

        st.stop()

    # Parse result
    try:
        result = response.json()
        prediction = result["response"]

        st.success(
            f"Predicted Premium Category: {prediction['predicted_category']}"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Confidence",
                f"{prediction['confidence']*100:.1f}%"
            )

        with col2:
            st.metric(
                "Category",
                prediction["predicted_category"]
            )

        st.progress(prediction["confidence"])

        st.caption(
                f"Prediction Confidence: {prediction['confidence']*100:.1f}%"
            )


        st.subheader("📊 Class Probabilities")

        df = pd.DataFrame(
            prediction["class_probabilities"].items(),
            columns=["Premium Category", "Probability"]
)

        df["Probability"] = df["Probability"].apply(
            lambda x: f"{x * 100:.2f}%"
)

        st.table(df)

    except Exception as e:
        st.error(f"Error parsing response: {e}")

st.divider()

st.table(df)

st.markdown(
    """
    <hr style="
        margin-top: 10px;
        margin-bottom: 8px;
        border: none;
        border-top: 1px solid #d9d9d9;
    ">
    """,
    unsafe_allow_html=True
)

st.caption(
    "Built using Scikit-Learn • FastAPI • Docker • Render • Streamlit"
)