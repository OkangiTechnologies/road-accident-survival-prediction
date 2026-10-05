import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Accident Survival Prediction",
    page_icon="🚗",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
}

.card {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_accident.csv")


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("models/best_model.pkl")


df = load_data()
model = load_model()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚗 Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "🏠 Overview",
        "📊 Data Explorer",
        "🧠 Model Performance",
        "🔮 Live Prediction"
    ]
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<p class="main-title">'
    '🚗 Accident Survival Prediction System'
    '</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">'
    'Machine Learning Dashboard for Predicting Accident Survival'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "🏠 Overview":

    st.header("🏠 Project Overview")

    st.write("""
    This machine learning system predicts whether an accident
    victim is likely to survive based on factors such as age,
    gender, speed of impact, helmet usage, and seatbelt usage.
    """)

    # KPI CARDS

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Records",
            df.shape[0]
        )

    with col2:
        st.metric(
            "Features",
            df.shape[1] - 1
        )

    with col3:
        st.metric(
            "Target",
            "Survived"
        )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =========================================================
# DATA EXPLORER
# =========================================================

elif page == "📊 Data Explorer":

    st.header("📊 Data Explorer")

    st.write(
        "Explore the accident dataset."
    )

    # FILTER

    selected_gender = st.multiselect(
        "Select Gender",
        options=df["Gender"].unique(),
        default=df["Gender"].unique()
    )

    filtered_data = df[
        df["Gender"].isin(
            selected_gender
        )
    ]

    st.dataframe(
        filtered_data,
        use_container_width=True
    )

    st.divider()

    # CHARTS

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Survival Distribution"
        )

        survival_counts = (
            filtered_data["Survived"]
            .value_counts()
        )

        st.bar_chart(
            survival_counts
        )


    with col2:

        st.subheader(
            "Speed of Impact"
        )

        fig, ax = plt.subplots()

        ax.hist(
            filtered_data[
                "Speed_of_Impact"
            ],
            bins=20
        )

        ax.set_xlabel(
            "Speed"
        )

        ax.set_ylabel(
            "Frequency"
        )

        st.pyplot(fig)


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🧠 Model Performance":

    st.header(
        "🧠 Model Performance"
    )

    # LOAD MODEL RESULTS

    if os.path.exists(
        "results/model_results.csv"
    ):

        results = pd.read_csv(
            "results/model_results.csv"
        )

        st.subheader(
            "Model Comparison"
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        st.bar_chart(
            results.set_index(
                "Model"
            )
        )

    # LOAD METRICS

    if os.path.exists(
        "results/metrics.json"
    ):

        with open(
            "results/metrics.json",
            "r"
        ) as file:

            metrics = json.load(
                file
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Accuracy",
                f"{metrics['accuracy']:.2%}"
            )

        with col2:

            st.metric(
                "Precision",
                f"{metrics['precision']:.2%}"
            )

        with col3:

            st.metric(
                "Recall",
                f"{metrics['recall']:.2%}"
            )

    # CONFUSION MATRIX

    if os.path.exists(
        "results/confusion_matrix.png"
    ):

        st.subheader(
            "Confusion Matrix"
        )

        st.image(
            "results/confusion_matrix.png"
        )


# =========================================================
# LIVE PREDICTION
# =========================================================

elif page == "🔮 Live Prediction":

    st.header(
        "🔮 Live Accident Survival Prediction"
    )

    st.write(
        "Enter accident information below."
    )


    # TWO COLUMNS

    col1, col2 = st.columns(2)


    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        speed = st.number_input(
            "Speed of Impact",
            min_value=0.0,
            max_value=200.0,
            value=60.0
        )


    with col2:

        helmet = st.selectbox(
            "Helmet Used",
            ["Yes", "No"]
        )

        seatbelt = st.selectbox(
            "Seatbelt Used",
            ["Yes", "No"]
        )


    st.divider()


    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    if age <= 25:
        age_group = "Young"

    elif age <= 45:
        age_group = "Adult"

    elif age <= 60:
        age_group = "Middle_Aged"

    else:
        age_group = "Senior"


    if speed <= 40:
        speed_category = "Low"

    elif speed <= 80:
        speed_category = "Medium"

    else:
        speed_category = "High"


    safety_score = (
        int(helmet == "Yes")
        +
        int(seatbelt == "Yes")
    )


    # =====================================================
    # CREATE INPUT DATA
    # =====================================================

    input_data = pd.DataFrame(
        {
            "Age": [age],

            "Gender": [gender],

            "Speed_of_Impact": [speed],

            "Helmet_Used": [helmet],

            "Seatbelt_Used": [seatbelt],

            "Age_Group": [age_group],

            "Speed_Category": [
                speed_category
            ],

            "Safety_Score": [
                safety_score
            ]
        }
    )


    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button(
        "🚀 Predict Survival",
        use_container_width=True
    ):

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0]


        if prediction == 1:

            st.success(
                "🎉 Prediction: SURVIVED"
            )

        else:

            st.error(
                "⚠️ Prediction: DID NOT SURVIVE"
            )


        st.subheader(
            "Prediction Probability"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Survival Probability",
                f"{probability[1] * 100:.2f}%"
            )

        with col2:

            st.metric(
                "Non-Survival Probability",
                f"{probability[0] * 100:.2f}%"
            )