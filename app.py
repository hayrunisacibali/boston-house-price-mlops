import streamlit as st
import pandas as pd
import mlflow

from mlflow import MlflowClient

from src.prediction.storage import (
    save_prediction,
    get_cycle_prediction_count,
)

from src.monitoring.automatic_monitoring import (
    run_automatic_monitoring,
)


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Boston House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>

    /* ==================================================
       GLOBAL BACKGROUND
       ================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 50% 15%,
                rgba(30, 64, 175, 0.28) 0%,
                rgba(8, 15, 30, 0.65) 35%,
                rgba(5, 10, 20, 1) 75%
            ),
            linear-gradient(
                135deg,
                #0B0F17 0%,
                #0D1524 50%,
                #080C14 100%
            );

        color: #f8fafc;
    }


    /* ==================================================
       MAIN CONTAINER
       ================================================== */

    .block-container {
        max-width: 900px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
    }


    /* ==================================================
       MAIN TITLE
       ================================================== */

    .main-title {
        text-align: center;

        font-size: 42px;
        font-weight: 700;

        letter-spacing: -1px;

        color: #f8fafc;

        margin-bottom: 10px;

        text-shadow:
            0 0 25px rgba(56, 189, 248, 0.08);
    }


    /* ==================================================
       SUBTITLE
       ================================================== */

    .subtitle {
        text-align: center;

        color: #94a3b8;

        font-size: 16px;

        margin-bottom: 42px;
    }


    /* ==================================================
       SECTION TITLE
       ================================================== */

    .section-title {
        color: #e2e8f0;

        font-size: 21px;
        font-weight: 600;

        margin-bottom: 22px;

        padding-bottom: 12px;

        border-bottom:
            1px solid rgba(255, 255, 255, 0.08);
    }


    /* ==================================================
       INPUT LABELS
       ================================================== */

    label {
        color: #ffffff !important;

        font-weight: 600 !important;

        font-size: 14px !important;
    }


    /* ==================================================
       INPUT OUTER CONTAINER
       ================================================== */

    div[data-baseweb="input"] {
        background: #0d1117 !important;

        border:
            1px solid rgba(56, 189, 248, 0.30) !important;

        border-radius: 12px !important;

        box-shadow:
            0 6px 20px rgba(0, 0, 0, 0.25);

        transition:
            border-color 0.25s ease,
            box-shadow 0.25s ease;
    }


    /* ==================================================
       INPUT INNER CONTAINER
       ================================================== */

    div[data-baseweb="input"] > div {
        background: #0d1117 !important;
    }


    /* ==================================================
       INPUT ELEMENT
       ================================================== */

    div[data-baseweb="input"] input {
        background: #0d1117 !important;

        color: #ffffff !important;

        -webkit-text-fill-color: #ffffff !important;

        caret-color: #ffffff !important;

        font-size: 15px !important;
    }


    /* ==================================================
       INPUT FOCUS
       ================================================== */

    div[data-baseweb="input"]:focus-within {
        border:
            1px solid rgba(56, 189, 248, 0.80) !important;

        box-shadow:
            0 0 12px rgba(56, 189, 248, 0.50) !important;
    }


    /* ==================================================
       INPUT HOVER
       ================================================== */

    div[data-baseweb="input"]:hover {
        border:
            1px solid rgba(56, 189, 248, 0.50) !important;
    }


    /* ==================================================
       NUMBER INPUT + / - BUTTONS
       ================================================== */

    div[data-baseweb="input"] button {
        background: #0d1117 !important;

        color: #ffffff !important;

        border: none !important;
    }


    div[data-baseweb="input"] button svg {
        fill: #ffffff !important;

        color: #ffffff !important;
    }


    div[data-baseweb="input"] button:hover {
        color: #38bdf8 !important;

        background: #111827 !important;
    }


    div[data-baseweb="input"] button:hover svg {
        fill: #38bdf8 !important;

        color: #38bdf8 !important;
    }


    /* ==================================================
       PREDICTION BUTTON
       ================================================== */

    .stButton > button {
        width: 100%;

        height: 54px;

        border-radius: 12px;

        border:
            1px solid rgba(125, 211, 252, 0.35);

        background:
            linear-gradient(
                135deg,
                #1e3a5f 0%,
                #2563a6 50%,
                #1e6b8f 100%
            );

        color: #ffffff;

        font-size: 16px;

        font-weight: 600;

        letter-spacing: 0.2px;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.30);

        transition:
            all 0.25s ease;
    }


    /* ==================================================
       BUTTON HOVER
       ================================================== */

    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #2563a6 0%,
                #0ea5c7 50%,
                #2563a6 100%
            );

        border:
            1px solid rgba(125, 211, 252, 0.65);

        box-shadow:
            0 0 15px rgba(56, 189, 248, 0.40),
            0 12px 35px rgba(0, 0, 0, 0.35);

        transform: translateY(-1px);
    }


    /* ==================================================
       PREDICTION RESULT BOX
       ================================================== */

    div[data-testid="stAlert"] {
        background:
            rgba(15, 23, 42, 0.85) !important;

        border:
            1px solid rgba(74, 222, 128, 0.40) !important;

        border-radius: 12px !important;

        color: #4ade80 !important;

        margin-top: 28px;

        box-shadow:
            0 0 15px rgba(74, 222, 128, 0.12),
            0 8px 30px rgba(0, 0, 0, 0.25);
    }


    /* Result text */

    div[data-testid="stAlert"] p {
        color: #4ade80 !important;

        font-weight: 600 !important;
    }


    /* ==================================================
       COLUMNS
       ================================================== */

    div[data-testid="column"] {
        padding-left: 8px;
        padding-right: 8px;
    }


    /* ==================================================
       HEADER
       ================================================== */

    header[data-testid="stHeader"] {
        background: transparent;
    }


    /* ==================================================
       MOBILE
       ================================================== */

    @media (max-width: 768px) {

        .main-title {
            font-size: 32px;
        }

        .subtitle {
            font-size: 14px;
        }

        .block-container {
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# MLflow Configuration
# --------------------------------------------------

TRACKING_URI = (
    "sqlite:///mlflow.db"
)

MODEL_NAME = "BostonHousePriceModel"

CHAMPION_ALIAS = "champion"

mlflow.set_tracking_uri(TRACKING_URI)

client = MlflowClient()


# --------------------------------------------------
# Get Current Champion Version Dynamically
# --------------------------------------------------

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    CHAMPION_ALIAS
)

current_model_version = str(
    champion.version
)


# --------------------------------------------------
# Load Current Champion Model
# --------------------------------------------------

MODEL_URI = (
    f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"
)

model = mlflow.pyfunc.load_model(
    MODEL_URI
)


# --------------------------------------------------
# Get Current Cycle Prediction Count
# --------------------------------------------------

prediction_count = get_cycle_prediction_count()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="main-title">'
    '🏠 Boston House Price Prediction'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter the house characteristics below to '
    'predict the median house value.'
    '</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Current Champion Status
# --------------------------------------------------

st.markdown(
    f"""
    <div class="model-status-title">
        Current Champion Model
    </div>

    <div class="model-status-value">
        Version {current_model_version}
    </div>

    <div class="model-status-count">
        Predictions in current monitoring cycle:<br>
        <strong>{prediction_count}/50</strong>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.markdown(
    '<div class="section-title">'
    'House Characteristics'
    '</div>',
    unsafe_allow_html=True
)


col1, col2 = st.columns(2)


# --------------------------------------------------
# RM
# --------------------------------------------------

with col1:

    rm = st.number_input(
        "RM - Average number of rooms",
        min_value=0.0,
        max_value=10.0,
        value=6.208,
        step=0.1,
        format="%.3f"
    )


# --------------------------------------------------
# LSTAT
# --------------------------------------------------

with col2:

    lstat = st.number_input(
        "LSTAT - Lower status population (%)",
        min_value=0.0,
        max_value=40.0,
        value=11.36,
        step=0.1,
        format="%.2f"
    )


# --------------------------------------------------
# DIS
# --------------------------------------------------

with col1:

    dis = st.number_input(
        "DIS - Distance to employment centers",
        min_value=0.0,
        max_value=15.0,
        value=3.2075,
        step=0.1,
        format="%.4f"
    )


# --------------------------------------------------
# NOX
# --------------------------------------------------

with col2:

    nox = st.number_input(
        "NOX - Nitric oxide concentration",
        min_value=0.0,
        max_value=1.0,
        value=0.538,
        step=0.01,
        format="%.3f"
    )


# --------------------------------------------------
# PTRATIO
# --------------------------------------------------

with col1:

    ptratio = st.number_input(
        "PTRATIO - Pupil-teacher ratio",
        min_value=0.0,
        max_value=25.0,
        value=19.1,
        step=0.1,
        format="%.1f"
    )


# --------------------------------------------------
# CRIM
# --------------------------------------------------

with col2:

    crim = st.number_input(
        "CRIM - Crime rate",
        min_value=0.0,
        max_value=100.0,
        value=0.25651,
        step=0.1,
        format="%.5f"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.write("")


if st.button(
    "Predict House Price",
    use_container_width=True
):

    # --------------------------------------------------
    # Prepare Input
    # --------------------------------------------------

    input_data = pd.DataFrame([{
        "RM": rm,
        "LSTAT": lstat,
        "DIS": dis,
        "NOX": nox,
        "PTRATIO": ptratio,
        "CRIM": crim
    }])


    # --------------------------------------------------
    # Make Prediction
    # --------------------------------------------------

    prediction = model.predict(
        input_data
    )[0]


    # --------------------------------------------------
    # Save Production Prediction
    # --------------------------------------------------

    save_prediction(
        rm=rm,
        lstat=lstat,
        dis=dis,
        nox=nox,
        ptratio=ptratio,
        crim=crim,
        prediction=prediction,
        model_alias=CHAMPION_ALIAS,
        model_version=current_model_version,
    )


    # --------------------------------------------------
    # Automatic Monitoring
    # --------------------------------------------------

    monitoring_result = (
        run_automatic_monitoring()
    )


    # --------------------------------------------------
    # Prediction Result
    # --------------------------------------------------

    st.success(
        f"Predicted House Price: "
        f"${prediction:.2f}k"
    )


    # --------------------------------------------------
    # Pipeline Status
    # --------------------------------------------------

    if monitoring_result is None:

        # Current cycle has not reached 50 yet.

        prediction_count = (
            get_cycle_prediction_count()
        )

        st.info(
            "Production predictions collected: "
            f"{prediction_count}/50"
        )


    elif monitoring_result["status"] == "no_drift":

        # --------------------------------------------------
        # No Drift
        # --------------------------------------------------

        st.success(
            "50 predictions completed. "
            "Monitoring detected no drift. "
            "A new prediction cycle has started."
        )


    elif monitoring_result["status"] == "drift":

        # --------------------------------------------------
        # Drift Detected
        # --------------------------------------------------

        st.warning(
            "50 predictions completed. "
            "Data drift was detected. "
            "Candidate retraining will be triggered."
        )