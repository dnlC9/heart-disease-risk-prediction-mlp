# app.py
import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on the path so we can import inference/ modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports (files live in the same folder)
from data_loader import load_heart_data
import page_1 as page_dataset_overview
import page_2 as page_model_training
import page_3 as page_data_viz
import page_4 as page_prediction


st.set_page_config(
    page_title="Heart Disease Neural Network – Analytics & Prediction",
    page_icon="🫀",
    layout="wide",
)


def init_session_state():
    """Initialize global session state (dataframe, filtered data, model cache flags, etc.)."""
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "df_fil" not in st.session_state:
        st.session_state["df_fil"] = None


def main():
    st.title("🫀 Heart Disease Neural Network Dashboard")

    init_session_state()

    # --- Load data once and store in session_state ---
    if st.session_state["df"] is None:
        with st.spinner("Loading and cleaning dataset..."):
            df = load_heart_data()
            st.session_state["df"] = df

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        (
            "1️⃣ Dataset Overview",
            "2️⃣ Model Training & Validation",
            "3️⃣ Data Exploration",
            "4️⃣ Prediction & What-If Analysis",
        ),
    )

    if page.startswith("1"):
        page_dataset_overview.show()
    elif page.startswith("2"):
        page_model_training.show()
    elif page.startswith("3"):
        page_data_viz.show()
    elif page.startswith("4"):
        page_prediction.show()


if __name__ == "__main__":
    main()
