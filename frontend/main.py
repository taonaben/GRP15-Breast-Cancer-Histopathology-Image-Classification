import streamlit as st
import os
from PIL import Image
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


predict_url = "https://breast-cancer-histopathology-image.onrender.com/predict/"


# Configure session with retries and longer timeout
def create_session_with_retries():
    """Create a requests session with automatic retries for slow/sleeping APIs"""
    session = requests.Session()

    # Configure retry strategy
    retry_strategy = Retry(
        total=5,  # Total number of retries
        backoff_factor=1,  # Wait 1, 2, 4, 8, 16 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["POST", "GET"],  # Retry on POST and GET requests
    )

    # Mount the retry strategy to both HTTP and HTTPS
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


# Create a session instance
session = create_session_with_retries()

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="DC",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #262730;
        border-radius: 10px;
        padding: 2rem;
        border: 2px dashed #4a4a5e;
    }
    
    [data-testid="stFileUploader"] > div {
        background-color: #262730;
    }
    
    /* Upload button */
    [data-testid="stFileUploader"] button {
        background-color: #0066ff !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 500 !important;
        transition: background-color 0.3s ease;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background-color: #0052cc !important;
    }
    
    /* Upload text */
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-size: 1rem !important;
    }
    
    /* Drag and drop text */
    .uploadMessage {
        color: #b4b4b4 !important;
    }
    
    /* Title styling */
    h1 {
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 0.5rem !important;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #b4b4b4;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    
    /* Results container */
    .results-container {
        background-color: #262730;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        background-color: #0066ff;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #0052cc;
    }
    </style>
""",
    unsafe_allow_html=True,
)


st.title("🔬 Breast Cancer Histopathology Image Classification")
st.markdown(
    '<p class="subtitle">Upload a histopathology image for classification</p>',
    unsafe_allow_html=True,
)


col1, col2, col3 = st.columns([1, 3, 1])

with col2:

    uploaded_file = st.file_uploader(
        "Upload a file",
        type=["png", "jpg", "jpeg", "tif", "tiff", "HEIC"],
        help="Drag and drop your files here or click to upload",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div style='text-align: center; color: #b4b4b4; margin-top: 1rem;'>
                No items selected
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file is not None:

        st.markdown("### Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

        # Classification button
        if st.button("Classify Image", type="primary"):
            with st.spinner(
                "Classifying... (This may take a minute if the API is waking up)"
            ):
                try:
                    # Call FastAPI backend with retries and longer timeout
                    uploaded_file.seek(0)
                    files = {
                        "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
                    }

                    # Attempt with retry logic and 120-second timeout
                    # The timeout allows the free Render service time to wake up
                    response = session.post(
                        predict_url,
                        files=files,
                        timeout=120,  # 2 minutes timeout for slow/sleeping APIs
                    )

                    if response.status_code == 200:
                        result = response.json()

                        st.markdown(
                            '<div class="results-container">', unsafe_allow_html=True
                        )
                        st.success("✅ Classification Complete!")

                        # Display results
                        st.markdown("#### Prediction Results")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(
                                "Classification",
                                result["classification"],
                                delta=f"{result['confidence']}% confidence",
                            )
                        with col_b:
                            st.metric(
                                "Processing Time", f"{result['processing_time']}s"
                            )

                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(
                            f"❌ API Error: {response.status_code} - {response.json().get('detail', 'Unknown error')}"
                        )

                except requests.exceptions.Timeout:
                    st.error(
                        "⏱️ **Request Timeout**: The API took too long to respond. "
                        "This may happen if the free Render service is sleeping. "
                        "Please wait a moment and try again. The backend should be warmed up by then."
                    )
                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ **Connection Error**: Cannot connect to the API server. "
                        "The backend service might be starting up. Please try again in a few moments."
                    )
                except requests.exceptions.RequestException as e:
                    st.error(f"❌ **Request Failed**: {str(e)}")
                except Exception as e:
                    st.error(f"❌ **Unexpected Error**: {str(e)}")

        if st.button("Upload Another Image"):
            st.rerun()

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6c6c6c; font-size: 0.875rem;'>
        Breast Cancer Histopathology Image Classification System
    </div>
    """,
    unsafe_allow_html=True,
)
