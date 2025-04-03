import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
import joblib
from model import RetinopathyModel
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="DR Detection System",
    page_icon="👁️",
    layout="wide"
)

# Clean, simple CSS
st.markdown("""
    <style>
    /* Clean fonts and spacing */
    .main {
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .header {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .timestamp {
        font-size: 14px;
        color: #666;
        margin-bottom: 10px;
    }
    
    .prediction-box {
        padding: 15px;
        border-radius: 5px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .probability-bar {
        margin-bottom: 15px;
    }
    
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        font-size: 14px;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        # Get the repository root directory
        repo_root = os.path.dirname(os.path.abspath(__file__))
        
        # Define paths to model files
        model_path = os.path.join(repo_root, 'retinopathy_model.joblib')
        scaler_path = os.path.join(repo_root, 'retinopathy_scaler.joblib')
        
        # Debug information
        st.write("Looking for model files in:", repo_root)
        st.write("Available files:", os.listdir(repo_root))
        
        # Check if both files exist
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            # Try alternative path
            alt_model_path = 'retinopathy_model.joblib'
            if os.path.exists(alt_model_path):
                model_path = alt_model_path
                st.success(f"Found model at alternate path: {alt_model_path}")
            else:
                return None
                
        if not os.path.exists(scaler_path):
            st.error(f"Scaler file not found at: {scaler_path}")
            # Try alternative path
            alt_scaler_path = 'retinopathy_scaler.joblib'
            if os.path.exists(alt_scaler_path):
                scaler_path = alt_scaler_path
                st.success(f"Found scaler at alternate path: {alt_scaler_path}")
            else:
                return None
            
        # Load the model with both files
        model = RetinopathyModel.load_model(model_path, scaler_path)
        st.success("Model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"""
        Error loading model:
        Error type: {type(e).__name__}
        Error message: {str(e)}
        
        Current working directory: {os.getcwd()}
        Directory contents: {os.listdir('.')}
        """)
        return None

def get_prediction_label(prediction):
    labels = {
        0: "No Diabetic Retinopathy",
        1: "Mild Diabetic Retinopathy",
        2: "Moderate Diabetic Retinopathy",
        3: "Severe Diabetic Retinopathy",
        4: "Proliferative Diabetic Retinopathy"
    }
    return labels.get(prediction, "Unknown")

def get_severity_color(prediction):
    colors = {
        0: "#28a745",  # Green
        1: "#ffc107",  # Yellow
        2: "#fd7e14",  # Orange
        3: "#dc3545",  # Red
        4: "#721c24"   # Dark Red
    }
    return colors.get(prediction, "#6c757d")

def preprocess_uploaded_image(uploaded_file):
    image_bytes = uploaded_file.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def main():
    # Current time and user display
    current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    st.markdown(f"""
    <div class="header">
        <div class="timestamp">
            Current Time (UTC): {current_time}<br>
            User: nishashetty1
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Simple title
    st.title("Diabetic Retinopathy Detection System")

    # File upload
    uploaded_file = st.file_uploader("Upload a fundus image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Create two columns
            col1, col2 = st.columns(2)

            with col1:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded fundus image")

                # Process image
                uploaded_file.seek(0)
                processed_image = preprocess_uploaded_image(uploaded_file)
                
                # Load model and get prediction
                model = load_model()
                if model is None:
                    st.error("Failed to load model")
                    return
                
                processed_features = model.preprocess_image(processed_image)
                prediction = model.predict([processed_features])[0]
                prediction_proba = model.predict_proba([processed_features])[0]

            with col2:
                # Display prediction
                prediction_label = get_prediction_label(prediction)
                severity_color = get_severity_color(prediction)
                
                st.markdown(f"""
                <div class="prediction-box" style="background-color: {severity_color};">
                    <h2 style="margin: 0;">Detection Result</h2>
                    <h3 style="margin: 10px 0;">{prediction_label}</h3>
                </div>
                """, unsafe_allow_html=True)

                # Probability Distribution
                st.subheader("Probability Distribution")
                
                for i, prob in enumerate(prediction_proba):
                    label = get_prediction_label(i)
                    prob_percentage = prob * 100
                    st.markdown(f"""
                    <div class="probability-bar">
                        <div style="font-size: 14px; margin-bottom: 5px;">{label}</div>
                        <div style="background-color: #f0f2f6; border-radius: 5px; height: 25px;">
                            <div style="background-color: {get_severity_color(i)}; 
                                      width: {prob_percentage}%; 
                                      height: 100%; 
                                      border-radius: 5px; 
                                      text-align: right; 
                                      padding: 0 10px; 
                                      color: white; 
                                      line-height: 25px;">
                                {prob_percentage:.1f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Simple disclaimer
                st.markdown("""
                <div class="disclaimer">
                    ⚠️ This is a screening tool (56.38% accuracy). 
                    Please consult a healthcare professional for diagnosis.
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error("Error processing image. Please upload a clear fundus image.")

if __name__ == "__main__":
    main()
