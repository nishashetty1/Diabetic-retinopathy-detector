import streamlit as st
# Must be the first Streamlit command
st.set_page_config(
    page_title="DR Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from model import DiabeticRetinopathyModel
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import logging
from models import download_model_files, verify_models, get_model_info, cleanup_invalid_models

def create_probability_chart(probabilities):
    """Create an interactive bar chart of probabilities using plotly"""
    df = pd.DataFrame({
        'Category': list(probabilities.keys()),
        'Probability': list(probabilities.values())
    })
    
    colors = px.colors.sequential.Viridis
    
    fig = px.bar(
        df,
        x='Category',
        y='Probability',
        color='Probability',
        color_continuous_scale=colors,
        text=df['Probability'].apply(lambda x: f'{x:.2%}')
    )
    
    fig.update_layout(
        title='Probability Distribution of DR Stages',
        xaxis_title='Diabetic Retinopathy Stage',
        yaxis_title='Probability',
        yaxis_range=[0, 1],
        showlegend=False
    )
    
    return fig

def get_stage_description(stage):
    """Get detailed description of DR stage"""
    descriptions = {
        'No DR': """
        No visible signs of diabetic retinopathy. Regular check-ups are still important 
        for monitoring any changes in your eye health.
        """,
        
        'Mild': """
        Early stage with small changes in the retina. Tiny swellings in the blood vessels, 
        called microaneurysms, may be present. Regular monitoring is essential.
        """,
        
        'Moderate': """
        More changes are visible, including:
        - Multiple microaneurysms
        - Bleeding in the retina
        - Cotton wool spots
        More frequent check-ups are recommended.
        """,
        
        'Severe': """
        Significant changes in the retina:
        - Widespread bleeding
        - Many cotton wool spots
        - Abnormal blood vessel growth
        Immediate consultation with an eye specialist is recommended.
        """,
        
        'Proliferative DR': """
        Most advanced stage:
        - New abnormal blood vessels grow
        - High risk of vision loss
        - May cause retinal detachment
        Requires immediate medical attention and treatment.
        """
    }
    return descriptions.get(stage, "Description not available")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Check for missing or invalid model files
    missing_files = verify_models()
    
    if missing_files:
        st.warning(f"Missing or invalid model files: {', '.join(missing_files)}")
        if st.button("Download Models"):
            cleanup_invalid_models()  # Clean up any partial downloads
            if download_model_files():
                st.success("Models downloaded successfully!")
                st.rerun()
            else:
                st.error("Failed to download model files. Please try again.")
                return
        return
    
    # Initialize model
    model = DiabeticRetinopathyModel()
    
    # Load trained models
    try:
        model.load_models(backup_dir='backup_training_results')
        model_loaded = True
        
        # Show model information in sidebar
        with st.sidebar:
            st.success("""
            🟢 Models loaded successfully!
            Using: backup_training_results/
            - CNN Model
            - RF Model
            """)
            
            # Display model details
            st.header("Model Information")
            model_info = get_model_info()
            for filename, info in model_info.items():
                st.info(f"""
                {filename}
                ────────────
                Status: {info['status']}
                Size: {info['size']}
                Last Modified: {info['last_modified']}
                """)
    except Exception as e:
        model_loaded = False
        with st.sidebar:
            st.error(f"❌ Error loading models: {str(e)}")
    
    # Title and introduction
    st.title("Diabetic Retinopathy Detection System")
    st.markdown("""
    This application helps detect different stages of Diabetic Retinopathy (DR) from retinal images.
    Upload a fundus photograph to get an analysis of DR severity.
    """)
    
    # Model status
    if not model_loaded:
        st.error("Error: Model not loaded. Please ensure model files are present in the correct location.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        Diabetic Retinopathy (DR) is a diabetes complication that affects eyes. 
        This tool uses AI to detect DR stages from eye fundus images.
        """)
        
        st.header("Image Guidelines")
        st.warning("""
        - Upload clear fundus photographs
        - Image should be centered on the macula
        - Ensure good lighting and focus
        - Supported formats: PNG, JPG, JPEG
        """)
        
        # Add timestamp
        st.markdown("---")
        st.caption(f"Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload fundus image", 
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Make prediction
            if st.button("Analyze Image"):
                with st.spinner("Analyzing image..."):
                    # Save temporary file for processing
                    temp_path = "temp_image.png"
                    image.save(temp_path)
                    
                    # Get prediction
                    try:
                        result = model.predict(temp_path)
                        
                        # Remove temporary file
                        os.remove(temp_path)
                        
                        with col2:
                            # Display results
                            st.header("Analysis Results")
                            
                            # Primary prediction
                            st.subheader("Predicted Stage")
                            st.markdown(f"### {result['class_name']}")
                            
                            # Confidence score
                            confidence = result['confidence'] * 100
                            st.metric(
                                "Confidence Score",
                                f"{confidence:.1f}%",
                                delta="high" if confidence > 80 else None
                            )
                            
                            # Probability chart
                            st.subheader("Probability Distribution")
                            fig = create_probability_chart(result['probabilities'])
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Stage description
                            st.subheader("Stage Description")
                            st.markdown(get_stage_description(result['class_name']))
                            
                            # Recommendations
                            st.subheader("Recommendations")
                            if result['prediction'] >= 2:
                                st.error("""
                                🚨 Please consult an eye specialist soon.
                                This analysis suggests significant DR progression.
                                """)
                            else:
                                st.success("""
                                ✅ Continue regular check-ups with your healthcare provider.
                                Monitor your blood sugar levels and maintain good eye health.
                                """)
                            
                            # Disclaimer
                            st.markdown("---")
                            st.caption("""
                            ⚠️ Disclaimer: This tool is for screening purposes only and should not 
                            replace professional medical advice. Always consult healthcare 
                            professionals for diagnosis and treatment.
                            """)
                    
                    except Exception as e:
                        st.error(f"Error analyzing image: {str(e)}")

if __name__ == "__main__":
    main()