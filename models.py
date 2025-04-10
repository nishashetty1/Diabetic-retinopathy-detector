import os
import requests
import logging
from datetime import datetime
import streamlit as st
import re
from urllib.parse import urlparse, parse_qs

# Model file mappings with new SharePoint links
MODEL_FILES = {
    'best_model.keras': "https://viteduin59337-my.sharepoint.com/:u:/g/personal/nisha_shetty_vit_edu_in/EU7NhSvAf4RBpq_JUQJDY1gBzwrb7ZMG33GUKIzw5pG5yA?e=nOfp3X",
    'cnn_model.keras': "https://viteduin59337-my.sharepoint.com/:u:/g/personal/nisha_shetty_vit_edu_in/ETMlf-w7o9JMisBGYXUyh5sB03fZAR1xb1QgM5v6xkTKmQ?e=kDIFu0",
    'rf_model.joblib': "https://viteduin59337-my.sharepoint.com/:u:/g/personal/nisha_shetty_vit_edu_in/EUrkC4sSr_tErxDBmeQUnIwBJwmq3DePO3aJ-3sK9ZaqoQ?e=bPQWcF"
}

def get_sharepoint_download_url(sharing_url):
    """Convert SharePoint sharing URL to direct download URL"""
    try:
        # Parse the URL
        parsed = urlparse(sharing_url)
        
        # Remove the 'e' parameter and add download=1
        base_url = sharing_url.split('?')[0]
        return f"{base_url}?download=1"
    except Exception as e:
        logging.error(f"Error converting SharePoint URL: {str(e)}")
        return sharing_url

def download_file_from_sharepoint(url, filepath):
    """Download a file from SharePoint with authentication handling"""
    try:
        # Configure download request with specific headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cookie': ''  # SharePoint might set cookies during redirect
        }
        
        # Get direct download URL
        download_url = get_sharepoint_download_url(url)
        
        # Create a session to handle redirects
        session = requests.Session()
        
        # First request to handle authentication
        response = session.get(download_url, headers=headers, allow_redirects=True, stream=True, timeout=60)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if 'html' in content_type.lower():
            raise Exception("Received HTML instead of file data. Authentication might be required.")
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size < 1024 * 1024:  # Less than 1MB
            raise Exception("File size too small, might be an error page")
        
        # Create progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Download with progress
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    if total_size:
                        progress = min(downloaded / total_size, 1.0)
                        progress_bar.progress(progress)
                        progress_text.text(f"Downloading {filepath}... {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
        
        # Verify downloaded file
        if os.path.getsize(filepath) < 1024 * 1024:  # Less than 1MB
            os.remove(filepath)
            raise Exception("Downloaded file is too small")
        
        progress_text.empty()
        progress_bar.empty()
        
        return True
    
    except requests.exceptions.RequestException as e:
        st.error(f"Network error downloading {os.path.basename(filepath)}: {str(e)}")
        logging.error(f"SharePoint download error: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False
    except Exception as e:
        st.error(f"Error downloading {os.path.basename(filepath)}: {str(e)}")
        logging.error(f"Download error: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def download_model_files():
    """Download model files from SharePoint if they don't exist locally"""
    os.makedirs('backup_training_results', exist_ok=True)
    
    try:
        total_files = len(MODEL_FILES)
        success_count = 0
        
        for idx, (filename, url) in enumerate(MODEL_FILES.items(), 1):
            filepath = os.path.join('backup_training_results', filename)
            
            # Remove existing file if it's invalid
            if os.path.exists(filepath) and os.path.getsize(filepath) < 1024 * 1024:
                os.remove(filepath)
            
            if not os.path.exists(filepath):
                st.info(f"Downloading {filename} ({idx}/{total_files})...")
                
                if download_file_from_sharepoint(url, filepath):
                    success_count += 1
                    st.success(f"Successfully downloaded {filename}")
                else:
                    st.error(f"Failed to download {filename}")
                    return False
            else:
                success_count += 1
                st.success(f"{filename} already exists")
        
        if success_count == total_files:
            st.success("✅ All model files downloaded successfully!")
            return True
        return False
        
    except Exception as e:
        st.error(f"Error downloading models: {str(e)}")
        logging.error(f"Download error: {str(e)}")
        return False

def verify_models():
    """Verify that all required model files exist and have correct sizes"""
    missing_files = []
    for filename in MODEL_FILES.keys():
        filepath = os.path.join('backup_training_results', filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
        else:
            # Check file size is reasonable (>1MB)
            if os.path.getsize(filepath) < 1024 * 1024:
                missing_files.append(filename)  # Include if file is suspiciously small
    return missing_files

def get_model_info():
    """Get information about the model files"""
    info = {}
    for filename in MODEL_FILES.keys():
        filepath = os.path.join('backup_training_results', filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            modified = datetime.fromtimestamp(os.path.getmtime(filepath))
            info[filename] = {
                'size': f"{size_mb:.2f} MB",
                'last_modified': modified.strftime('%Y-%m-%d %H:%M:%S'),
                'status': '✓ Loaded' if size_mb > 1 else '⚠️ Invalid file'
            }
        else:
            info[filename] = {
                'size': 'N/A',
                'last_modified': 'Not downloaded',
                'status': '❌ Missing'
            }
    return info

def cleanup_invalid_models():
    """Clean up any invalid or partially downloaded model files"""
    for filename in MODEL_FILES.keys():
        filepath = os.path.join('backup_training_results', filename)
        if os.path.exists(filepath):
            if os.path.getsize(filepath) < 1024 * 1024:  # Less than 1MB
                try:
                    os.remove(filepath)
                    logging.info(f"Removed invalid model file: {filepath}")
                except Exception as e:
                    logging.error(f"Error removing invalid file {filepath}: {str(e)}")