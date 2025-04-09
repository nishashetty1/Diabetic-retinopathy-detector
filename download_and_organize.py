import os
import kaggle
import json
import shutil
from pathlib import Path
import pandas as pd
import zipfile
from tqdm import tqdm
import random

def setup_kaggle():
    """Setup Kaggle credentials"""
    try:
        username = os.getenv('KAGGLE_USERNAME')
        key = os.getenv('KAGGLE_KEY')
        
        if not username or not key:
            print("Error: Kaggle credentials not found!")
            return False
            
        kaggle_dir = os.path.expanduser('~/.kaggle')
        os.makedirs(kaggle_dir, exist_ok=True)
        
        kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
        with open(kaggle_json, 'w') as f:
            json.dump({
                "username": username,
                "key": key
            }, f)
        
        os.chmod(kaggle_json, 0o600)
        print("Kaggle credentials configured successfully!")
        return True
    except Exception as e:
        print(f"Error setting up Kaggle credentials: {str(e)}")
        return False

def download_and_organize():
    """Download APTOS dataset and organize a small subset"""
    try:
        # Create directories
        base_dir = 'small_dataset'
        train_dir = os.path.join(base_dir, 'train')
        temp_dir = os.path.join(base_dir, 'temp')
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
        
        print("Downloading APTOS 2019 dataset...")
        # Download the entire competition dataset
        kaggle.api.competition_download_files(
            'aptos2019-blindness-detection',
            path=temp_dir
        )
        
        print("\nExtracting files...")
        # Extract the zip file
        zip_path = os.path.join(temp_dir, 'aptos2019-blindness-detection.zip')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Read the training CSV
        df = pd.read_csv(os.path.join(temp_dir, 'train.csv'))
        
        # Create directories for each class
        for i in range(5):
            os.makedirs(os.path.join(train_dir, str(i)), exist_ok=True)
        
        print("\nOrganizing images...")
        # Process each class
        for diagnosis in range(5):
            print(f"\nProcessing class {diagnosis}...")
            
            # Get all images for this class
            class_df = df[df['diagnosis'] == diagnosis]
            
            # Select 100 random images if we have more than 100
            if len(class_df) > 100:
                class_df = class_df.sample(n=100, random_state=42)
            
            # Copy images
            for _, row in tqdm(class_df.iterrows(), total=len(class_df)):
                src = os.path.join(temp_dir, 'train_images', f"{row['id_code']}.png")
                dst = os.path.join(train_dir, str(diagnosis), f"{row['id_code']}.png")
                shutil.copy2(src, dst)
        
        # Clean up
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
        # Verify dataset
        print("\nVerifying final dataset...")
        total_images = 0
        for i in range(5):
            class_dir = os.path.join(train_dir, str(i))
            num_images = len(os.listdir(class_dir))
            print(f"Class {i}: {num_images} images")
            total_images += num_images
        
        print(f"\nTotal images: {total_images}")
        return True
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you've accepted the competition rules at:")
        print("   https://www.kaggle.com/c/aptos2019-blindness-detection")
        print("2. Verify your internet connection")
        print("3. Check if you have enough disk space")
        return False

def main():
    print("Starting APTOS dataset download and organization...")
    
    if not setup_kaggle():
        return
    
    if not download_and_organize():
        return
    
    print("\nDataset processing completed!")
    print("Location: ./small_dataset/train/")
    print("Structure:")
    print("- small_dataset/")
    print("  - train/")
    print("    - 0/ (100 images)")
    print("    - 1/ (100 images)")
    print("    - 2/ (100 images)")
    print("    - 3/ (100 images)")
    print("    - 4/ (100 images)")

if __name__ == "__main__":
    main()