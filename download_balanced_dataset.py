import os
import kaggle
import random
import shutil
from pathlib import Path

# Create directories for the balanced dataset
def create_directories():
    base_dir = Path('balanced_dataset')
    for split in ['train', 'test']:
        for category in range(5):
            (base_dir / split / str(category)).mkdir(parents=True, exist_ok=True)
    return base_dir

# Download the dataset from Kaggle
def download_dataset():
    dataset_name = "ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"
    kaggle.api.dataset_download_files(dataset_name, unzip=True)

# Sample and copy files
def balance_dataset(base_dir):
    SAMPLES_PER_CATEGORY = 7000
    
    # Process train folder
    source_train_dir = Path('augmented_resized_V2/train')
    for category in range(5):
        source_category_dir = source_train_dir / str(category)
        target_category_dir = base_dir / 'train' / str(category)
        
        # List all files in the category
        all_files = list(source_category_dir.glob('*.jpg'))
        
        # Randomly sample files
        selected_files = random.sample(all_files, min(SAMPLES_PER_CATEGORY, len(all_files)))
        
        # Copy selected files
        for file in selected_files:
            shutil.copy2(file, target_category_dir)
        
        print(f"Category {category}: Copied {len(selected_files)} files")

    # Copy test folder entirely
    source_test_dir = Path('augmented_resized_V2/test')
    if source_test_dir.exists():
        for category in range(5):
            source_category_dir = source_test_dir / str(category)
            target_category_dir = base_dir / 'test' / str(category)
            
            if source_category_dir.exists():
                for file in source_category_dir.glob('*.jpg'):
                    shutil.copy2(file, target_category_dir)

def main():
    print("Creating directories...")
    base_dir = create_directories()
    
    print("Downloading dataset from Kaggle...")
    download_dataset()
    
    print("Balancing dataset...")
    balance_dataset(base_dir)
    
    print("Dataset processing completed!")

if __name__ == "__main__":
    main()