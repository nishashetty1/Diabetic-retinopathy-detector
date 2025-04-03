import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from model import RetinopathyModel
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

def verify_image_paths(csv_path, base_folder):
    """Verify if all images in the CSV exist in the specified folder structure"""
    df = pd.read_csv(csv_path)
    missing_images = []
    
    print(f"Verifying images in {base_folder}...")
    print(f"Current working directory: {os.getcwd()}")
    
    for idx, row in df.iterrows():
        # Look for the image in the category subfolder
        category = str(row['category'])
        image_path = os.path.join(base_folder, category, row['image_name'])
        
        if not os.path.exists(image_path):
            missing_images.append(f"{category}/{row['image_name']}")
    
    if missing_images:
        print(f"\nFound {len(missing_images)} missing images:")
        for img in missing_images[:10]:  # Show first 10 missing images
            print(f"- {img}")
        if len(missing_images) > 10:
            print(f"... and {len(missing_images) - 10} more")
        
        raise Exception("Missing images found. Please verify your dataset structure.")
    
    return True

def load_and_preprocess_data(csv_path, base_folder):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Verify image paths first
    verify_image_paths(csv_path, base_folder)
    
    # Initialize lists to store features and labels
    features = []
    labels = []
    
    # Create model instance for preprocessing
    model = RetinopathyModel()
    
    # Process each image
    print(f"Processing images from {base_folder}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Construct image path using category subfolder
            category = str(row['category'])
            image_path = os.path.join(base_folder, category, row['image_name'])
            
            # Extract features
            image_features = model.preprocess_image(image_path)
            
            features.append(image_features)
            labels.append(row['category'])
        except Exception as e:
            print(f"Error processing {category}/{row['image_name']}: {str(e)}")
            continue
    
    if not features:
        raise Exception("No images were successfully processed. Please check your dataset.")
    
    return np.array(features), np.array(labels)

def create_dataset_csv(base_folder, output_csv):
    """Create CSV file from directory structure if it doesn't exist"""
    if os.path.exists(output_csv):
        print(f"{output_csv} already exists, skipping creation.")
        return
    
    data = []
    for category in ['0', '1', '2', '3', '4']:
        category_path = os.path.join(base_folder, category)
        if os.path.exists(category_path):
            for image_name in os.listdir(category_path):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    data.append({
                        'image_name': image_name,
                        'category': int(category)
                    })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} images")

def plot_confusion_matrix(y_true, y_pred, save_path):
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def main():
    # Set paths
    train_csv = 'train_dataset.csv'
    test_csv = 'test_dataset.csv'
    train_folder = 'train'
    test_folder = 'test'
    
    # Create CSV files if they don't exist
    print("Checking/Creating dataset CSV files...")
    create_dataset_csv(train_folder, train_csv)
    create_dataset_csv(test_folder, test_csv)
    
    # Print directory structure
    print("\nCurrent directory structure:")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files[:5]:  # Show first 5 files in each directory
            print(f"{subindent}{f}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files)-5} more files")
    
    # Load and preprocess training data
    print("\nLoading training data...")
    X_train, y_train = load_and_preprocess_data(train_csv, train_folder)
    
    # Load and preprocess test data
    print("\nLoading test data...")
    X_test, y_test = load_and_preprocess_data(test_csv, test_folder)
    
    # Create and train the model
    print("\nTraining model...")
    model = RetinopathyModel(n_estimators=200, max_depth=20)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate and print metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot and save confusion matrix
    plot_confusion_matrix(y_test, y_pred, 'confusion_matrix.png')
    
    # Save the model
    print("Saving model...")
    model.save_model('retinopathy_model.joblib', 'retinopathy_scaler.joblib')
    
    # Calculate overall accuracy
    accuracy = (y_test == y_pred).mean()
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
