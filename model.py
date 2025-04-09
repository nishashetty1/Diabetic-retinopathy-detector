import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import joblib

class DiabeticRetinopathyModel:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.cnn_model = None
        self.rf_classifier = None
        self.class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    
    def preprocess_image(self, image_path):
        """Preprocess single image"""
        try:
            img = Image.open(image_path)
            img = img.resize((self.img_size, self.img_size))
            img = np.array(img) / 255.0
            if len(img.shape) == 2:  # Convert grayscale to RGB
                img = np.stack([img] * 3, axis=-1)
            return img
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    def load_data(self, base_path):
        """Load and preprocess all images"""
        X = []
        y = []
        
        for class_idx in range(5):
            class_path = os.path.join(base_path, str(class_idx))
            print(f"Loading class {class_idx} from {class_path}")
            
            # Get all image files in the directory
            image_files = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                img = self.preprocess_image(img_path)
                
                if img is not None:
                    X.append(img)
                    y.append(class_idx)
        
        return np.array(X), np.array(y)

    def create_cnn_model(self):
        """Create CNN model using EfficientNetB3 as base"""
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        predictions = Dense(128, activation='relu')(x)
        
        self.cnn_model = Model(inputs=base_model.input, outputs=predictions)
        return self.cnn_model

    def train_models(self, train_path, val_split=0.2):
        """Train both CNN and Random Forest models"""
        print("Loading and preprocessing data...")
        X, y = self.load_data(train_path)
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )
        
        # Create and train CNN
        print("Training CNN model...")
        self.create_cnn_model()
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Train CNN
        self.cnn_model.compile(
            optimizer='adam',
            loss='mse'
        )
        
        self.cnn_model.fit(
            datagen.flow(X_train, X_train, batch_size=32),
            epochs=10,
            validation_data=(X_val, X_val)
        )
        
        # Extract features using CNN
        print("Extracting features using CNN...")
        train_features = self.cnn_model.predict(X_train)
        val_features = self.cnn_model.predict(X_val)
        
        # Train Random Forest
        print("Training Random Forest model...")
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
        
        self.rf_classifier.fit(train_features, y_train)
        
        # Evaluate
        val_predictions = self.rf_classifier.predict(val_features)
        accuracy = accuracy_score(y_val, val_predictions)
        print("\nValidation Accuracy:", accuracy)
        print("\nClassification Report:")
        print(classification_report(y_val, val_predictions, target_names=self.class_names))
        
        return accuracy

    def save_models(self, cnn_path='cnn_model.h5', rf_path='rf_model.joblib'):
        """Save both models"""
        if self.cnn_model is not None:
            self.cnn_model.save(cnn_path)
        if self.rf_classifier is not None:
            joblib.dump(self.rf_classifier, rf_path)
        print("Models saved successfully!")

    def load_models(self, cnn_path='cnn_model.h5', rf_path='rf_model.joblib'):
        """Load saved models"""
        if os.path.exists(cnn_path):
            self.cnn_model = tf.keras.models.load_model(cnn_path)
        if os.path.exists(rf_path):
            self.rf_classifier = joblib.load(rf_path)
        print("Models loaded successfully!")

    def predict(self, image_path):
        """Predict DR grade for a single image"""
        if self.cnn_model is None or self.rf_classifier is None:
            raise ValueError("Models not trained or loaded!")
        
        # Preprocess image
        img = self.preprocess_image(image_path)
        if img is None:
            return None
        
        # Extract features using CNN
        features = self.cnn_model.predict(np.array([img]))
        
        # Predict using Random Forest
        probabilities = self.rf_classifier.predict_proba(features)
        prediction = self.rf_classifier.predict(features)[0]
        
        return {
            'prediction': prediction,
            'class_name': self.class_names[prediction],
            'probabilities': {
                self.class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities[0])
            }
        }

def train_and_save_models():
    """Utility function to train and save models"""
    model = DiabeticRetinopathyModel()
    
    # Train models using the organized dataset
    train_path = 'organized_dataset/train'
    accuracy = model.train_models(train_path)
    
    if accuracy >= 0.80:
        print(f"\nAchieved target accuracy of {accuracy:.2f}! Saving models...")
        model.save_models()
    else:
        print(f"\nAccuracy ({accuracy:.2f}) below target (0.80). Model needs improvement.")

if __name__ == "__main__":
    # Install required packages if not already installed
    required_packages = [
        'tensorflow',
        'scikit-learn',
        'pillow',
        'pandas',
        'joblib'
    ]
    
    print("Checking and installing required packages...")
    import subprocess
    for package in required_packages:
        subprocess.check_call(['pip', 'install', package])
    
    # Train and save the models
    train_and_save_models()