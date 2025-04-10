import os
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Input, Conv2D, LayerNormalization, Activation
)
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
    CSVLogger
)
import tensorflow.keras.backend as K
import joblib
import logging
from datetime import datetime, UTC
import gc
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_training.log')
    ]
)

def weighted_categorical_crossentropy(class_weights):
    """Stable weighted categorical crossentropy loss"""
    class_weights = K.variable(class_weights)
    
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        loss = y_true * K.log(y_pred) * class_weights
        return -K.mean(loss, axis=-1)
    
    return loss

class DiabeticRetinopathyModel:
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.cnn_model = None
        self.rf_classifier = None
        self.class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        self.history = None
        self.batch_size = 32
        logging.info(f"Initialized model with image size {img_size}")

    def preprocess_image(self, image_path):
        """Enhanced image preprocessing"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE to each channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            img = img.astype(np.float32)
            
            for i in range(3):
                img[:,:,i] = clahe.apply(img[:,:,i].astype(np.uint8))
            
            # Standardization
            mean = np.mean(img)
            std = np.std(img)
            img = (img - mean) / (std + 1e-7)
            
            # Resize
            img = cv2.resize(img, (self.img_size, self.img_size), 
                           interpolation=cv2.INTER_LANCZOS4)
            
            return img
            
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return None

    def load_models(self, backup_dir='backup_training_results'):
        """Load models from backup directory"""
        try:
            # Load CNN model
            cnn_path = os.path.join(backup_dir, 'cnn_model.keras')
            if not os.path.exists(cnn_path):
                raise ValueError(f"CNN model not found at {cnn_path}")
            
            self.cnn_model = tf.keras.models.load_model(
                cnn_path,
                custom_objects={
                    'loss': weighted_categorical_crossentropy([1.0, 2.0, 3.0, 3.0, 3.0])
                }
            )
            logging.info(f"CNN model loaded from {cnn_path}")
            
            # Load RF model
            rf_path = os.path.join(backup_dir, 'rf_model.joblib')
            if not os.path.exists(rf_path):
                raise ValueError(f"RF model not found at {rf_path}")
            
            self.rf_classifier = joblib.load(rf_path)
            logging.info(f"RF model loaded from {rf_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise

    def predict(self, image_path):
        """Make predictions with enhanced confidence"""
        if self.cnn_model is None or self.rf_classifier is None:
            raise ValueError("Models not loaded. Please load models first.")
        
        try:
            # Preprocess image
            img = self.preprocess_image(image_path)
            if img is None:
                raise ValueError("Failed to preprocess image")
            
            if np.isnan(img).any():
                raise ValueError("NaN values detected in preprocessed image")
            
            # Get CNN features
            feature_model = Model(
                inputs=self.cnn_model.input,
                outputs=self.cnn_model.get_layer('global_average_pooling2d').output
            )
            
            features = feature_model.predict(np.array([img]), verbose=0)
            
            # Get predictions from both models
            rf_proba = self.rf_classifier.predict_proba(features)[0]
            cnn_proba = self.cnn_model.predict(np.array([img]), verbose=0)[0]
            
            # Weighted ensemble (CNN has higher weight)
            combined_proba = (0.7 * cnn_proba + 0.3 * rf_proba)
            prediction = np.argmax(combined_proba)
            
            # Prepare result
            result = {
                'prediction': int(prediction),
                'class_name': self.class_names[prediction],
                'confidence': float(combined_proba[prediction]),
                'probabilities': {
                    self.class_names[i]: float(prob) 
                    for i, prob in enumerate(combined_proba)
                }
            }
            
            logging.info(f"Prediction for {image_path}: {result['class_name']} "
                        f"(confidence: {result['confidence']:.2f})")
            
            return result
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise

    def get_model_info(self):
        """Get model information"""
        return {
            'image_size': self.img_size,
            'class_names': self.class_names,
            'models_loaded': self.cnn_model is not None and self.rf_classifier is not None,
            'cnn_summary': self.cnn_model.summary() if self.cnn_model else None,
            'timestamp': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
        }