import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops

class RetinopathyModel:
    def __init__(self, n_estimators=500, max_depth=30):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=2
        )
        self.scaler = StandardScaler()
        
    def extract_lbp_features(self, image, P=8, R=1):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, P, R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return hist
    
    def extract_glcm_features(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        distances = [1, 2, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
        features = []
        
        for prop in properties:
            features.extend(graycoprops(glcm, prop).ravel())
            
        return np.array(features)

    def preprocess_image(self, image_path, target_size=(299, 299)):
        # Read and resize image
        if isinstance(image_path, str):
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_path
            
        img = cv2.resize(img, target_size)
        
        # Apply CLAHE to enhance contrast
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Extract basic statistical features
        features = []
        
        # Color channel statistics
        for channel in cv2.split(img):
            features.extend([
                np.mean(channel),
                np.std(channel),
                np.percentile(channel, 25),
                np.percentile(channel, 75),
                np.max(channel),
                np.min(channel),
                np.median(channel)
            ])
        
        # Add LBP features
        lbp_features = self.extract_lbp_features(img)
        features.extend(lbp_features)
        
        # Add GLCM features
        glcm_features = self.extract_glcm_features(img)
        features.extend(glcm_features)
        
        # Extract edge features
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Sobel edges
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features.extend([
            np.mean(np.abs(sobelx)),
            np.std(np.abs(sobelx)),
            np.mean(np.abs(sobely)),
            np.std(np.abs(sobely))
        ])
        
        # Laplacian features
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([
            np.mean(np.abs(laplacian)),
            np.std(np.abs(laplacian))
        ])
        
        # Add circular region analysis
        height, width = gray.shape
        center = (width // 2, height // 2)
        radius = min(width, height) // 3
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        
        optic_disc_region = cv2.bitwise_and(gray, gray, mask=mask)
        features.extend([
            np.mean(optic_disc_region[mask > 0]),
            np.std(optic_disc_region[mask > 0])
        ])
        
        return np.array(features)

    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        # Train the model
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        # Scale features and predict
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        # Scale features and predict probabilities
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save_model(self, model_path, scaler_path):
        # Save both the model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
    
    @classmethod
    def load_model(cls, model_path, scaler_path):
        # Create a new instance
        instance = cls()
        # Load the saved model and scaler
        instance.model = joblib.load(model_path)
        instance.scaler = joblib.load(scaler_path)
        return instance
