import joblib
import os
import numpy as np

class ASLClassifier:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Returning 'UNKNOWN' predictions.")
            self.model = None
        else:
            self.model = joblib.load(model_path)
            
    def predict(self, normalized_features):
        \"\"\"
        Predict the ASL letter from normalized 1D landmark array (shape 63).
        Returns string class label.
        \"\"\"
        if self.model is None:
            return 'UNKNOWN'
        
        # Reshape to 2D array for sklearn
        features = np.array(normalized_features).reshape(1, -1)
        
        # Optionally, get probability if the model supports it
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(features)[0]
            max_prob = np.max(probs)
            # If highest probability is too low, reject
            if max_prob < 0.6:
                return 'UNKNOWN'
                
        prediction = self.model.predict(features)[0]
        return str(prediction)
