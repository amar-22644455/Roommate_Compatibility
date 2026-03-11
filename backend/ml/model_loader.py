# backend/ml/model_loader.py
import joblib
import os

# Training saves the model under backend/model/rf_model.pkl.
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model/rf_model.pkl'))

# Global variable to hold the model in memory
_rf_model = None

def get_model():
    """
    Loads the model from disk if it hasn't been loaded yet,
    otherwise returns the already loaded model from memory.
    """
    global _rf_model
    
    # If the model is already in memory, just return it instantly
    if _rf_model is not None:
        return _rf_model
        
    # If not, check if the file exists and load it
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"The model file was not found at {MODEL_PATH}. "
            "Make sure you have run backend/training/train_model.py first."
        )
        
    print(f"Loading RandomForest model from {MODEL_PATH} into memory...")
    _rf_model = joblib.load(MODEL_PATH)
    
    return _rf_model