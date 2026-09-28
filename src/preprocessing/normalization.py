import numpy as np

def normalize_landmarks(landmarks, handedness):
    \"\"\"
    Normalizes MediaPipe landmarks for the ASL classifier.
    landmarks: list of 21 objects with .x, .y, .z attributes
    handedness: str, 'Left' or 'Right'
    
    Returns:
    numpy array of shape (63,) containing the normalized coordinates.
    \"\"\"
    x = np.array([lm.x for lm in landmarks])
    y = np.array([lm.y for lm in landmarks])
    z = np.array([lm.z for lm in landmarks])
    
    # Mirror left hands to look like right hands
    if handedness == 'Left':
        x = 1.0 - x
        
    # Translate wrist to origin
    x = x - x[0]
    y = y - y[0]
    z = z - z[0]
    
    # Scale normalization
    scale = np.sqrt(x[9]**2 + y[9]**2 + z[9]**2)
    if scale > 0:
        x = x / scale
        y = y / scale
        z = z / scale
        
    # Flatten to 1D array
    features = []
    for i in range(21):
        features.extend([x[i], y[i], z[i]])
        
    return np.array(features)
