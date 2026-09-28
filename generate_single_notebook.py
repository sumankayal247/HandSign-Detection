import nbformat as nbf

def create_single_notebook():
    nb = nbf.v4.new_notebook()
    
    nb['cells'] = [
        # Part 1
        nbf.v4.new_markdown_cell("# Real-Time ASL Fingerspelling Recognition Pipeline\n\nThis notebook handles downloading the dataset, extracting landmarks, normalizing them, training the model, and exporting it in one go."),
        nbf.v4.new_markdown_cell("## 1. Setup & Download Dataset"),
        nbf.v4.new_code_cell("!pip install kaggle mediapipe opencv-python pandas tqdm scikit-learn"),
        nbf.v4.new_code_cell("from google.colab import files\nprint('Please upload your kaggle.json file:')\nfiles.upload()"),
        nbf.v4.new_code_cell("!mkdir -p ~/.kaggle\n!cp kaggle.json ~/.kaggle/\n!chmod 600 ~/.kaggle/kaggle.json\n!kaggle datasets download -d grassknoted/asl-alphabet\n!unzip -q asl-alphabet.zip -d data/raw/asl-alphabet"),
        
        # Part 2
        nbf.v4.new_markdown_cell("## 2. Extract Hand Landmarks using MediaPipe"),
        nbf.v4.new_code_cell("""import cv2
import mediapipe as mp
import os
import glob
import pandas as pd
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

dataset_path = 'data/raw/asl-alphabet/asl_alphabet_train/asl_alphabet_train'
classes = os.listdir(dataset_path)

os.makedirs('data/processed', exist_ok=True)

data = []
failed = []

print("Extracting landmarks (this may take a while)...")
for cls in tqdm(classes):
    img_paths = glob.glob(os.path.join(dataset_path, cls, '*.jpg'))
    # Optional: uncomment the next line to run a faster test on a subset of data
    # img_paths = img_paths[:200] 
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None: continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            row = {'class': cls, 'handedness': handedness, 'image_path': img_path}
            for i, lm in enumerate(landmarks.landmark):
                row[f'x_{i}'] = lm.x
                row[f'y_{i}'] = lm.y
                row[f'z_{i}'] = lm.z
            data.append(row)
        else:
            failed.append({'path': img_path})

df = pd.DataFrame(data)
df.to_csv('data/processed/landmarks_raw.csv', index=False)
print(f"\\nSuccessfully processed {len(df)} images.")
"""),
        
        # Part 3
        nbf.v4.new_markdown_cell("## 3. Normalize Landmarks\nTranslate to wrist origin, scale by hand size, and mirror left hands to look like right hands."),
        nbf.v4.new_code_cell("""import numpy as np

df = pd.read_csv('data/processed/landmarks_raw.csv')

def normalize_landmarks(row):
    x = np.array([row[f'x_{i}'] for i in range(21)])
    y = np.array([row[f'y_{i}'] for i in range(21)])
    z = np.array([row[f'z_{i}'] for i in range(21)])
    
    if row['handedness'] == 'Left':
        x = 1.0 - x
        
    x = x - x[0]
    y = y - y[0]
    z = z - z[0]
    
    scale = np.sqrt(x[9]**2 + y[9]**2 + z[9]**2)
    if scale > 0:
        x, y, z = x / scale, y / scale, z / scale
        
    res = {}
    for i in range(21):
        res[f'norm_x_{i}'] = x[i]
        res[f'norm_y_{i}'] = y[i]
        res[f'norm_z_{i}'] = z[i]
    return pd.Series(res)

print("Normalizing features...")
normalized = df.apply(normalize_landmarks, axis=1)
df_final = pd.concat([df[['class', 'image_path']], normalized], axis=1)
df_final.to_csv('data/processed/landmarks_normalized.csv', index=False)
print("Done.")
"""),

        # Part 4
        nbf.v4.new_markdown_cell("## 4. Train Models\nTrain a Random Forest model as our baseline classifier."),
        nbf.v4.new_code_cell("""from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv('data/processed/landmarks_normalized.csv')
X = df.drop(['class', 'image_path'], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_preds))
print("\\nClassification Report:\\n", classification_report(y_test, rf_preds))

os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/best_asl_model.pkl')
print("Model saved to models/best_asl_model.pkl")
"""),

        # Part 5
        nbf.v4.new_markdown_cell("## 5. Download the Model\nRun this cell to download the trained model to your computer."),
        nbf.v4.new_code_cell("""from google.colab import files
files.download('models/best_asl_model.pkl')
""")
    ]
    with open('notebooks/asl_training_pipeline.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_single_notebook()
