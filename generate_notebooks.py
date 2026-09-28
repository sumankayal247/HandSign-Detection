import nbformat as nbf
import os

def create_notebook_01():
    nb = nbf.v4.new_notebook()
    
    nb['cells'] = [
        nbf.v4.new_markdown_cell("# 01 - Dataset Research and Download\n\nThis notebook demonstrates how to download the Kaggle ASL Alphabet Dataset. You will need a Kaggle account and an API token (`kaggle.json`)."),
        nbf.v4.new_markdown_cell("### Install Kaggle API"),
        nbf.v4.new_code_cell("!pip install kaggle"),
        nbf.v4.new_markdown_cell("### Upload Kaggle JSON\nRun this cell to upload your `kaggle.json` file if you are in Google Colab."),
        nbf.v4.new_code_cell("from google.colab import files\nfiles.upload()"),
        nbf.v4.new_code_cell("!mkdir -p ~/.kaggle\n!cp kaggle.json ~/.kaggle/\n!chmod 600 ~/.kaggle/kaggle.json"),
        nbf.v4.new_markdown_cell("### Download the Dataset"),
        nbf.v4.new_code_cell("!kaggle datasets download -d grassknoted/asl-alphabet\n!unzip -q asl-alphabet.zip -d ../data/raw/asl-alphabet")
    ]
    with open('notebooks/01_dataset_research_and_download.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_notebook_02():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell("# 02 - Extract Hand Landmarks\n\nIn this notebook, we use MediaPipe to extract 21 3D hand landmarks from the dataset images. Failed extractions are logged."),
        nbf.v4.new_code_cell("!pip install mediapipe opencv-python pandas tqdm"),
        nbf.v4.new_code_cell("""import cv2
import mediapipe as mp
import os
import glob
import pandas as pd
from tqdm import tqdm

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

dataset_path = '../data/raw/asl-alphabet/asl_alphabet_train/asl_alphabet_train'
classes = os.listdir(dataset_path)

data = []
failed = []

for cls in tqdm(classes):
    img_paths = glob.glob(os.path.join(dataset_path, cls, '*.jpg'))
    # Optional: limit images per class for faster extraction in testing
    # img_paths = img_paths[:500] 
    
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 1:
                landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label
                row = {'class': cls, 'handedness': handedness, 'image_path': img_path}
                for i, lm in enumerate(landmarks.landmark):
                    row[f'x_{i}'] = lm.x
                    row[f'y_{i}'] = lm.y
                    row[f'z_{i}'] = lm.z
                data.append(row)
            else:
                failed.append({'path': img_path, 'reason': 'multiple_hands'})
        else:
            failed.append({'path': img_path, 'reason': 'no_hands'})

df = pd.DataFrame(data)
df.to_csv('../data/processed/landmarks_raw.csv', index=False)

df_failed = pd.DataFrame(failed)
df_failed.to_csv('../data/processed/failed_extractions.csv', index=False)

print(f"Successfully processed: {len(df)}")
print(f"Failed to process: {len(df_failed)}")
""")
    ]
    with open('notebooks/02_extract_hand_landmarks.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_notebook_03():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell("# 03 - Prepare Dataset\n\nNormalize the extracted landmarks (translation to wrist, scale normalization, canonical handedness)."),
        nbf.v4.new_code_cell("""import pandas as pd
import numpy as np

df = pd.read_csv('../data/processed/landmarks_raw.csv')

def normalize_landmarks(row):
    # Extract x, y, z arrays
    x = np.array([row[f'x_{i}'] for i in range(21)])
    y = np.array([row[f'y_{i}'] for i in range(21)])
    z = np.array([row[f'z_{i}'] for i in range(21)])
    
    # 1. Handedness (Mirror left hands to look like right hands)
    if row['handedness'] == 'Left':
        x = 1.0 - x # Assuming x is normalized 0-1 by mediapipe
        
    # 2. Translation (Make wrist the origin)
    x = x - x[0]
    y = y - y[0]
    z = z - z[0]
    
    # 3. Scale (Normalize by distance between wrist and middle finger MCP)
    scale = np.sqrt(x[9]**2 + y[9]**2 + z[9]**2)
    if scale > 0:
        x = x / scale
        y = y / scale
        z = z / scale
        
    res = {}
    for i in range(21):
        res[f'norm_x_{i}'] = x[i]
        res[f'norm_y_{i}'] = y[i]
        res[f'norm_z_{i}'] = z[i]
    return pd.Series(res)

normalized = df.apply(normalize_landmarks, axis=1)
df_final = pd.concat([df[['class', 'image_path']], normalized], axis=1)
df_final.to_csv('../data/processed/landmarks_normalized.csv', index=False)
df_final.head()
""")
    ]
    with open('notebooks/03_prepare_dataset.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_notebook_04():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell("# 04 - Train Models\n\nTrain Random Forest and MLP classifiers on the normalized landmarks."),
        nbf.v4.new_code_cell("""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv('../data/processed/landmarks_normalized.csv')

# Exclude dynamic letters J and Z, and non-letters if desired
# df = df[~df['class'].isin(['J', 'Z', 'del', 'nothing', 'space'])]

X = df.drop(['class', 'image_path'], axis=1)
y = df['class']

# IMPORTANT: In a real scenario, group split by person. Since Kaggle ASL Alphabet lacks person IDs, we do a basic split here.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("RF Accuracy:", accuracy_score(y_test, rf_preds))

print("Training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test)
print("MLP Accuracy:", accuracy_score(y_test, mlp_preds))

joblib.dump(rf, '../models/rf_baseline.pkl')
joblib.dump(mlp, '../models/mlp_baseline.pkl')
""")
    ]
    with open('notebooks/04_train_models.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_notebook_05():
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell("# 05 & 06 - Evaluate and Export\n\nEvaluate the confusion matrix and export the best model."),
        nbf.v4.new_code_cell("""import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('../data/processed/landmarks_normalized.csv')
X = df.drop(['class', 'image_path'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = joblib.load('../models/rf_baseline.pkl')
mlp = joblib.load('../models/mlp_baseline.pkl')

print(rf.classes_)
cm = confusion_matrix(y_test, rf.predict(X_test), labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
fig, ax = plt.subplots(figsize=(15,15))
disp.plot(ax=ax)
plt.show()

# Export chosen model
joblib.dump(rf, '../models/best_asl_model.pkl')
""")
    ]
    with open('notebooks/05_06_evaluate_export.ipynb', 'w') as f:
        nbf.write(nb, f)

if __name__ == '__main__':
    create_notebook_01()
    create_notebook_02()
    create_notebook_03()
    create_notebook_04()
    create_notebook_05()
    print("Notebooks created successfully!")
