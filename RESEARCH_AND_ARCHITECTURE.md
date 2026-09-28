# Research and Architecture Report: Real-Time ASL Fingerspelling Recognition

## 1. Problem Definition
The objective is to build a real-time computer vision system capable of recognizing American Sign Language (ASL) fingerspelling (letter-by-letter) via a standard webcam. The system will convert these recognized static gestures into accumulated text and provide an option to synthesize the text to speech (TTS). 

## 2. Exact MVP Scope
- **Inputs**: Real-time webcam video feed (focusing on one hand at a time).
- **Outputs**: Accumulated text string of spelled words, with optional TTS.
- **Capabilities**: Recognize static ASL letters A–Z (with special handling or exclusion for dynamic letters J and Z initially).
- **Core Technology**: Hand landmark extraction (MediaPipe) followed by a lightweight classifier (e.g., SVM, Random Forest, or shallow MLP).
- **Out of Scope**: Full continuous ASL sentence translation, two-handed signs, facial expression analysis.

## 3. ASL Alphabet / Fingerspelling Considerations
- **Static vs. Dynamic**: Letters A-I, K-Y are primarily static poses. However, **J** and **Z** involve motion (J draws a 'J' in the air, Z draws a 'Z' in the air). 
- **Recommendation for MVP**: We will initially focus on the static letters. For J and Z, we can either exclude them from the strict static evaluation or implement a simple trajectory heuristic (e.g., tracking the index finger tip over a sliding window of 10 frames) if we decide to include them. 
- **Viewpoint Dependence**: ASL is usually signed facing the observer. Hand orientation relative to the camera is crucial.

## 4. Candidate Datasets
1. **Kaggle ASL Alphabet Dataset**: ~87,000 images, 200x200 pixels, 29 classes (A-Z, SPACE, DEL, NOTHING).
2. **Massey University ASL Dataset**: ~2,500 images of 5 different signers under different lighting conditions and backgrounds.
3. **Sign Language MNIST**: 27,455 cases for training and 7172 cases for testing. Pixel-based, mostly cropped on the hand.
4. **ChicagoFSWild**: Real-world continuous fingerspelling (mostly video/temporal). 

## 5. Dataset Comparison
- **Kaggle ASL Alphabet**: Huge number of images, but heavily correlated (extracted from video of a single or very few signers). High risk of data leakage if not split properly. Images are clear and suitable for landmark extraction.
- **Massey University**: Smaller, but provides explicit signer separation (5 signers). Better for evaluating generalization to unseen users. 
- **Sign Language MNIST**: Low resolution (28x28 grayscale). Might be too degraded for robust MediaPipe landmark extraction.
- **ChicagoFSWild**: Too complex for the MVP scope (in-the-wild, continuous, motion blur).

## 6. Recommended Dataset and Justification
**Recommendation**: Start with the **Kaggle ASL Alphabet Dataset** for initial pipeline validation due to its ease of access and large volume of clear images, but perform strict grouping (if metadata allows) or artificial clustering to avoid train/test leakage. 
If signer metadata is completely absent and images are just video frames, we will blend in the **Massey University ASL Dataset** for the **Test Set** to ensure the model actually generalizes to unseen users.

## 7. Image → Landmark Conversion Viability
This is highly viable. We will use **Google MediaPipe Hands**. It is robust to background noise and extracts 21 3D hand landmarks ($x, y, z$). We expect a high conversion rate on the Kaggle and Massey datasets because they feature prominent, mostly centered hands.
We will filter out images where MediaPipe fails to detect a hand or detects multiple hands, logging the failure rates per class.

## 8. Recommended Landmark Representation
Instead of raw image pixels, the classifier will use the 21 extracted landmarks.
- **Raw**: 21 points $\times$ (x, y, z) = 63 features.
- **Normalization Strategy**: 
  1. Translation invariance: Set the wrist (landmark 0) as the origin (0,0,0). Subtract wrist coordinates from all other landmarks.
  2. Scale invariance: Compute the maximum bounding box dimension or the distance from the wrist to the middle finger MCP joint, and divide all coordinates by this scalar.
  3. Handedness: MediaPipe detects left/right hands. We will mirror the x-coordinates of left hands to canonicalize all inputs to appear as right hands.

## 9. Recommended Preprocessing
1. Read Image $\rightarrow$ MediaPipe Hands $\rightarrow$ Extract (x,y,z).
2. Apply Translation & Scale Normalization.
3. Apply Handedness Canonicalization.
4. Flatten to a 1D array of 63 floats.
5. (Optional Augmentation in Training): Add slight Gaussian noise to coordinates, small affine rotations (simulated by rotating the 3D points).

## 10. Candidate ML Models
Since the feature space is extremely small (63 continuous variables), deep CNNs are unnecessary.
- **Support Vector Machine (SVM)** with RBF kernel: Excellent for high-dimensional margin separation.
- **Random Forest / XGBoost**: Highly interpretable, fast inference, robust to overfitting.
- **Multi-Layer Perceptron (MLP)**: A small feed-forward network (e.g., 2 hidden layers of 128 and 64 neurons).
*Selection*: We will benchmark these three. XGBoost or a small MLP usually provides the best balance of speed and accuracy for landmark-based classification.

## 11. Evaluation Methodology
- **Metric**: Macro F1-Score (to ensure less frequent classes or harder letters are not masked by overall accuracy), Confusion Matrix.
- **Data Split**: We must test on a hold-out set of images that are significantly visually different (ideally, the Massey dataset with unseen signers) to prove the model learned the ASL pose, not the background or the specific user's skin tone/clothing.

## 12. Real-time Architecture
```text
Webcam (OpenCV) -> Frame (RGB) -> MediaPipe Hands -> Normalize -> Classifier (scikit-learn/XGBoost) -> Raw Prediction -> Temporal Buffer -> UI
```

## 13. Temporal Stabilization Strategy
A single frame prediction can be noisy.
- **Windowed Majority Vote**: Maintain a rolling buffer of the last $N=15$ frames (at 30 FPS, this is 0.5 seconds).
- The stabilized letter is the mode (most frequent prediction) in the buffer, provided its frequency exceeds a threshold (e.g., 70% of the buffer).
- If no class meets the threshold, output "UNKNOWN/TRANSITION".

## 14. Gesture Segmentation Strategy
To prevent "H" from registering as "HHHH":
- **State Machine**: 
  - `IDLE`: No hand / low confidence.
  - `RECOGNIZING`: Hand detected, buffer filling.
  - `COMMITTED`: A letter is stable for $T$ seconds. Append to text buffer.
  - `COOLDOWN`: Wait for the hand to drop, transition to an "UNKNOWN" state, or detect a significant shift in wrist position before allowing the next letter to commit.

## 15. TTS Options
- **pyttsx3**: Best local, offline, cross-platform python library. No API keys needed, zero cost, completely local.
- **Web Speech API**: If we were building a web app, this is built into the browser.
*Decision*: For a local Python application MVP, `pyttsx3` is the clear choice.

## 16. Google Colab Training Workflow
The repository will contain modular Jupyter notebooks. Data will be pulled using Kaggle API or wget. MediaPipe extraction will run on Colab CPUs. Training (XGBoost/MLP) will be very fast on CPU/standard GPU. Models will be exported using `joblib` or `pickle`.

## 17. Local Inference Architecture
- Language: Python.
- UI: OpenCV window with text overlays.
- Libraries: `cv2`, `mediapipe`, `joblib`, `xgboost`/`sklearn`, `pyttsx3`.
- Hardware: Standard CPU. MediaPipe and small MLPs/Trees run comfortably at 30+ FPS on typical laptops.

## 18. Expected Limitations
- **Motion Letters**: J and Z will likely fail or require hacky workarounds in a purely static classifier.
- **Occlusion**: Letters like 'M' and 'N' or 'R' and 'U' look very similar from a 2D camera perspective and heavily rely on subtle finger occlusion which MediaPipe sometimes struggles to resolve accurately.

## 19. Risks
- **Data Leakage**: Overestimating model performance due to evaluating on frames adjacent to the training frames.
- **False Positives**: The model might confidently predict a letter when the user is just scratching their face. An "UNKNOWN" class or a strict confidence threshold is required.

## 20. Project Directory Structure
```text
asl-fingerspelling-ai/
│
├── notebooks/
│   ├── 01_dataset_research_and_download.ipynb
│   ├── 02_extract_hand_landmarks.ipynb
│   ├── 03_prepare_dataset.ipynb
│   ├── 04_train_models.ipynb
│   ├── 05_evaluate_models.ipynb
│   └── 06_export_model.ipynb
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│
├── models/
│
├── src/
│   ├── hand_tracking/
│   ├── preprocessing/
│   ├── recognition/
│   ├── temporal/
│   └── tts/
│
├── app/
│   └── main.py
│
├── evaluation/
│
├── requirements.txt
│
└── README.md
```

## 21. Step-by-Step Implementation Plan
1. **Repository Setup**: Create the directory structure and initialize `requirements.txt`.
2. **Notebook 01 & 02**: Setup data downloading and write the MediaPipe extraction script. Test extraction on a sample of images.
3. **Notebook 03**: Write the normalization logic (translation, scaling, handedness) and generate the `.csv` or `.npy` feature dataset.
4. **Notebook 04 & 05**: Train Baseline classifiers (Random Forest, SVM) and evaluate them with a confusion matrix.
5. **Notebook 06**: Export the best performing model pipeline.
6. **Core Modules (`src/`)**: Port the normalization and prediction code into clean Python modules.
7. **Real-time Engine (`app/main.py`)**: Integrate webcam, MediaPipe, model inference, and temporal stabilization.
8. **Text & TTS**: Add string accumulation, clear/space controls, and `pyttsx3` voice output.
9. **Final Testing**: Test locally with a webcam and refine cooldown/stabilization thresholds.
