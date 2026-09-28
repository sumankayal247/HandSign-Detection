# Real-Time ASL Fingerspelling Recognition Using Hand Landmarks

<a href="https://sumankayal247.github.io/HandSign-Detection/" target="_blank">
  <img src="https://img.shields.io/badge/Play_%3E-39ff14?style=for-the-badge&logo=play&logoColor=black" alt="Play >" width="250" height="75" />
</a>

This project implements a real-time computer vision system to recognize American Sign Language (ASL) fingerspelling using a webcam.

## Features
- Recognizes static ASL letters (A-Z)
- Temporal stabilization to prevent noisy frame predictions
- Gesture segmentation to accumulate spelled words
- Text-to-Speech (TTS) playback

## Setup
1. Create a virtual environment: `python -m venv venv`
2. Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\\Scripts\\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`

## Training
1. Open the `.ipynb` files in the `notebooks/` directory.
2. We recommend running these notebooks in Google Colab.
3. Download the Kaggle ASL Alphabet dataset (Notebook 01).
4. Run the pipeline to extract landmarks (Notebook 02) and prepare the dataset (Notebook 03).
5. Train and evaluate the models (Notebooks 04 & 05).
6. Save the final model as `models/best_asl_model.pkl`.

## Running the Application
Once the model is trained and saved:
```bash
python app/main.py
```
- **Controls**:
  - `q` : Quit
  - `c` : Clear text
  - `s` : Speak current text
  - `space`: Add a space to text
