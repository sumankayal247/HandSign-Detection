# Future Scope & Project Roadmap

As the foundational architecture (real-time browser inference, skeletal mesh tracking, static sign classification) is complete and stable, this document outlines the engineering roadmap to elevate the project into an advanced, state-of-the-art ASL recognition system.

## 1. Dynamic Gesture & Word Recognition (Temporal Modeling)
**Objective:** Transition from static alphabet recognition to full dynamic word recognition (e.g., "Hello", "Thank You", and moving letters like "J" and "Z").
*   **Approach:** Implement a sliding window architecture that captures the last *N* frames of hand landmarks.
*   **Model:** Replace the Random Forest with sequential deep learning models such as **LSTMs**, **GRUs**, or a **1D-CNN**. 
*   **Deployment:** Export the trained model to ONNX or TensorFlow.js for continued edge execution in the browser.

## 2. Holistic ASL Context (Face & Body Tracking)
**Objective:** Capture the complete meaning of ASL, which heavily relies on body posture and facial expressions (non-manual markers).
*   **Approach:** Upgrade the MediaPipe `HandLandmarker` to the `HolisticLandmarker` to extract face mesh, pose (shoulders/arms), and hands simultaneously.
*   **Impact:** This prevents occlusion errors (when hands cross each other) and allows the AI to understand nuances, such as raised eyebrows indicating a question.

## 3. NLP Autocorrect & Predictive Text
**Objective:** Provide a magical, forgiving user experience by correcting computer vision misclassifications using linguistic context.
*   **Approach:** Integrate a lightweight Natural Language Processing (NLP) layer (such as a simple n-gram model, Hidden Markov Model, or spell-check algorithm based on Levenshtein distance).
*   **Impact:** If the CV model predicts "HELLP", the NLP layer seamlessly auto-corrects it to "HELLO" based on standard dictionary probabilities, drastically increasing perceived accuracy.

## 4. Educational & Gamification Engine ("Duolingo Mode")
**Objective:** Transform the repository from a technical demo into an interactive web application that teaches users ASL.
*   **Approach:** Implement a UI module that prompts the user to spell specific words or perform signs under a time limit.
*   **Features:** Add scoring algorithms based on latency (how fast the user forms the sign) and accuracy (pose confidence scores).

## 5. WebWorker Offloading for Performance
**Objective:** Ensure buttery-smooth 60 FPS UI rendering, even on low-end mobile devices.
*   **Approach:** Decouple the ML inference loop from the main browser UI thread. 
*   **Architecture:** Move MediaPipe landmark extraction and model prediction into a dedicated **Web Worker**. This allows the heavy matrix multiplication to run on an isolated CPU core.

## 6. Few-Shot Personalization & Calibration
**Objective:** Make the model robust to different users' hand proportions, skin tones, and lighting conditions.
*   **Approach:** Introduce a 5-second calibration phase when the app opens, prompting the user to hold their hand up. 
*   **Impact:** The system extracts a baseline hand topology and adjusts the normalization algorithms to perfectly fit the user's specific biomechanics.

## 7. Continuous Sign Language Translation (CSLT)
**Objective:** Move beyond isolated word/letter recognition into translating continuous, flowing sentences in real-time.
*   **Approach:** Implement Connectionist Temporal Classification (CTC) loss during training. CTC allows neural networks to predict sequences of words without needing the training data to be perfectly aligned frame-by-frame.
*   **Impact:** The user can naturally sign a complete sentence without artificial pauses between words.

## 8. Edge Device & Wearable Deployment
**Objective:** Take the model off the web browser and put it into physical hardware.
*   **Approach:** Apply post-training quantization (reducing precision from FP32 to INT8) using TensorRT or Apache TVM.
*   **Impact:** The drastically reduced model size can be deployed directly onto Raspberry Pi-based smart cameras or even AR Smart Glasses to provide live ASL-to-text subtitles in the real world.
