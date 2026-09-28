import sys
import os
import cv2
import time

# Add parent dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_tracking.detector import HandDetector
from src.preprocessing.normalization import normalize_landmarks
from src.recognition.classifier import ASLClassifier
from src.temporal.stabilizer import TemporalStabilizer, GestureSegmenter
from src.tts.speaker import TTSEngine

def main():
    detector = HandDetector()
    classifier = ASLClassifier(model_path='../models/best_asl_model.pkl')
    stabilizer = TemporalStabilizer(window_size=15, min_consensus=0.7)
    segmenter = GestureSegmenter(cooldown_frames=20)
    tts = TTSEngine()
    
    cap = cv2.VideoCapture(0)
    
    accumulated_text = ""
    current_prediction = "None"
    stable_letter = "None"
    
    print("Press 'q' to quit. Press 'c' to clear text. Press 's' to speak. Press 'space' for a space.")
    
    start_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1) # Mirror for self-view
        timestamp_ms = (time.time() - start_time) * 1000
        
        results = detector.process_frame(frame, timestamp_ms)
        
        # Predict
        if results.hand_landmarks and len(results.hand_landmarks) == 1:
            detector.draw_landmarks(frame, results)
            landmarks = results.hand_landmarks[0]
            handedness = results.handedness[0][0].category_name
            
            # Extract and normalize
            features = normalize_landmarks(landmarks, handedness)
            
            # Predict
            current_prediction = classifier.predict(features)
        else:
            current_prediction = 'UNKNOWN'
            
        # Temporal stabilization
        stb_let = stabilizer.update(current_prediction)
        if stb_let:
            stable_letter = stb_let
        else:
            stable_letter = 'STABILIZING...'
            
        # Segmentation
        new_char = segmenter.process_stable_letter(stb_let)
        if new_char and new_char not in ['UNKNOWN', 'del', 'nothing', 'space']:
            accumulated_text += new_char
        elif new_char == 'space':
            accumulated_text += " "
        elif new_char == 'del':
            accumulated_text = accumulated_text[:-1]
            
        # UI overlays
        cv2.putText(frame, f"Raw Pred: {current_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Stable: {stable_letter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Text: {accumulated_text}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow("ASL Fingerspelling", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            accumulated_text = ""
        elif key == ord('s'):
            tts.speak(accumulated_text)
        elif key == 32: # spacebar
            accumulated_text += " "
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
