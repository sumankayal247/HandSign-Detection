import sys
import os
import cv2
import time
from flask import Flask, render_template_string, Response, jsonify, request
from flask_cors import CORS

# Add parent dir to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_tracking.detector import HandDetector
from src.preprocessing.normalization import normalize_landmarks
from src.recognition.classifier import ASLClassifier
from src.temporal.stabilizer import TemporalStabilizer, GestureSegmenter
from src.tts.speaker import TTSEngine

app = Flask(__name__)
CORS(app)

# Global states
accumulated_text = ""
current_prediction = "None"
stable_letter = "None"
tts = TTSEngine()

# Initialize AI models globally so they aren't reloaded every frame
detector = HandDetector()
classifier = ASLClassifier(model_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'best_asl_model.pkl'))
stabilizer = TemporalStabilizer(window_size=15, min_consensus=0.7)
segmenter = GestureSegmenter(cooldown_frames=20)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ASL Web Recognition</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #1a1a1a; color: white; text-align: center; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: auto; background: #2a2a2a; padding: 20px; border-radius: 12px; box-shadow: 0 8px 16px rgba(0,0,0,0.5); }
        h1 { color: #39ff14; }
        .video-container { position: relative; width: 640px; height: 480px; margin: auto; background: #000; border-radius: 8px; overflow: hidden; border: 2px solid #444; }
        .controls { margin-top: 20px; display: flex; justify-content: center; gap: 15px; }
        button { padding: 12px 24px; font-size: 18px; cursor: pointer; border: none; border-radius: 8px; background: #39ff14; color: #000; font-weight: bold; transition: 0.3s; }
        button:hover { background: #2ecc11; transform: scale(1.05); }
        #permission-msg { display: none; color: #ff4444; font-weight: bold; padding: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>ASL Real-Time Recognition</h1>
        <p>Ensure your webcam is enabled and sign ASL alphabets to the camera.</p>
        
        <div id="permission-msg">Camera permission denied! Please allow camera access in your browser settings.</div>
        
        <div class="video-container">
            <img id="feed" src="/video_feed" width="640" height="480" style="display:none;" />
        </div>
        <div class="controls">
            <button onclick="sendCommand('clear')">Clear Text</button>
            <button onclick="sendCommand('space')">Add Space</button>
            <button onclick="sendCommand('speak')">Speak (TTS)</button>
        </div>
    </div>
    
    <script>
        // Check permissions before showing feed
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                // Permission granted
                stream.getTracks().forEach(track => track.stop()); // Stop the stream we just requested
                document.getElementById('feed').style.display = 'block';
            })
            .catch(function(err) {
                // Permission denied or error
                document.getElementById('permission-msg').style.display = 'block';
            });

        function sendCommand(cmd) {
            fetch('/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: cmd })
            });
        }
    </script>
</body>
</html>
"""

def generate_frames():
    global accumulated_text, current_prediction, stable_letter
    
    cap = cv2.VideoCapture(0)
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
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
    cap.release()

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command', methods=['POST'])
def command():
    global accumulated_text, tts
    data = request.json
    action = data.get('action')
    
    if action == 'clear':
        accumulated_text = ""
    elif action == 'space':
        accumulated_text += " "
    elif action == 'speak':
        tts.speak(accumulated_text)
        
    return jsonify({"status": "success", "text": accumulated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
