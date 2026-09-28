import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

let handLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const rawText = document.getElementById("raw-text");
const stableText = document.getElementById("stable-text");
const accumText = document.getElementById("accum-text");

// Classifier state
const classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'];
let buffer = [];
const WINDOW_SIZE = 15;
const MIN_CONSENSUS = 0.7;

let accumulated_text = "";
let cooldown_frames = 0;
const COOLDOWN_MAX = 20;

async function createHandLandmarker() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 1
    });
    
    // Start webcam automatically if permissions are granted
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

function normalizeLandmarks(landmarks, handedness) {
    let x = landmarks.map(lm => lm.x);
    let y = landmarks.map(lm => lm.y);
    let z = landmarks.map(lm => lm.z);
    
    if (handedness === 'Left') {
        x = x.map(val => 1.0 - val);
    }
    
    const x0 = x[0], y0 = y[0], z0 = z[0];
    x = x.map(val => val - x0);
    y = y.map(val => val - y0);
    z = z.map(val => val - z0);
    
    const scale = Math.sqrt(x[9]*x[9] + y[9]*y[9] + z[9]*z[9]);
    if (scale > 0) {
        x = x.map(val => val / scale);
        y = y.map(val => val / scale);
        z = z.map(val => val / scale);
    }
    
    let features = [];
    for (let i = 0; i < 21; i++) {
        features.push(x[i], y[i], z[i]);
    }
    return features;
}

let lastVideoTime = -1;

async function predictWebcam() {
    if (!handLandmarker) return;

    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        
        const startTimeMs = performance.now();
        const results = handLandmarker.detectForVideo(video, startTimeMs);

        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        let current_prediction = 'UNKNOWN';

        if (results.landmarks && results.landmarks.length > 0) {
            const landmarks = results.landmarks[0];
            const handedness = results.handednesses[0][0].categoryName;
            
            // Draw landmarks (simple)
            canvasCtx.fillStyle = '#39ff14';
            for (const lm of landmarks) {
                canvasCtx.beginPath();
                canvasCtx.arc(lm.x * canvasElement.width, lm.y * canvasElement.height, 4, 0, 2 * Math.PI);
                canvasCtx.fill();
            }

            const features = normalizeLandmarks(landmarks, handedness);
            
            if (typeof predict === 'function') {
                const scores = predict(features);
                // m2cgen model usually returns class index or scores array. 
                // Since it's a classifier, it might return an array of scores
                let maxIdx = 0;
                if (Array.isArray(scores)) {
                    for(let i=1; i<scores.length; i++){
                        if (scores[i] > scores[maxIdx]) maxIdx = i;
                    }
                    current_prediction = classes[maxIdx];
                } else {
                    // if predict returns the class directly (m2cgen sometimes does this)
                    // Wait, m2cgen random forest classification usually returns a 1D array of scores (or 2D depending on the library version)
                    // We'll handle array safely, or if it's string, we'll use it directly
                    if (typeof scores === 'string') {
                        current_prediction = scores;
                    } else if (typeof scores === 'number') {
                         current_prediction = classes[scores];
                    }
                }
            }
        }
        
        rawText.innerText = "Raw Pred: " + current_prediction;
        
        // Stabilization
        buffer.push(current_prediction);
        if (buffer.length > WINDOW_SIZE) buffer.shift();
        
        let counts = {};
        let maxCount = 0;
        let mode = 'UNKNOWN';
        for (const p of buffer) {
            counts[p] = (counts[p] || 0) + 1;
            if (counts[p] > maxCount) {
                maxCount = counts[p];
                mode = p;
            }
        }
        
        let stable_letter = 'STABILIZING...';
        if (maxCount / buffer.length >= MIN_CONSENSUS) {
            stable_letter = mode;
        }
        stableText.innerText = "Stable: " + stable_letter;
        
        // Segmentation
        if (cooldown_frames > 0) {
            cooldown_frames--;
        } else {
            if (stable_letter !== 'UNKNOWN' && stable_letter !== 'STABILIZING...' && stable_letter !== 'nothing') {
                if (stable_letter === 'space') {
                    accumulated_text += " ";
                } else if (stable_letter === 'del') {
                    accumulated_text = accumulated_text.slice(0, -1);
                } else {
                    accumulated_text += stable_letter;
                }
                cooldown_frames = COOLDOWN_MAX;
                // clear buffer to force re-stabilization
                buffer = [];
            }
        }
        accumText.innerText = "Text: " + accumulated_text;
    }
    
    window.requestAnimationFrame(predictWebcam);
}

// Attach to globals
window.clearText = () => { accumulated_text = ""; accumText.innerText = "Text: "; };
window.addSpace = () => { accumulated_text += " "; accumText.innerText = "Text: " + accumulated_text; };
window.speakText = () => {
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(accumulated_text);
        window.speechSynthesis.speak(utterance);
    }
};

createHandLandmarker();
