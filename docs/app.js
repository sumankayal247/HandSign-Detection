import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

let modelTrees = null;
fetch('model_flat.json')
    .then(res => res.json())
    .then(data => {
        modelTrees = data;
        console.log("Model loaded successfully.");
    });

window.score = function(input) {
    if (!modelTrees) return null;
    let result = new Array(28).fill(0);
    for (const tree of modelTrees) {
        let nodeIdx = 0;
        while (true) {
            const node = tree[nodeIdx];
            if (node[0] === -1) {
                const vals = node[1];
                for(let i=0; i<28; i++) result[i] += vals[i];
                break;
            } else {
                if (input[node[0]] <= node[1]) {
                    nodeIdx = node[2];
                } else {
                    nodeIdx = node[3];
                }
            }
        }
    }
    return result;
};

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
let last_added_letter = null;

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

const connections = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[17,18],[18,19],[19,20],
    [0,17]
];

let lastVideoTime = -1;
let smoothedLandmarks = null;
const SMOOTHING_FACTOR = 0.5; // Lower is smoother

async function predictWebcam() {
    if (!handLandmarker) return;

    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        
        const startTimeMs = performance.now();
        const results = handLandmarker.detectForVideo(video, startTimeMs);

        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        let current_prediction = 'UNKNOWN';
        let landmarks_to_draw = null;

        if (results.landmarks && results.landmarks.length > 0) {
            const raw_landmarks = results.landmarks[0];
            
            if (!smoothedLandmarks) {
                smoothedLandmarks = raw_landmarks.map(lm => ({x: lm.x, y: lm.y, z: lm.z}));
            } else {
                for (let i = 0; i < raw_landmarks.length; i++) {
                    smoothedLandmarks[i].x = SMOOTHING_FACTOR * raw_landmarks[i].x + (1 - SMOOTHING_FACTOR) * smoothedLandmarks[i].x;
                    smoothedLandmarks[i].y = SMOOTHING_FACTOR * raw_landmarks[i].y + (1 - SMOOTHING_FACTOR) * smoothedLandmarks[i].y;
                    smoothedLandmarks[i].z = SMOOTHING_FACTOR * raw_landmarks[i].z + (1 - SMOOTHING_FACTOR) * smoothedLandmarks[i].z;
                }
            }
            landmarks_to_draw = smoothedLandmarks;
            const handedness = results.handednesses[0][0].categoryName;

            const features = normalizeLandmarks(landmarks_to_draw, handedness);
            
            if (typeof window.score === 'function') {
                try {
                    const scores = window.score(features);
                    let maxIdx = 0;
                    if (Array.isArray(scores)) {
                        for(let i=1; i<scores.length; i++){
                            if (scores[i] > scores[maxIdx]) maxIdx = i;
                        }
                        current_prediction = classes[maxIdx];
                    } else {
                        if (typeof scores === 'string') {
                            current_prediction = scores;
                        } else if (typeof scores === 'number') {
                             current_prediction = classes[scores];
                        }
                    }
                } catch (e) {
                    current_prediction = "ERR";
                    console.error("Score error:", e);
                }
            } else {
                current_prediction = "NO_MDL";
            }
        } else {
            smoothedLandmarks = null; // reset if hand is lost
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

        // Draw landmarks and mesh
        if (landmarks_to_draw) {
            let isValid = (stable_letter !== 'UNKNOWN' && stable_letter !== 'STABILIZING...' && stable_letter !== 'nothing');
            let color = isValid ? '#39ff14' : '#aaaaaa';
            let lineColor = isValid ? '#2ecc11' : '#666666';

            // Draw connections
            canvasCtx.strokeStyle = lineColor;
            canvasCtx.lineWidth = 2;
            for (const [startIdx, endIdx] of connections) {
                const startNode = landmarks_to_draw[startIdx];
                const endNode = landmarks_to_draw[endIdx];
                canvasCtx.beginPath();
                canvasCtx.moveTo(startNode.x * canvasElement.width, startNode.y * canvasElement.height);
                canvasCtx.lineTo(endNode.x * canvasElement.width, endNode.y * canvasElement.height);
                canvasCtx.stroke();
            }

            // Draw points
            canvasCtx.fillStyle = color;
            for (const lm of landmarks_to_draw) {
                canvasCtx.beginPath();
                canvasCtx.arc(lm.x * canvasElement.width, lm.y * canvasElement.height, 4, 0, 2 * Math.PI);
                canvasCtx.fill();
            }
        }
        
        // Segmentation
        if (cooldown_frames > 0) {
            cooldown_frames--;
        } else {
            if (stable_letter !== 'UNKNOWN' && stable_letter !== 'STABILIZING...' && stable_letter !== 'nothing') {
                if (stable_letter !== last_added_letter) {
                    if (stable_letter === 'space') {
                        accumulated_text += " ";
                    } else if (stable_letter === 'del') {
                        accumulated_text = accumulated_text.slice(0, -1);
                    } else {
                        accumulated_text += stable_letter;
                    }
                    last_added_letter = stable_letter;
                    cooldown_frames = COOLDOWN_MAX;
                    // clear buffer to force re-stabilization
                    buffer = [];
                }
            } else if (stable_letter === 'nothing' || stable_letter === 'UNKNOWN') {
                // If the user drops their hand or shows nothing, reset last_added_letter 
                // so they can type the same letter again.
                last_added_letter = null;
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
