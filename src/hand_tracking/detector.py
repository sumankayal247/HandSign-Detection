import cv2
import mediapipe as mp
import os
import urllib.request

class HandDetector:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7):
        # We use the new Tasks API to avoid the deprecated 'solutions' error
        self.model_path = 'hand_landmarker.task'
        if not os.path.exists(self.model_path):
            print("Downloading hand_landmarker.task...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Use VIDEO mode for processing frames in a stream
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence)
            
        self.detector = HandLandmarker.create_from_options(options)
        
        # We'll still use drawing_utils to draw landmarks easily
        self.mp_draw = mp.solutions.drawing_utils if hasattr(mp, 'solutions') else None
        self.mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS if hasattr(mp, 'solutions') else None

    def process_frame(self, frame, timestamp_ms):
        \"\"\"
        Process an BGR OpenCV frame and return hand detection results.
        \"\"\"
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # The new Tasks API for video requires a timestamp
        results = self.detector.detect_for_video(mp_image, int(timestamp_ms))
        return results
        
    def draw_landmarks(self, frame, results):
        \"\"\"
        Draw landmarks on the frame.
        \"\"\"
        if not self.mp_draw:
            return frame # Can't draw easily without solutions API
            
        if results.hand_landmarks:
            from mediapipe.framework.formats import landmark_pb2
            for hand_lms in results.hand_landmarks:
                # Convert back to legacy format for drawing_utils
                landmark_list = landmark_pb2.NormalizedLandmarkList()
                for lm in hand_lms:
                    landmark_list.landmark.add(x=lm.x, y=lm.y, z=lm.z)
                    
                self.mp_draw.draw_landmarks(frame, landmark_list, self.mp_hands_connections)
        return frame
