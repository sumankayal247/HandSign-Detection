import pyttsx3
import threading

class TTSEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        # Optional: configure voice/rate here
        
    def speak(self, text):
        \"\"\"
        Speaks the text in a separate thread to avoid blocking the main cv2 loop.
        \"\"\"
        if not text.strip():
            return
            
        def _speak():
            # pyttsx3 can be tricky with threads, creating a new instance per thread or using OS specific hacks might be needed.
            # A safer approach for pyttsx3 threading:
            local_engine = pyttsx3.init()
            local_engine.say(text)
            local_engine.runAndWait()
            
        t = threading.Thread(target=_speak)
        t.daemon = True
        t.start()
