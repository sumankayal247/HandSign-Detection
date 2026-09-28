from collections import deque
from statistics import mode, StatisticsError

class TemporalStabilizer:
    def __init__(self, window_size=15, min_consensus=0.7):
        self.window_size = window_size
        self.min_consensus = min_consensus
        self.buffer = deque(maxlen=window_size)
        
    def update(self, prediction):
        \"\"\"
        Adds a new prediction to the buffer and returns the stabilized letter,
        or None if no stable letter is found.
        \"\"\"
        self.buffer.append(prediction)
        
        if len(self.buffer) < self.window_size:
            return None
            
        try:
            most_common = mode(self.buffer)
        except StatisticsError:
            return None
            
        count = self.buffer.count(most_common)
        if count / self.window_size >= self.min_consensus:
            return most_common
            
        return None
        
    def clear(self):
        self.buffer.clear()

class GestureSegmenter:
    def __init__(self, cooldown_frames=15):
        self.cooldown_frames = cooldown_frames
        self.current_letter = None
        self.cooldown_counter = 0
        
    def process_stable_letter(self, stable_letter):
        \"\"\"
        Decides whether to commit a stable letter based on state.
        Returns the letter to append to text, or None.
        \"\"\"
        if stable_letter == 'UNKNOWN' or stable_letter is None:
            # Drop hand / invalid gesture resets state after cooldown
            if self.current_letter is not None:
                self.cooldown_counter += 1
                if self.cooldown_counter > self.cooldown_frames:
                    self.current_letter = None
                    self.cooldown_counter = 0
            return None
            
        if stable_letter != self.current_letter:
            # New valid letter
            self.current_letter = stable_letter
            self.cooldown_counter = 0
            return stable_letter
            
        # Same letter being held, don't repeat
        return None
