from collections import deque

class ContextManager:
    def __init__(self, max_turns=5):
        self.history = deque(maxlen=max_turns)

    def add_turn(self, text: str, speaker: str):
        self.history.append({"speaker": speaker, "text": text})

    def get_recent_context(self):
        return list(self.history)
    
    def format_context(self):
        return "\n".join([f"{t['speaker']}: {t['text']}" for t in self.history])

    def reset(self):
        self.history.clear()

