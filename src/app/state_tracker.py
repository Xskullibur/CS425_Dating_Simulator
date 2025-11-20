class StateTracker:
    def __init__(self):
        self.affection = 0
        self.mood = "neutral"

    def update(self, emotion):
        if emotion in ["joy", "love", "admiration"]:
            self.affection += 2
            self.mood = "positive"
        elif emotion in ["anger", "disgust", "fear"]:
            self.affection -= 2
            self.mood = "defensive"
        elif emotion == "sadness":
            self.affection += 1
            self.mood = "empathetic"
        return self.affection, self.mood

    def reset(self):
        self.affection = 0
        self.mood = "neutral"
