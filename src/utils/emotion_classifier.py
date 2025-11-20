# emotion_classifier.py

from typing import Dict, List, Optional, Union
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


class EmotionClassifier:
    """
    Emotion classifier using EmoBERTa
    Detects: admiration, amusement, anger, annoyance, approval, caring, confusion,
             curiosity, desire, disappointment, disapproval, disgust, embarrassment,
             excitement, fear, gratitude, grief, joy, love, nervousness, optimism,
             pride, realization, relief, remorse, sadness, surprise, neutral
    """

    def __init__(self, model_name: str = "tae898/emoberta-base", device: str = None):
        """
        Initialize emotion classifier

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"Loading emotion classifier on {device}...")

        # Suppress sequential pipeline warning (expected for chat applications)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*pipelines sequentially.*")
            self.classifier = pipeline(
                "text-classification",
                model=model_name,
                top_k=3,
                device=0 if device == "cuda" else -1,
            )

        print("✓ Emotion classifier loaded")

    def classify(self, text: str, top_k: int = 3) -> List[Dict[str, Union[str, float]]]:
        """
        Classify emotion in text

        Args:
            text: Text to classify
            top_k: Number of top emotions to return

        Returns:
            List of dicts with 'label' and 'score'
        """
        if not text or not text.strip():
            return [{"label": "neutral", "score": 1.0}]

        # Truncate long text
        text = text[:512]

        try:
            results = self.classifier(text)[0]  # Returns list of all emotions

            # Sort by score and get top_k
            results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
            return results_sorted[:top_k]

        except Exception as e:
            print(f"Emotion classification error: {e}")
            return [{"label": "neutral", "score": 1.0}]

    def get_primary_emotion(self, text: str):
        """Get the primary (highest scoring) emotion"""
        results = self.classify(text, top_k=1)
        return results[0]["label"] if results else "neutral"

    def get_emotion_with_confidence(self, text: str):
        """Get primary emotion with confidence score"""
        results = self.classify(text, top_k=1)
        if results:
            return {"emotion": results[0]["label"], "confidence": results[0]["score"]}
        return {"emotion": "neutral", "confidence": 1.0}

    def classify_batch(self, texts: List[str], top_k: int = 3) -> List[List[Dict]]:
        """Classify multiple texts efficiently"""
        results = []
        for text in texts:
            results.append(self.classify(text, top_k))
        return results


# ============================================================================
# EMOTION UTILITIES
# ============================================================================


class EmotionUtils:
    """Utilities for working with emotions"""

    # Emotion groupings for affection tracking
    POSITIVE_EMOTIONS = {
        "admiration",
        "amusement",
        "approval",
        "caring",
        "desire",
        "excitement",
        "gratitude",
        "joy",
        "love",
        "optimism",
        "pride",
        "relief",
    }

    NEGATIVE_EMOTIONS = {
        "anger",
        "annoyance",
        "disappointment",
        "disapproval",
        "disgust",
        "embarrassment",
        "fear",
        "grief",
        "nervousness",
        "remorse",
        "sadness",
    }

    NEUTRAL_EMOTIONS = {"confusion", "curiosity", "realization", "surprise", "neutral"}

    # Affection impact weights
    AFFECTION_WEIGHTS = {
        # Strong positive
        "love": 12,
        "caring": 10,
        "admiration": 8,
        "gratitude": 8,
        "joy": 6,
        "excitement": 5,
        "amusement": 4,
        "approval": 4,
        "desire": 7,
        "optimism": 3,
        "pride": 3,
        "relief": 2,
        # Strong negative
        "anger": -12,
        "disgust": -15,
        "disapproval": -8,
        "grief": -5,
        "sadness": -3,
        # Medium negative
        "annoyance": -6,
        "disappointment": -5,
        "fear": -4,
        "embarrassment": -2,
        "nervousness": -1,
        "remorse": 2,  # Actually positive (showing care)
        # Neutral/Contextual
        "confusion": 1,  # Slight positive (asking for help)
        "curiosity": 3,  # Positive (showing interest)
        "surprise": 0,
        "realization": 0,
        "neutral": 0,
    }

    @staticmethod
    def get_emotion_group(emotion: str) -> str:
        """Get emotion group (positive/negative/neutral)"""
        if emotion in EmotionUtils.POSITIVE_EMOTIONS:
            return "positive"
        elif emotion in EmotionUtils.NEGATIVE_EMOTIONS:
            return "negative"
        else:
            return "neutral"

    @staticmethod
    def get_affection_impact(emotion: str, confidence: float = 1.0) -> int:
        """Calculate affection impact from emotion"""
        base_weight = EmotionUtils.AFFECTION_WEIGHTS.get(emotion, 0)
        return int(base_weight * confidence)

    @staticmethod
    def format_emotion_string(emotions: List[Dict]) -> str:
        """Format emotions for display/logging"""
        return ", ".join([f"{e['label']}({e['score']:.2f})" for e in emotions])
