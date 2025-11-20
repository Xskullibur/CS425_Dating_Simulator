"""
Emotion-Only Affection Tracking System
Uses only EmoBERTa emotion classification for affection tracking
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import deque, Counter

from emotion_classifier import EmotionClassifier, EmotionUtils

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AffectionTrackingConfig:
    """Configuration for emotion-based affection tracking"""
    initial_affection: int = 50
    min_affection: int = 0
    max_affection: int = 100
    
    # Emotion sensitivity (multiplier for all emotion impacts)
    emotion_sensitivity: float = 1.0
    
    # Context bonuses
    enable_reciprocity_bonus: bool = True  # Bonus for matching emotions
    enable_support_bonus: bool = True      # Bonus for emotional support
    
    # Memory
    emotion_memory_size: int = 10  # How many emotions to remember
    
    # Decay
    decay_enabled: bool = False
    decay_per_message: float = 0.0  # Affection decay per message

@dataclass
class AffectionEvent:
    """Record of an emotion-based affection event"""
    timestamp: str
    user_emotion: str
    user_confidence: float
    assistant_emotion: Optional[str]
    affection_change: int
    new_affection: int
    reason: str

# ============================================================================
# EMOTION-ONLY AFFECTION TRACKER
# ============================================================================

class AffectionTracker:
    """
    Pure emotion-based affection tracking
    Uses ONLY EmoBERTa classifications, no keyword matching
    """
    
    def __init__(self,
                 character_name: str,
                 config: Optional[AffectionTrackingConfig] = None,
                 emotion_classifier: Optional[EmotionClassifier] = None):
        """
        Initialize emotion-only tracker
        
        Args:
            character_name: Character being tracked
            config: Tracking configuration
            emotion_classifier: EmoBERTa classifier instance
        """
        self.character_name = character_name
        self.config = config or AffectionTrackingConfig()
        self.emotion_classifier = emotion_classifier or EmotionClassifier()
        
        # State
        self.affection = self.config.initial_affection
        self.emotion_history = deque(maxlen=self.config.emotion_memory_size)
        self.event_history: List[AffectionEvent] = []
        
        # Statistics
        self.total_messages = 0
        self.emotion_counts = Counter()
        
        print(f"✓ Initialized emotion-only tracker for {character_name}")
    
    def update(self,
               user_message: str,
               assistant_response: Optional[str] = None) -> Tuple[int, Dict]:
        """
        Update affection based purely on emotion classification
        
        Args:
            user_message: User's message
            assistant_response: Assistant's response (optional, for reciprocity)
            
        Returns:
            Tuple of (new_affection_level, emotion_info_dict)
        """
        self.total_messages += 1
        
        # === 1. Classify user emotion ===
        user_emotion_result = self.emotion_classifier.get_emotion_with_confidence(user_message)
        user_emotion = user_emotion_result['emotion']
        user_confidence = user_emotion_result['confidence']
        
        # Track emotion
        self.emotion_counts[user_emotion] += 1
        
        # === 2. Calculate base affection change from user emotion ===
        base_change = self._calculate_emotion_impact(user_emotion, user_confidence)
        
        # === 3. Calculate bonuses ===
        reciprocity_bonus = 0
        support_bonus = 0
        assistant_emotion = None
        
        if assistant_response:
            assistant_emotion_result = self.emotion_classifier.get_emotion_with_confidence(assistant_response)
            assistant_emotion = assistant_emotion_result['emotion']
            
            if self.config.enable_reciprocity_bonus:
                reciprocity_bonus = self._calculate_reciprocity_bonus(
                    user_emotion,
                    assistant_emotion
                )
            
            if self.config.enable_support_bonus:
                support_bonus = self._calculate_support_bonus(
                    user_emotion,
                    assistant_emotion
                )
        
        # === 4. Apply decay ===
        decay = 0
        if self.config.decay_enabled and self.config.decay_per_message > 0:
            if self.affection > 50:
                decay = -int(self.config.decay_per_message)
        
        # === 5. Calculate total change ===
        total_change = base_change + reciprocity_bonus + support_bonus + decay
        
        # Update affection
        old_affection = self.affection
        self.affection += total_change
        self.affection = max(self.config.min_affection,
                            min(self.config.max_affection, self.affection))
        
        # === 6. Build reason string ===
        reason_parts = []
        if base_change != 0:
            reason_parts.append(f"{user_emotion}({'+' if base_change > 0 else ''}{base_change})")
        if reciprocity_bonus != 0:
            reason_parts.append(f"reciprocity(+{reciprocity_bonus})")
        if support_bonus != 0:
            reason_parts.append(f"support(+{support_bonus})")
        if decay != 0:
            reason_parts.append(f"decay({decay})")
        
        reason = ", ".join(reason_parts) if reason_parts else "no_change"
        
        # === 7. Record event ===
        event = AffectionEvent(
            timestamp=datetime.now().isoformat(),
            user_emotion=user_emotion,
            user_confidence=user_confidence,
            assistant_emotion=assistant_emotion,
            affection_change=total_change,
            new_affection=self.affection,
            reason=reason
        )
        
        self.event_history.append(event)
        self.emotion_history.append({
            "user": user_emotion,
            "assistant": assistant_emotion,
            "affection": self.affection
        })
        
        # === 8. Build info dict ===
        info = {
            "user_emotion": user_emotion,
            "user_confidence": user_confidence,
            "assistant_emotion": assistant_emotion,
            "affection_change": total_change,
            "affection": self.affection,
            "affection_tier": self.get_affection_tier(),
            "reason": reason,
            "breakdown": {
                "base": base_change,
                "reciprocity": reciprocity_bonus,
                "support": support_bonus,
                "decay": decay
            }
        }
        
        return self.affection, info
    
    def _calculate_emotion_impact(self, emotion, confidence) -> int:
        """
        Calculate affection change from emotion classification
        Pure emotion -> affection mapping
        """
        base_weight = EmotionUtils.AFFECTION_WEIGHTS.get(emotion, 0)
        
        # Apply confidence and sensitivity
        impact = base_weight * confidence * self.config.emotion_sensitivity
        
        return int(impact)
    
    def _calculate_reciprocity_bonus(self,
                                     user_emotion,
                                     assistant_emotion) -> int:
        """
        Bonus for emotional reciprocity
        When emotions match or complement well
        """
        bonus = 0
        
        # Same emotion = good reciprocity
        if user_emotion == assistant_emotion:
            bonus = 4
        
        # Both positive = great reciprocity
        elif (user_emotion in EmotionUtils.POSITIVE_EMOTIONS and
              assistant_emotion in EmotionUtils.POSITIVE_EMOTIONS):
            bonus = 3
        
        # Complementary emotions
        complementary_pairs = [
            ({'curiosity', 'confusion'}, {'realization', 'approval'}),
            ({'sadness', 'grief', 'disappointment'}, {'caring', 'optimism'}),
            ({'fear', 'nervousness'}, {'caring', 'approval', 'relief'}),
            ({'excitement', 'joy'}, {'amusement', 'joy', 'excitement'}),
        ]
        
        for user_set, assistant_set in complementary_pairs:
            if user_emotion in user_set and assistant_emotion in assistant_set:
                bonus = 5
                break
        
        return bonus
    
    def _calculate_support_bonus(self,
                                 user_emotion,
                                 assistant_emotion) -> int:
        """
        Bonus for providing emotional support
        When user has negative emotion and assistant responds supportively
        """
        bonus = 0
        
        # User is experiencing negative emotion
        if user_emotion in EmotionUtils.NEGATIVE_EMOTIONS:
            # Assistant responds with supportive emotions
            supportive_emotions = {
                'caring', 'optimism', 'approval', 'love',
                'gratitude', 'relief', 'amusement'
            }
            
            if assistant_emotion in supportive_emotions:
                # Big bonus for emotional support
                bonus = 8
                
                # Extra bonus for particularly caring responses
                if assistant_emotion == 'caring':
                    bonus += 2
        
        return bonus
    
    # === QUERY METHODS ===
    
    def get_affection_tier(self) -> str:
        """Get descriptive affection tier"""
        if self.affection < 20:
            return "hostile"
        elif self.affection < 40:
            return "cold"
        elif self.affection < 60:
            return "neutral"
        elif self.affection < 80:
            return "friendly"
        elif self.affection < 95:
            return "affectionate"
        else:
            return "loving"
    
    def get_dominant_user_emotions(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """Get most common user emotions"""
        return self.emotion_counts.most_common(top_k)
    
    def get_recent_trend(self, n: int = 5) -> str:
        """Get recent affection trend"""
        if len(self.event_history) < n:
            return "insufficient_data"
        
        recent = self.event_history[-n:]
        total_change = sum(event.affection_change for event in recent)
        
        if total_change > 5:
            return "improving"
        elif total_change < -5:
            return "declining"
        else:
            return "stable"
    
    def get_emotion_distribution(self) -> Dict[str, float]:
        """Get emotion distribution percentages"""
        if self.total_messages == 0:
            return {}
        
        return {
            emotion: (count / self.total_messages) * 100
            for emotion, count in self.emotion_counts.items()
        }
    
    def get_emotional_profile(self) -> Dict:
        """Get user's emotional profile"""
        emotions = [e['user'] for e in self.emotion_history if e['user']]
        
        if not emotions:
            return {
                "profile": "insufficient_data",
                "positive_ratio": 0,
                "negative_ratio": 0,
                "neutral_ratio": 0
            }
        
        positive = sum(1 for e in emotions if e in EmotionUtils.POSITIVE_EMOTIONS)
        negative = sum(1 for e in emotions if e in EmotionUtils.NEGATIVE_EMOTIONS)
        neutral = len(emotions) - positive - negative
        
        total = len(emotions)
        
        # Determine profile
        pos_ratio = positive / total
        neg_ratio = negative / total
        
        if pos_ratio > 0.6:
            profile = "optimistic"
        elif neg_ratio > 0.4:
            profile = "troubled"
        elif pos_ratio > 0.4 and neg_ratio < 0.2:
            profile = "balanced_positive"
        elif neg_ratio > 0.3:
            profile = "balanced_negative"
        else:
            profile = "neutral"
        
        return {
            "profile": profile,
            "positive_ratio": pos_ratio,
            "negative_ratio": neg_ratio,
            "neutral_ratio": neutral / total,
            "dominant_emotions": self.get_dominant_user_emotions(3),
            "emotional_range": len(set(emotions))
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        emotional_profile = self.get_emotional_profile()
        
        stats = {
            "character": self.character_name,
            "current_affection": self.affection,
            "affection_tier": self.get_affection_tier(),
            "total_messages": self.total_messages,
            "emotional_profile": emotional_profile,
            "recent_trend": self.get_recent_trend(),
            "top_emotions": self.get_dominant_user_emotions(5),
        }
        
        if self.event_history:
            stats["highest_affection"] = max(e.new_affection for e in self.event_history)
            stats["lowest_affection"] = min(e.new_affection for e in self.event_history)
            stats["average_change"] = sum(e.affection_change for e in self.event_history) / len(self.event_history)
        
        return stats
    
    # === PERSISTENCE ===
    
    def save_state(self, filepath: str):
        """Save tracker state"""
        state = {
            "character_name": self.character_name,
            "config": asdict(self.config),
            "affection": self.affection,
            "total_messages": self.total_messages,
            "emotion_counts": dict(self.emotion_counts),
            "event_history": [asdict(event) for event in self.event_history],
            "statistics": self.get_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        print(f"✓ Saved tracker state to {filepath}")
    
    @classmethod
    def load_state(cls, filepath: str, emotion_classifier: Optional[EmotionClassifier] = None) -> 'AffectionTracker':
        """Load tracker state"""
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        config = AffectionTrackingConfig(**state['config'])
        tracker = cls(state['character_name'], config, emotion_classifier)
        
        tracker.affection = state['affection']
        tracker.total_messages = state['total_messages']
        tracker.emotion_counts = Counter(state['emotion_counts'])
        tracker.event_history = [AffectionEvent(**event) for event in state['event_history']]
        
        print(f"✓ Loaded tracker state from {filepath}")
        return tracker