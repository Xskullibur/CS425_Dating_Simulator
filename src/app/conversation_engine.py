import os
import sys
import importlib

# Ensure project root is on sys.path so package imports work when running as a script
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the module object for the real module and register an alias so
# any code doing `from emotion_classifier import ...` (incorrect absolute import)
# will still find the correct implementation.
_ec_mod = importlib.import_module("src.utils.emotion_classifier")
sys.modules.setdefault("emotion_classifier", _ec_mod)

from typing import Optional, Dict
from datetime import datetime
import json
from src.utils.emotion_classifier import EmotionClassifier
from src.utils.affection_tracker import AffectionTracker, AffectionTrackingConfig
from src.models.persona_engine import PersonaEngine


class ConversationManager:
    """
    VN-style conversation manager with emotion detection and affection tracking.

    Uses PersonaEngine for character personalities and VN-format system prompts.
    """

    def __init__(
        self,
        character_name: str,
        model,
        tokenizer,
        tracker: Optional[AffectionTracker] = None,
        emotion_classifier: Optional[EmotionClassifier] = None,
        model_type: str = "individual",
    ):
        """
        Initialize conversation manager for VN character.

        Args:
            character_name: VN character name (Monika, Sayori, Natsuki, or Yuri)
                           In merged mode, this is cosmetic only
            model: LLM model
            tokenizer: Tokenizer
            tracker: Affection tracker (creates new if None)
            emotion_classifier: Emotion classifier (creates new if None)
            model_type: Type of model ('individual' or 'merged')
                       - 'individual': Use character-specific persona (default)
                       - 'merged': Use unified persona, character_name is cosmetic
        """
        self.character_name = character_name
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type

        # Configure tokenizer padding for LLaMA models
        if self.tokenizer is not None and self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            # Set padding side to left for generation (common for decoder-only models)
            self.tokenizer.padding_side = "left"

        # Initialize VN persona engine
        self.persona_engine = PersonaEngine(character_name, model_type=model_type)

        self.emotion_classifier = emotion_classifier or EmotionClassifier()
        self.tracker = tracker or AffectionTracker(
            character_name, emotion_classifier=self.emotion_classifier
        )

        self.conversation_history = []

        print(f"✓ Initialized VN conversation manager for {character_name}")

    def get_system_prompt(self, user_emotion: str, user_confidence: float) -> str:
        """
        Generate VN-style emotion-aware system prompt.

        Uses PersonaEngine to build system prompt in the exact format
        used during VN training.

        Args:
            user_emotion: Detected user emotion
            user_confidence: Confidence of emotion detection (not used in VN format)

        Returns:
            VN-formatted system prompt string
        """
        affection = self.tracker.affection

        # Use persona engine to build VN-style system prompt
        return self.persona_engine.build_system_prompt(
            affection=affection,
            user_emotion=user_emotion
        )


    def generate_response(self, user_input: str, verbose: bool = True) -> Dict:
        """
        Generate emotion-aware response

        Args:
            user_input: User's message
            verbose: Whether to print detailed info

        Returns:
            Dict with response and emotion/affection info
        """
        # Detect user emotion
        user_emotion_result = self.emotion_classifier.get_emotion_with_confidence(
            user_input
        )
        user_emotion = user_emotion_result["emotion"]
        user_confidence = user_emotion_result["confidence"]

        # Build messages with emotion-aware system prompt
        messages = [
            {
                "role": "system",
                "content": self.get_system_prompt(user_emotion, user_confidence),
            }
        ]

        # Add recent conversation history (last 10 turns)
        messages.extend(self.conversation_history[-10:])

        # Add current user input
        messages.append({"role": "user", "content": user_input})

        # Generate response with proper attention mask
        # Try return_dict first (newer transformers), fallback to manual tokenization
        try:
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True  # Return dict with input_ids and attention_mask
            )
            input_ids = model_inputs["input_ids"].to(self.model.device)
            attention_mask = model_inputs["attention_mask"].to(self.model.device)
        except (TypeError, KeyError):
            # Fallback for older transformers versions
            # Get formatted text first
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False  # Get text instead of tokens
            )
            # Tokenize with attention mask
            model_inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=True,
                return_attention_mask=True
            )
            input_ids = model_inputs["input_ids"].to(self.model.device)
            attention_mask = model_inputs["attention_mask"].to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            attention_mask=attention_mask,  # Pass attention mask
            max_new_tokens=256,  # FIXED: Reduced from 256 to match training data (avg ~18-20 tokens)
            num_beams=4,  # Beam search: explores 4 candidate sequences
            early_stopping=True,  # Stop when EOS token is generated (works with beam search)
            repetition_penalty=1.2,  # Increased from 1.1 to reduce repetition
            no_repeat_ngram_size=3,  # Prevent repetitive 3-grams
            length_penalty=1.0,  # Neutral length penalty (1.0 = no penalty)
            pad_token_id=self.tokenizer.pad_token_id,  # Explicitly set pad token
            eos_token_id=self.tokenizer.eos_token_id,  # CRITICAL: Stop when EOS token is generated
        )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        ).strip()

        # Update affection with both messages
        old_affection = self.tracker.affection
        new_affection, tracking_info = self.tracker.update(user_input, response)

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": response})

        # Build result
        result = {
            "response": response,
            "user_emotion": user_emotion,
            "user_confidence": user_confidence,
            "assistant_emotion": tracking_info["assistant_emotion"],
            "affection": new_affection,
            "affection_change": new_affection - old_affection,
            "affection_tier": tracking_info["affection_tier"],
            "tracking_info": tracking_info,
        }

        # Verbose output
        if verbose:
            self._print_interaction_info(result)

        return result

    def _print_interaction_info(self, result: Dict):
        """Print formatted interaction information"""
        print(f"\n{'='*70}")
        print(f"🎭 EMOTION ANALYSIS")
        print(f"{'='*70}")
        print(
            f"User emotion: {result['user_emotion']} (confidence: {result['user_confidence']:.2%})"
        )
        if result["assistant_emotion"]:
            print(f"Assistant emotion: {result['assistant_emotion']}")

        print(f"\n💝 AFFECTION UPDATE")
        print(f"{'='*70}")
        change = result["affection_change"]
        change_str = (
            f"({'+' if change > 0 else ''}{change})" if change != 0 else "(no change)"
        )
        print(
            f"Level: {result['affection']}/100 ({result['affection_tier']}) {change_str}"
        )
        print(f"Reason: {result['tracking_info']['reason']}")

        breakdown = result["tracking_info"]["breakdown"]
        if any(breakdown.values()):
            print(f"\nBreakdown:")
            if breakdown["base"] != 0:
                print(f"  - Emotion impact: {breakdown['base']:+d}")
            if breakdown["reciprocity"] != 0:
                print(f"  - Reciprocity bonus: {breakdown['reciprocity']:+d}")
            if breakdown["support"] != 0:
                print(f"  - Support bonus: {breakdown['support']:+d}")
            if breakdown["decay"] != 0:
                print(f"  - Decay: {breakdown['decay']:+d}")
        print(f"{'='*70}\n")

    def get_conversation_stats(self) -> Dict:
        """Get conversation statistics"""
        return {
            "turns": len(self.conversation_history) // 2,
            "tracker_stats": self.tracker.get_statistics(),
        }

    def save_session(self, filepath: str):
        """Save conversation session"""
        session = {
            "character": self.character_name,
            "model_type": self.model_type,
            "conversation_history": self.conversation_history,
            "statistics": self.get_conversation_stats(),
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)

        # Save tracker state separately
        tracker_path = filepath.replace(".json", "_tracker.json")
        self.tracker.save_state(tracker_path)

        print(f"✓ Session saved to {filepath}")

    @classmethod
    def load_session(
        cls,
        filepath: str,
        character_name: str,
        model,
        tokenizer,
        emotion_classifier: Optional[EmotionClassifier] = None,
        model_type: Optional[str] = None,
    ) -> "ConversationManager":
        """
        Load conversation session.

        Args:
            filepath: Path to session file
            character_name: VN character name
            model: LLM model
            tokenizer: Tokenizer
            emotion_classifier: Emotion classifier (creates new if None)
            model_type: Type of model ('individual' or 'merged').
                       If None, will try to load from session file, defaulting to 'individual'

        Returns:
            Loaded ConversationManager instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            session = json.load(f)

        # Load tracker
        tracker_path = filepath.replace(".json", "_tracker.json")
        tracker = AffectionTracker.load_state(tracker_path, emotion_classifier)

        # Determine model_type (prioritize parameter, then session file, then default)
        if model_type is None:
            model_type = session.get("model_type", "individual")

        # Create manager (persona loaded automatically by character_name)
        manager = cls(
            character_name,
            model,
            tokenizer,
            tracker,
            emotion_classifier,
            model_type,
        )

        manager.conversation_history = session["conversation_history"]

        print(f"✓ Session loaded from {filepath}")
        return manager

    def reset_conversation(self, keep_affection: bool = False):
        """Reset conversation history"""
        self.conversation_history = []

        if not keep_affection:
            self.tracker.affection = self.tracker.config.initial_affection
            self.tracker.emotion_history.clear()
            self.tracker.event_history = []
            self.tracker.emotion_counts.clear()
            self.tracker.total_messages = 0

        print(
            f"✓ Conversation reset (affection {'preserved' if keep_affection else 'reset'})"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("EMOTION-ONLY AFFECTION TRACKER - TEST")
    print("=" * 70)

    # Initialize
    emotion_classifier = EmotionClassifier()

    config = AffectionTrackingConfig(
        initial_affection=50,
        emotion_sensitivity=1.0,
        enable_reciprocity_bonus=True,
        enable_support_bonus=True,
    )

    tracker = AffectionTracker(
        character_name="Sayori", config=config, emotion_classifier=emotion_classifier
    )

    # Test conversations
    test_scenarios = [
        (
            "You look absolutely beautiful today!",
            "Aww, thank you so much! You're so sweet!",
        ),
        (
            "I'm feeling really sad about something...",
            "Oh no... I'm here for you. Want to talk about it?",
        ),
        ("This is so annoying!", "I understand, that sounds frustrating."),
        (
            "I love spending time with you",
            "I love our time together too! You're special to me.",
        ),
        ("Whatever, I don't really care", "Oh... okay then."),
    ]

    print("\n" + "=" * 70)
    print("TESTING SCENARIOS")
    print("=" * 70)

    for i, (user_msg, assistant_msg) in enumerate(test_scenarios, 1):
        print(f"\n--- Scenario {i} ---")
        print(f"User: {user_msg}")
        print(f"Assistant: {assistant_msg}")

        affection, info = tracker.update(user_msg, assistant_msg)

        print(f"\nResult:")
        print(f"  User emotion: {info['user_emotion']} ({info['user_confidence']:.2%})")
        print(f"  Assistant emotion: {info['assistant_emotion']}")
        print(f"  Affection: {info['affection']}/100 ({info['affection_tier']})")
        print(f"  Change: {info['affection_change']:+d}")
        print(f"  Reason: {info['reason']}")

    # Final statistics
    print(f"\n{'='*70}")
    print("FINAL STATISTICS")
    print(f"{'='*70}")
    stats = tracker.get_statistics()
    print(json.dumps(stats, indent=2))

    # Save test
    tracker.save_state("test_tracker.json")

    # Load test
    loaded = AffectionTracker.load_state(
        "test_tracker.json", emotion_classifier
    )
    print(f"\nLoaded affection: {loaded.affection}/100")
