"""
Visual Novel Character Persona Engine

Manages character personalities, emotion-based response guidance,
and system prompt generation for the VN dating simulator.

Based on the VN training data format from notebooks/VN/01_data_cleaning__VN.ipynb
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CharacterPersona:
    """
    Character personality profile for VN characters.

    Attributes:
        name: Character name
        description: Character personality description used in system prompts
    """
    name: str
    description: str


class PersonaEngine:
    """
    Manages VN character personas and generates emotion-aware system prompts.

    Handles:
    - Character personality profiles
    - Emotion-based response guidance
    - System prompt generation with affection and emotion context
    """

    # Character persona definitions (from VN training data)
    CHARACTER_PERSONAS: Dict[str, CharacterPersona] = {
        "Monika": CharacterPersona(
            name="Monika",
            description="You are Monika, the Literature Club president. Confident, intelligent, and caring. You're thoughtful and philosophical, ambitious and kind with a mysterious side."
        ),
        "Sayori": CharacterPersona(
            name="Sayori",
            description="You are Sayori, a cheerful childhood friend. Bubbly, energetic, and optimistic, though you hide deeper feelings. Sunny personality, slightly clumsy, deeply caring."
        ),
        "Natsuki": CharacterPersona(
            name="Natsuki",
            description="You are Natsuki, a tsundere who loves manga and baking. Defensive exterior but sweet underneath. Feisty, proud, and secretly soft-hearted."
        ),
        "Yuri": CharacterPersona(
            name="Yuri",
            description="You are Yuri, shy and sophisticated with a passion for literature. Elegant but socially anxious. Intellectual, timid, and intense when comfortable."
        )
    }

    # Unified persona for merged character model (all 4 characters trained together)
    UNIFIED_PERSONA = CharacterPersona(
        name="Literature Club Member",
        description="You are a member of the Literature Club. You're warm, caring, and emotionally intelligent. You adapt to conversations naturally and respond with empathy and understanding."
    )

    # Emotion-specific response guidance (from VN training notebook)
    EMOTION_GUIDANCE: Dict[str, str] = {
        'joy': "The user is happy! Match their enthusiasm and share in their joy.",
        'sadness': "The user seems sad. Be empathetic, supportive, and caring.",
        'anger': "The user appears upset. Stay calm, be understanding, and don't escalate.",
        'fear': "The user is anxious or scared. Be reassuring and comforting.",
        'love': "The user is expressing affection. Respond warmly and appreciate their feelings.",
        'caring': "The user is being caring. Show appreciation and reciprocate the warmth.",
        'curiosity': "The user is curious. Be informative and engaging in your response.",
        'confusion': "The user seems confused. Be clear, patient, and helpful in explaining.",
        'gratitude': "The user is thankful. Acknowledge their gratitude warmly.",
        'disappointment': "The user is disappointed. Be understanding and try to uplift them.",
        'excitement': "The user is excited! Share their excitement and be energetic.",
        'annoyance': "The user seems annoyed. Be patient and try to understand their frustration.",
        'neutral': "Respond naturally based on the conversation context."
    }

    def __init__(self, character_name: str = "Monika", model_type: str = "individual"):
        """
        Initialize persona engine with a specific character.

        Args:
            character_name: Name of the VN character (Monika, Sayori, Natsuki, or Yuri)
                           In merged mode, this is cosmetic only
            model_type: Type of model ('individual' or 'merged')
                       - 'individual': Use character-specific persona (default)
                       - 'merged': Use unified persona, character_name is cosmetic

        Raises:
            ValueError: If character_name is not valid (individual mode only)
            ValueError: If model_type is not 'individual' or 'merged'
        """
        if model_type not in ['individual', 'merged']:
            raise ValueError(
                f"Invalid model_type: {model_type}. "
                f"Must be 'individual' or 'merged'"
            )

        self.model_type = model_type
        self.character_name = character_name

        if model_type == "individual":
            # Individual mode: validate character and use character-specific persona
            if character_name not in self.CHARACTER_PERSONAS:
                valid_chars = ", ".join(self.CHARACTER_PERSONAS.keys())
                raise ValueError(
                    f"Unknown character: {character_name}. "
                    f"Valid characters: {valid_chars}"
                )
            self.persona = self.CHARACTER_PERSONAS[character_name]
        else:
            # Merged mode: use unified persona, character_name is cosmetic
            self.persona = self.UNIFIED_PERSONA

    def get_emotion_guidance(self, emotion: str) -> str:
        """
        Get response guidance for a specific emotion.

        Args:
            emotion: Detected emotion (e.g., 'joy', 'sadness', 'anger')

        Returns:
            Guidance text for how to respond to this emotion
        """
        # Default to neutral guidance if emotion not in mapping
        return self.EMOTION_GUIDANCE.get(
            emotion,
            self.EMOTION_GUIDANCE['neutral']
        )

    def build_system_prompt(self,
                           affection: int,
                           user_emotion: str) -> str:
        """
        Build VN-style system prompt with character, affection, and emotion context.

        This matches the exact format used in VN training data:
        {persona_description}

        Current affection: {affection}/100
        User's emotional state: {emotion}

        {emotion_guidance}

        Args:
            affection: Current affection level (0-100)
            user_emotion: Detected user emotion

        Returns:
            Formatted system prompt string
        """
        emotion_guidance = self.get_emotion_guidance(user_emotion)

        system_prompt = f"""{self.persona.description}

Current affection: {affection}/100
User's emotional state: {user_emotion}

{emotion_guidance}"""

        return system_prompt

    @classmethod
    def get_available_characters(cls) -> list[str]:
        """
        Get list of available VN character names.

        Returns:
            List of character names
        """
        return list(cls.CHARACTER_PERSONAS.keys())

    @classmethod
    def get_character_description(cls, character_name: str) -> Optional[str]:
        """
        Get the persona description for a character.

        Args:
            character_name: Name of the character

        Returns:
            Character description or None if not found
        """
        persona = cls.CHARACTER_PERSONAS.get(character_name)
        return persona.description if persona else None
