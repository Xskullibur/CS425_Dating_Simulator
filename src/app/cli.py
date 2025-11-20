"""
VN Dating Simulator CLI

Interactive command-line interface for the Visual Novel dating simulator.
Supports all 4 VN characters with emotion detection and affection tracking.
"""

import os
import sys
import warnings
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure project root is on sys.path so "import src.app..." works when running the script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.app.conversation_engine import ConversationManager
from src.models.persona_engine import PersonaEngine
import argparse


def get_user_name() -> str:
    """
    Prompt user for their name.

    Returns:
        User's name
    """
    print("\n" + "=" * 70)
    print("💞 WELCOME TO THE LITERATURE CLUB DATING SIMULATOR")
    print("=" * 70)
    print("\nBefore we begin, what's your name?")

    while True:
        name = input("> ").strip()
        if name:
            return name
        print("Please enter your name.")


def select_character() -> str:
    """
    Interactive character selection.

    Returns:
        Selected character name
    """
    characters = PersonaEngine.get_available_characters()

    print("\n" + "=" * 70)
    print("📚 LITERATURE CLUB - CHARACTER SELECTION")
    print("=" * 70)

    for i, char in enumerate(characters, 1):
        description = PersonaEngine.get_character_description(char)
        # Extract first sentence for preview
        preview = description.split('.')[0] if description else ""
        print(f"{i}. {char} - {preview}")

    print("\nSelect a character (1-{}) or enter name:".format(len(characters)))

    while True:
        choice = input("> ").strip()

        # Try as number
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(characters):
                return characters[idx]

        # Try as name (case insensitive)
        for char in characters:
            if choice.lower() == char.lower():
                return char

        print(f"Invalid choice. Please enter 1-{len(characters)} or character name.")


def print_banner(character_name: str, model_type: str = "individual"):
    """Print welcome banner for selected character."""
    print("\n" + "=" * 70)
    print(f"💞 VN DATING SIMULATOR - {character_name.upper()}")
    if model_type == "merged":
        print("   (Merged Model - Character name is cosmetic)")
    print("=" * 70)
    print("\nCommands:")
    print("  /status  - Show affection level and statistics")
    print("  /reset   - Reset conversation (keep affection)")
    print("  /restart - Reset everything (clear affection)")
    print("  quit     - Exit the simulator")
    print("=" * 70 + "\n")


def print_status(engine: ConversationManager):
    """Print current affection status and statistics."""
    affection = engine.tracker.affection
    tier = engine.tracker.get_affection_tier()
    stats = engine.get_conversation_stats()

    print("\n" + "=" * 70)
    print("💝 RELATIONSHIP STATUS")
    print("=" * 70)
    print(f"Character: {engine.character_name}")
    print(f"Affection: {affection}/100 ({tier})")
    print(f"Conversation turns: {stats['turns']}")

    tracker_stats = stats['tracker_stats']
    if 'emotion_distribution' in tracker_stats:
        print("\n📊 Top emotions:")
        for emotion, count in list(tracker_stats['emotion_distribution'].items())[:5]:
            print(f"  - {emotion}: {count}")

    print("=" * 70 + "\n")


def main(character: str = None, model_path: str = None, model_type: str = "individual", user_name: str = None):
    """
    Run the VN dating simulator CLI.

    Args:
        character: Character name (prompts if None)
        model_path: Path to trained model checkpoint (uses dummy if None)
        model_type: Type of model ('individual' or 'merged')
                   - 'individual': Use character-specific persona (default)
                   - 'merged': Use unified persona, character is cosmetic
        user_name: User's name (prompts if None)
    """
    # Validate model_type
    if model_type not in ['individual', 'merged']:
        print(f"Error: Invalid model_type '{model_type}'")
        print("Must be 'individual' or 'merged'")
        return

    # Get user name
    if user_name is None:
        user_name = get_user_name()

    # Character selection
    if character is None:
        character = select_character()
    elif model_type == "individual" and character not in PersonaEngine.get_available_characters():
        print(f"Error: Unknown character '{character}'")
        print(f"Available: {', '.join(PersonaEngine.get_available_characters())}")
        return
    elif model_type == "merged" and character not in PersonaEngine.get_available_characters():
        # For merged mode, any character name is valid (cosmetic), but warn if not standard
        print(f"⚠ Note: '{character}' is not a standard character, but that's OK in merged mode (cosmetic name)")
        print(f"   Standard characters: {', '.join(PersonaEngine.get_available_characters())}")

    # Model loading
    model = None
    tokenizer = None

    if model_path:
        print(f"\n⏳ Loading model...")
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            import torch
            from pathlib import Path

            # Check if this is a LoRA adapter checkpoint (has adapter_model.safetensors)
            adapter_file = Path(model_path) / "adapter_model.safetensors"

            # Suppress output during model loading
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                if adapter_file.exists():
                    # LoRA adapter - load base model + adapter
                    BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

                    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
                    base_model = AutoModelForCausalLM.from_pretrained(
                        BASE_MODEL,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )

                    model = PeftModel.from_pretrained(base_model, model_path)
                else:
                    # Full model checkpoint
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )

            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            print("Running in demo mode (no responses will be generated)")
    else:
        print("\n⚠ No model path provided. Running in demo mode.")
        print("  To load a model, use: --model path/to/checkpoint")

    # Initialize conversation engine (suppress initialization message)
    with redirect_stdout(StringIO()):
        engine = ConversationManager(
            character_name=character,
            model=model,
            tokenizer=tokenizer,
            model_type=model_type
        )

    # Print banner
    print_banner(character, model_type=model_type)

    # Show initial status
    print(f"✨ Starting conversation with {character}")
    print(f"💝 Affection: {engine.tracker.affection}/100 ({engine.tracker.get_affection_tier()})\n")

    # Main conversation loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["quit", "exit", "q"]:
                print(f"\n👋 Goodbye! Final affection with {character}: {engine.tracker.affection}/100")
                break

            elif user_input.lower() == "/status":
                print_status(engine)
                continue

            elif user_input.lower() == "/reset":
                engine.reset_conversation(keep_affection=True)
                print("🔄 Conversation reset! (Affection preserved)\n")
                continue

            elif user_input.lower() == "/restart":
                engine.reset_conversation(keep_affection=False)
                print("🔄 Everything reset! Starting fresh.\n")
                continue

            elif user_input.startswith("/"):
                print(f"❌ Unknown command: {user_input}")
                print("Available commands: /status, /reset, /restart, quit")
                continue

            # Generate response (if model is loaded)
            if model and tokenizer:
                result = engine.generate_response(user_input, verbose=False)

                # Replace <USER> with actual name in response
                response = result['response'].replace("<USER>", user_name)

                # Print response
                print(f"\n{character}: {response}")

                # Print affection update (compact)
                affection = result['affection']
                tier = result['affection_tier']
                change = result['affection_change']

                if change != 0:
                    change_str = f"({'+' if change > 0 else ''}{change})"
                    print(f"💝 {affection}/100 ({tier}) {change_str}")
                else:
                    print(f"💝 {affection}/100 ({tier})")

                print()  # Blank line for readability

            else:
                # Demo mode - just show emotion detection
                user_emotion = engine.emotion_classifier.get_primary_emotion(user_input)
                print(f"\n[Demo Mode] Detected emotion: {user_emotion}")
                print(f"[Demo Mode] Affection: {engine.tracker.affection}/100")
                print("\n(Load a model with --model to get actual responses)\n")

        except KeyboardInterrupt:
            print(f"\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VN Dating Simulator - Interactive CLI"
    )
    parser.add_argument(
        "--name", "-n",
        default=None,
        help="Your name (prompts if not provided)"
    )
    parser.add_argument(
        "--character", "-c",
        default=None,
        help="Character to talk with (Monika, Sayori, Natsuki, Yuri). Prompts if not provided."
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to trained model checkpoint (e.g., checkpoints/vn_model)"
    )
    parser.add_argument(
        "--model-type", "-t",
        default="individual",
        choices=["individual", "merged"],
        help="Model type: 'individual' (character-specific, default) or 'merged' (unified, character is cosmetic)"
    )

    args = parser.parse_args()
    main(character=args.character, model_path=args.model, model_type=args.model_type, user_name=args.name)
