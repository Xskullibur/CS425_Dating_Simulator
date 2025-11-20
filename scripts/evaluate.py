"""
Command-line interface for evaluating dialogue models.

This script loads trained models and runs comprehensive evaluation
on test datasets or generated responses.
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.dialogue_evaluator import DialogueEvaluator


def load_model(checkpoint_path: str, model_type: str = 'dialogue', device: str = 'cpu'):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model ('dialogue' or 'emotion')
        device: Device to load model on

    Returns:
        Loaded model and tokenizer
    """
    print(f"Loading {model_type} model from {checkpoint_path}...")

    checkpoint_path = Path(checkpoint_path)

    if model_type == 'dialogue':
        # Load dialogue generation model
        # Adjust based on your model architecture
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    elif model_type == 'emotion':
        # Load emotion classification model
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)
    print(f"Model loaded successfully on {device}")

    return model, tokenizer


def load_test_conversations(data_path: str) -> list:
    """
    Load test conversations from JSON file.

    Expected format:
    [
        [
            {"user": "Hi!", "bot": "Hello!"},
            {"user": "How are you?", "bot": "I'm great!"}
        ],
        ...
    ]

    Args:
        data_path: Path to test data JSON file

    Returns:
        List of conversations
    """
    print(f"Loading test data from {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} conversations")
    return data


def load_test_contexts(data_path: str) -> list:
    """
    Load test contexts for generation.

    Expected format (JSON):
    [
        "Context 1",
        "Context 2",
        ...
    ]

    Or plain text file (one context per line)

    Args:
        data_path: Path to test contexts file

    Returns:
        List of context strings
    """
    print(f"Loading test contexts from {data_path}...")

    data_path = Path(data_path)

    if data_path.suffix == '.json':
        with open(data_path, 'r', encoding='utf-8') as f:
            contexts = json.load(f)
    else:
        # Plain text, one per line
        with open(data_path, 'r', encoding='utf-8') as f:
            contexts = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(contexts)} contexts")
    return contexts


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dialogue generation models"
    )

    # Model arguments
    parser.add_argument(
        '--dialogue-checkpoint',
        type=str,
        required=True,
        help='Path to dialogue model checkpoint'
    )
    parser.add_argument(
        '--emotion-checkpoint',
        type=str,
        default=None,
        help='Path to emotion detector checkpoint (optional)'
    )

    # Data arguments
    parser.add_argument(
        '--eval-mode',
        type=str,
        choices=['conversations', 'generation'],
        default='generation',
        help='Evaluation mode: evaluate existing conversations or generate and evaluate'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to test data (conversations JSON or contexts file)'
    )

    # Generation arguments
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum generation length'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Generation temperature'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Nucleus sampling top-p'
    )

    # Output arguments
    parser.add_argument(
        '--output-path',
        type=str,
        default='results/evaluation_results.json',
        help='Path to save evaluation results'
    )

    # Other arguments
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run evaluation on'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for evaluation (currently unused)'
    )
    parser.add_argument(
        '--no-emotion',
        action='store_true',
        help='Disable emotion consistency evaluation'
    )
    parser.add_argument(
        '--no-affection',
        action='store_true',
        help='Disable affection tracking'
    )

    args = parser.parse_args()

    # Load dialogue model
    dialogue_model, dialogue_tokenizer = load_model(
        args.dialogue_checkpoint,
        model_type='dialogue',
        device=args.device
    )

    # Load emotion detector if provided
    emotion_detector = None
    if args.emotion_checkpoint and not args.no_emotion:
        emotion_detector, _ = load_model(
            args.emotion_checkpoint,
            model_type='emotion',
            device=args.device
        )

    # Initialize evaluator
    print("\nInitializing evaluator...")
    evaluator = DialogueEvaluator(
        dialogue_model=dialogue_model,
        tokenizer=dialogue_tokenizer,
        emotion_detector=emotion_detector,
        device=args.device
    )

    # Run evaluation
    print(f"\nRunning evaluation in '{args.eval_mode}' mode...")

    if args.eval_mode == 'conversations':
        # Evaluate existing conversations
        conversations = load_test_conversations(args.data_path)
        results = evaluator.evaluate_dataset(
            conversations,
            batch_size=args.batch_size,
            verbose=True
        )
    else:
        # Generate and evaluate
        contexts = load_test_contexts(args.data_path)
        generate_kwargs = {
            'max_length': args.max_length,
            'temperature': args.temperature,
            'top_p': args.top_p
        }
        results = evaluator.evaluate_from_generations(
            contexts,
            generate_kwargs=generate_kwargs,
            verbose=True
        )

    # Print summary
    evaluator.print_summary(results)

    # Save results
    evaluator.save_results(results, args.output_path)

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
