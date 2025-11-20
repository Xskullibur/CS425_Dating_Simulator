"""
Character Consistency Evaluation

Metrics to detect character bleeding in multi-character dialogue models.
Evaluates if each character maintains distinct personality and doesn't mention wrong character names.
"""

import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np


class CharacterConsistencyEvaluator:
    """
    Evaluates character consistency in generated responses.

    Detects:
    - Character name bleeding (wrong character names mentioned)
    - Personality trait consistency
    - Style consistency across responses
    """

    def __init__(self, character_names: List[str]):
        """
        Args:
            character_names: List of valid character names in the system
        """
        self.character_names = character_names

        # Character-specific keywords (expand based on your characters)
        self.character_traits = {
            "Monika": ["confident", "intelligent", "president", "philosophical", "ambitious"],
            "Sayori": ["cheerful", "optimistic", "happy", "sunny", "energetic"],
            "Natsuki": ["tsundere", "manga", "baking", "cute", "fierce"],
            "Yuri": ["shy", "intellectual", "literature", "elegant", "timid"],
        }

    def detect_name_bleeding(self, character: str, response: str) -> Dict[str, any]:
        """
        Detect if response mentions wrong character names.

        Args:
            character: The character who should be speaking
            response: Generated response text

        Returns:
            Dict with bleeding detection results
        """
        other_characters = [c for c in self.character_names if c != character]

        mentioned_chars = []
        for other_char in other_characters:
            # Case-insensitive search for character names
            pattern = r'\b' + re.escape(other_char) + r'\b'
            if re.search(pattern, response, re.IGNORECASE):
                mentioned_chars.append(other_char)

        has_bleeding = len(mentioned_chars) > 0

        return {
            "has_name_bleeding": has_bleeding,
            "wrong_characters_mentioned": mentioned_chars,
            "num_wrong_mentions": len(mentioned_chars),
            "response_length": len(response.split())
        }

    def check_trait_consistency(self, character: str, response: str) -> Dict[str, float]:
        """
        Check if response aligns with character's personality traits.

        Args:
            character: The character speaking
            response: Generated response text

        Returns:
            Dict with trait consistency scores
        """
        if character not in self.character_traits:
            return {"trait_score": 0.0, "traits_found": []}

        expected_traits = self.character_traits[character]
        response_lower = response.lower()

        # Count how many expected traits appear (loose match)
        traits_found = []
        for trait in expected_traits:
            if trait.lower() in response_lower:
                traits_found.append(trait)

        trait_score = len(traits_found) / len(expected_traits) if expected_traits else 0.0

        return {
            "trait_score": trait_score,
            "traits_found": traits_found,
            "expected_traits": expected_traits
        }

    def evaluate_response(self, character: str, response: str) -> Dict[str, any]:
        """
        Complete evaluation of a single response.

        Args:
            character: The character speaking
            response: Generated response text

        Returns:
            Dict with all consistency metrics
        """
        name_bleeding = self.detect_name_bleeding(character, response)
        trait_consistency = self.check_trait_consistency(character, response)

        return {
            "character": character,
            "response": response,
            **name_bleeding,
            **trait_consistency,
            "is_consistent": not name_bleeding["has_name_bleeding"]
        }

    def evaluate_batch(self,
                      characters: List[str],
                      responses: List[str]) -> Dict[str, any]:
        """
        Evaluate multiple responses and compute aggregate metrics.

        Args:
            characters: List of character names (parallel to responses)
            responses: List of generated responses

        Returns:
            Dict with aggregate metrics
        """
        results = []
        for char, resp in zip(characters, responses):
            results.append(self.evaluate_response(char, resp))

        # Aggregate metrics
        total_responses = len(results)
        bleeding_count = sum(1 for r in results if r["has_name_bleeding"])
        consistent_count = total_responses - bleeding_count

        avg_trait_score = np.mean([r["trait_score"] for r in results])

        # Per-character metrics
        per_character_metrics = defaultdict(lambda: {"total": 0, "bleeding": 0, "trait_scores": []})
        for r in results:
            char = r["character"]
            per_character_metrics[char]["total"] += 1
            if r["has_name_bleeding"]:
                per_character_metrics[char]["bleeding"] += 1
            per_character_metrics[char]["trait_scores"].append(r["trait_score"])

        # Compute per-character averages
        per_character_summary = {}
        for char, metrics in per_character_metrics.items():
            per_character_summary[char] = {
                "consistency_rate": 1 - (metrics["bleeding"] / metrics["total"]),
                "avg_trait_score": np.mean(metrics["trait_scores"]) if metrics["trait_scores"] else 0.0,
                "total_responses": metrics["total"]
            }

        return {
            "total_responses": total_responses,
            "consistent_responses": consistent_count,
            "bleeding_responses": bleeding_count,
            "consistency_rate": consistent_count / total_responses if total_responses > 0 else 0.0,
            "avg_trait_score": avg_trait_score,
            "per_character_metrics": per_character_summary,
            "detailed_results": results
        }

    def print_summary(self, batch_results: Dict[str, any]):
        """
        Pretty print evaluation summary.

        Args:
            batch_results: Results from evaluate_batch()
        """
        print("=" * 80)
        print("CHARACTER CONSISTENCY EVALUATION SUMMARY")
        print("=" * 80)

        print(f"\nOverall Metrics:")
        print(f"  Total responses: {batch_results['total_responses']}")
        print(f"  Consistent: {batch_results['consistent_responses']} "
              f"({batch_results['consistency_rate']*100:.1f}%)")
        print(f"  Character bleeding: {batch_results['bleeding_responses']} "
              f"({(1-batch_results['consistency_rate'])*100:.1f}%)")
        print(f"  Average trait score: {batch_results['avg_trait_score']:.3f}")

        print(f"\nPer-Character Metrics:")
        for char, metrics in batch_results['per_character_metrics'].items():
            print(f"  {char}:")
            print(f"    Consistency: {metrics['consistency_rate']*100:.1f}%")
            print(f"    Trait score: {metrics['avg_trait_score']:.3f}")
            print(f"    Responses: {metrics['total_responses']}")

        # Show bleeding examples
        bleeding_examples = [r for r in batch_results['detailed_results'] if r['has_name_bleeding']]
        if bleeding_examples:
            print(f"\n⚠️  Character Bleeding Examples:")
            for i, ex in enumerate(bleeding_examples[:3], 1):  # Show first 3
                print(f"\n  Example {i}:")
                print(f"    Character: {ex['character']}")
                print(f"    Wrong mentions: {', '.join(ex['wrong_characters_mentioned'])}")
                print(f"    Response: {ex['response'][:100]}...")


def compare_models(model1_results: Dict, model2_results: Dict,
                  model1_name: str = "Model 1", model2_name: str = "Model 2"):
    """
    Compare character consistency between two models.

    Args:
        model1_results: Results from evaluate_batch() for first model
        model2_results: Results from evaluate_batch() for second model
        model1_name: Display name for first model
        model2_name: Display name for second model
    """
    print("=" * 80)
    print("MODEL COMPARISON: Character Consistency")
    print("=" * 80)

    print(f"\n{'Metric':<30} {model1_name:<20} {model2_name:<20} {'Delta':<15}")
    print("-" * 80)

    # Consistency rate
    m1_consistency = model1_results['consistency_rate'] * 100
    m2_consistency = model2_results['consistency_rate'] * 100
    delta_consistency = m2_consistency - m1_consistency
    print(f"{'Consistency Rate':<30} {m1_consistency:>6.1f}% {m2_consistency:>18.1f}% "
          f"{delta_consistency:>+11.1f}%")

    # Trait score
    m1_trait = model1_results['avg_trait_score']
    m2_trait = model2_results['avg_trait_score']
    delta_trait = m2_trait - m1_trait
    print(f"{'Avg Trait Score':<30} {m1_trait:>8.3f} {m2_trait:>20.3f} "
          f"{delta_trait:>+13.3f}")

    # Bleeding count
    m1_bleeding = model1_results['bleeding_responses']
    m2_bleeding = model2_results['bleeding_responses']
    delta_bleeding = m2_bleeding - m1_bleeding
    print(f"{'Character Bleeding Count':<30} {m1_bleeding:>8d} {m2_bleeding:>20d} "
          f"{delta_bleeding:>+13d}")

    print("\n" + "=" * 80)

    # Determine winner
    if delta_consistency > 0:
        print(f"✅ {model2_name} shows {delta_consistency:.1f}% improvement in consistency")
    elif delta_consistency < 0:
        print(f"⚠️  {model2_name} shows {abs(delta_consistency):.1f}% worse consistency")
    else:
        print(f"➖ No change in consistency between models")


if __name__ == "__main__":
    # Example usage
    evaluator = CharacterConsistencyEvaluator(["Monika", "Sayori", "Natsuki", "Yuri"])

    # Test cases
    test_characters = ["Monika", "Yuri", "Monika"]
    test_responses = [
        "I love organizing the Literature Club! It's so rewarding.",
        "I-I would like to recommend a book... if that's okay.",
        "Sayori always brightens my day with her smile."  # Name bleeding!
    ]

    # Evaluate
    results = evaluator.evaluate_batch(test_characters, test_responses)
    evaluator.print_summary(results)
