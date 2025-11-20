"""
Emotion Appropriateness Evaluation

Metrics to evaluate if generated responses match the user's emotional state.
Validates emotional consistency and appropriateness in dialogue.
"""

import re
from typing import List, Dict, Tuple
from collections import defaultdict
import numpy as np


class EmotionAppropriatenessEvaluator:
    """
    Evaluates if responses appropriately match user's emotional state.

    Checks:
    - Emotion alignment (does response tone match user emotion?)
    - Emotional keywords presence
    - Response appropriateness for emotion context
    """

    # Define emotion categories and expected response characteristics
    EMOTION_KEYWORDS = {
        # Positive emotions
        "joy": {
            "expected_in_response": ["happy", "glad", "wonderful", "great", "excited", "amazing", "love", "yay", "!"],
            "avoid_in_response": ["sad", "unfortunately", "sorry", "terrible"],
            "tone": "positive"
        },
        "excitement": {
            "expected_in_response": ["exciting", "amazing", "wow", "incredible", "can't wait", "!"],
            "avoid_in_response": ["boring", "dull", "meh"],
            "tone": "positive"
        },
        "gratitude": {
            "expected_in_response": ["thank", "appreciate", "grateful", "kind", "sweet"],
            "avoid_in_response": ["ungrateful", "demanding"],
            "tone": "positive"
        },
        "love": {
            "expected_in_response": ["care", "love", "special", "dear", "sweet", "warm"],
            "avoid_in_response": ["hate", "dislike"],
            "tone": "positive"
        },

        # Negative emotions
        "sadness": {
            "expected_in_response": ["understand", "here for you", "support", "help", "okay", "comfort"],
            "avoid_in_response": ["happy", "celebrate", "exciting", "great"],
            "tone": "supportive"
        },
        "anger": {
            "expected_in_response": ["understand", "calm", "okay", "help", "sorry", "frustrating"],
            "avoid_in_response": ["silly", "overreacting", "just"],
            "tone": "calm"
        },
        "fear": {
            "expected_in_response": ["safe", "okay", "here", "protect", "don't worry", "reassure"],
            "avoid_in_response": ["scary", "afraid", "worried"],
            "tone": "reassuring"
        },
        "disappointment": {
            "expected_in_response": ["sorry", "understand", "unfortunate", "help", "better"],
            "avoid_in_response": ["great", "wonderful", "perfect"],
            "tone": "supportive"
        },

        # Neutral emotions
        "neutral": {
            "expected_in_response": [],  # No specific expectations
            "avoid_in_response": [],
            "tone": "neutral"
        },
        "curiosity": {
            "expected_in_response": ["interesting", "let me", "explain", "sure", "tell you"],
            "avoid_in_response": ["boring", "not interested"],
            "tone": "engaging"
        },
    }

    def __init__(self, emotion_classifier=None):
        """
        Args:
            emotion_classifier: Optional emotion classifier model to analyze response emotion
        """
        self.emotion_classifier = emotion_classifier

    def check_keyword_presence(self, user_emotion: str, response: str) -> Dict[str, any]:
        """
        Check if response contains appropriate keywords for the emotion.

        Args:
            user_emotion: User's emotional state
            response: Generated response text

        Returns:
            Dict with keyword analysis
        """
        if user_emotion not in self.EMOTION_KEYWORDS:
            return {
                "has_expected_keywords": False,
                "has_inappropriate_keywords": False,
                "expected_keywords_found": [],
                "inappropriate_keywords_found": [],
                "keyword_score": 0.5  # Neutral for unknown emotions
            }

        emotion_data = self.EMOTION_KEYWORDS[user_emotion]
        response_lower = response.lower()

        # Check for expected keywords
        expected_found = []
        for keyword in emotion_data["expected_in_response"]:
            if keyword.lower() in response_lower:
                expected_found.append(keyword)

        # Check for inappropriate keywords
        inappropriate_found = []
        for keyword in emotion_data["avoid_in_response"]:
            if keyword.lower() in response_lower:
                inappropriate_found.append(keyword)

        # Calculate keyword score
        # Score = (expected keywords found / total expected) - (inappropriate found / total avoid)
        expected_score = len(expected_found) / len(emotion_data["expected_in_response"]) \
            if emotion_data["expected_in_response"] else 0.5

        inappropriate_penalty = len(inappropriate_found) / len(emotion_data["avoid_in_response"]) \
            if emotion_data["avoid_in_response"] else 0

        keyword_score = max(0, expected_score - inappropriate_penalty)

        return {
            "has_expected_keywords": len(expected_found) > 0,
            "has_inappropriate_keywords": len(inappropriate_found) > 0,
            "expected_keywords_found": expected_found,
            "inappropriate_keywords_found": inappropriate_found,
            "keyword_score": keyword_score,
            "expected_tone": emotion_data["tone"]
        }

    def check_response_length_appropriateness(self, user_emotion: str, response: str) -> Dict[str, any]:
        """
        Check if response length is appropriate for the emotion.

        Some emotions (like sadness, fear) benefit from longer, more supportive responses.
        Others (like joy) can be shorter and more energetic.

        Args:
            user_emotion: User's emotional state
            response: Generated response text

        Returns:
            Dict with length analysis
        """
        word_count = len(response.split())

        # Expected length ranges by emotion type
        length_expectations = {
            "sadness": (15, 40),  # Longer, supportive
            "anger": (15, 35),    # Calm, measured
            "fear": (15, 40),     # Reassuring, detailed
            "joy": (5, 25),       # Can be short and energetic
            "excitement": (5, 25),
            "neutral": (5, 30),
            "gratitude": (5, 20),
        }

        min_length, max_length = length_expectations.get(user_emotion, (5, 30))

        is_appropriate_length = min_length <= word_count <= max_length
        length_score = 1.0 if is_appropriate_length else \
            max(0, 1 - abs(word_count - ((min_length + max_length) / 2)) / max_length)

        return {
            "word_count": word_count,
            "expected_range": (min_length, max_length),
            "is_appropriate_length": is_appropriate_length,
            "length_score": length_score
        }

    def evaluate_response(self,
                         user_emotion: str,
                         response: str,
                         user_input: str = None) -> Dict[str, any]:
        """
        Complete evaluation of response appropriateness for user emotion.

        Args:
            user_emotion: User's emotional state
            response: Generated response text
            user_input: Optional user input for context

        Returns:
            Dict with all emotion appropriateness metrics
        """
        keyword_analysis = self.check_keyword_presence(user_emotion, response)
        length_analysis = self.check_response_length_appropriateness(user_emotion, response)

        # Compute overall appropriateness score
        overall_score = (keyword_analysis["keyword_score"] * 0.7 +
                        length_analysis["length_score"] * 0.3)

        is_appropriate = (
            keyword_analysis["has_expected_keywords"] or user_emotion == "neutral"
        ) and not keyword_analysis["has_inappropriate_keywords"]

        return {
            "user_emotion": user_emotion,
            "response": response,
            "user_input": user_input,
            **keyword_analysis,
            **length_analysis,
            "overall_appropriateness_score": overall_score,
            "is_appropriate": is_appropriate
        }

    def evaluate_batch(self,
                      user_emotions: List[str],
                      responses: List[str],
                      user_inputs: List[str] = None) -> Dict[str, any]:
        """
        Evaluate multiple responses and compute aggregate metrics.

        Args:
            user_emotions: List of user emotional states
            responses: List of generated responses
            user_inputs: Optional list of user inputs

        Returns:
            Dict with aggregate emotion appropriateness metrics
        """
        if user_inputs is None:
            user_inputs = [None] * len(responses)

        results = []
        for emotion, response, user_input in zip(user_emotions, responses, user_inputs):
            results.append(self.evaluate_response(emotion, response, user_input))

        # Aggregate metrics
        total_responses = len(results)
        appropriate_count = sum(1 for r in results if r["is_appropriate"])

        avg_keyword_score = np.mean([r["keyword_score"] for r in results])
        avg_length_score = np.mean([r["length_score"] for r in results])
        avg_overall_score = np.mean([r["overall_appropriateness_score"] for r in results])

        # Per-emotion metrics
        per_emotion_metrics = defaultdict(lambda: {
            "total": 0,
            "appropriate": 0,
            "keyword_scores": [],
            "overall_scores": []
        })

        for r in results:
            emotion = r["user_emotion"]
            per_emotion_metrics[emotion]["total"] += 1
            if r["is_appropriate"]:
                per_emotion_metrics[emotion]["appropriate"] += 1
            per_emotion_metrics[emotion]["keyword_scores"].append(r["keyword_score"])
            per_emotion_metrics[emotion]["overall_scores"].append(r["overall_appropriateness_score"])

        # Compute per-emotion averages
        per_emotion_summary = {}
        for emotion, metrics in per_emotion_metrics.items():
            per_emotion_summary[emotion] = {
                "appropriateness_rate": metrics["appropriate"] / metrics["total"],
                "avg_keyword_score": np.mean(metrics["keyword_scores"]),
                "avg_overall_score": np.mean(metrics["overall_scores"]),
                "total_responses": metrics["total"]
            }

        return {
            "total_responses": total_responses,
            "appropriate_responses": appropriate_count,
            "inappropriate_responses": total_responses - appropriate_count,
            "appropriateness_rate": appropriate_count / total_responses if total_responses > 0 else 0.0,
            "avg_keyword_score": avg_keyword_score,
            "avg_length_score": avg_length_score,
            "avg_overall_score": avg_overall_score,
            "per_emotion_metrics": per_emotion_summary,
            "detailed_results": results
        }

    def print_summary(self, batch_results: Dict[str, any]):
        """
        Pretty print evaluation summary.

        Args:
            batch_results: Results from evaluate_batch()
        """
        print("=" * 80)
        print("EMOTION APPROPRIATENESS EVALUATION SUMMARY")
        print("=" * 80)

        print(f"\nOverall Metrics:")
        print(f"  Total responses: {batch_results['total_responses']}")
        print(f"  Appropriate: {batch_results['appropriate_responses']} "
              f"({batch_results['appropriateness_rate']*100:.1f}%)")
        print(f"  Inappropriate: {batch_results['inappropriate_responses']} "
              f"({(1-batch_results['appropriateness_rate'])*100:.1f}%)")
        print(f"  Average keyword score: {batch_results['avg_keyword_score']:.3f}")
        print(f"  Average length score: {batch_results['avg_length_score']:.3f}")
        print(f"  Average overall score: {batch_results['avg_overall_score']:.3f}")

        print(f"\nPer-Emotion Metrics:")
        for emotion, metrics in sorted(batch_results['per_emotion_metrics'].items()):
            print(f"  {emotion}:")
            print(f"    Appropriateness: {metrics['appropriateness_rate']*100:.1f}%")
            print(f"    Keyword score: {metrics['avg_keyword_score']:.3f}")
            print(f"    Overall score: {metrics['avg_overall_score']:.3f}")
            print(f"    Responses: {metrics['total_responses']}")

        # Show inappropriate examples
        inappropriate_examples = [r for r in batch_results['detailed_results']
                                 if not r['is_appropriate']]
        if inappropriate_examples:
            print(f"\n⚠️  Inappropriate Response Examples:")
            for i, ex in enumerate(inappropriate_examples[:3], 1):  # Show first 3
                print(f"\n  Example {i}:")
                print(f"    User emotion: {ex['user_emotion']}")
                if ex.get('user_input'):
                    print(f"    User input: {ex['user_input']}")
                print(f"    Response: {ex['response'][:100]}...")
                if ex['inappropriate_keywords_found']:
                    print(f"    ⚠️  Inappropriate keywords: {', '.join(ex['inappropriate_keywords_found'])}")


def compare_models(model1_results: Dict, model2_results: Dict,
                  model1_name: str = "Model 1", model2_name: str = "Model 2"):
    """
    Compare emotion appropriateness between two models.

    Args:
        model1_results: Results from evaluate_batch() for first model
        model2_results: Results from evaluate_batch() for second model
        model1_name: Display name for first model
        model2_name: Display name for second model
    """
    print("=" * 80)
    print("MODEL COMPARISON: Emotion Appropriateness")
    print("=" * 80)

    print(f"\n{'Metric':<30} {model1_name:<20} {model2_name:<20} {'Delta':<15}")
    print("-" * 80)

    # Appropriateness rate
    m1_appropriate = model1_results['appropriateness_rate'] * 100
    m2_appropriate = model2_results['appropriateness_rate'] * 100
    delta_appropriate = m2_appropriate - m1_appropriate
    print(f"{'Appropriateness Rate':<30} {m1_appropriate:>6.1f}% {m2_appropriate:>18.1f}% "
          f"{delta_appropriate:>+11.1f}%")

    # Keyword score
    m1_keyword = model1_results['avg_keyword_score']
    m2_keyword = model2_results['avg_keyword_score']
    delta_keyword = m2_keyword - m1_keyword
    print(f"{'Avg Keyword Score':<30} {m1_keyword:>8.3f} {m2_keyword:>20.3f} "
          f"{delta_keyword:>+13.3f}")

    # Overall score
    m1_overall = model1_results['avg_overall_score']
    m2_overall = model2_results['avg_overall_score']
    delta_overall = m2_overall - m1_overall
    print(f"{'Avg Overall Score':<30} {m1_overall:>8.3f} {m2_overall:>20.3f} "
          f"{delta_overall:>+13.3f}")

    print("\n" + "=" * 80)

    # Determine winner
    if delta_appropriate > 0:
        print(f"✅ {model2_name} shows {delta_appropriate:.1f}% improvement in emotion matching")
    elif delta_appropriate < 0:
        print(f"⚠️  {model2_name} shows {abs(delta_appropriate):.1f}% worse emotion matching")
    else:
        print(f"➖ No change in emotion appropriateness between models")


if __name__ == "__main__":
    # Example usage
    evaluator = EmotionAppropriatenessEvaluator()

    # Test cases
    test_emotions = ["joy", "sadness", "anger"]
    test_inputs = [
        "I got accepted to my dream school!",
        "I'm feeling really down today...",
        "This is so frustrating!"
    ]
    test_responses = [
        "That's wonderful! I'm so happy for you!",  # Good match for joy
        "Cheer up! Everything is great!",  # Bad match for sadness
        "I understand this must be frustrating. Is there anything I can help with?",  # Good match for anger
    ]

    # Evaluate
    results = evaluator.evaluate_batch(test_emotions, test_responses, test_inputs)
    evaluator.print_summary(results)
