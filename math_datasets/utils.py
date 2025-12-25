"""
Shared utility functions for math datasets
"""

import re
from typing import List


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text (including decimals and negatives).

    Args:
        text: Input text

    Returns:
        List of extracted numbers as floats
    """
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return [float(n) for n in numbers if n]


def accuracy(predictions, answers):
    """Calculate accuracy of predictions.

    Args:
        predictions: List of predicted answers
        answers: List of ground truth answers

    Returns:
        Accuracy ratio (0.0 to 1.0)
    """
    correct = 0
    total = len(predictions)

    for prediction, answer in zip(predictions, answers):
        if prediction == answer:
            correct += 1

    return correct / total if total > 0 else 0.0
