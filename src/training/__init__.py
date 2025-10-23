"""
Training module for fine-tuning embedding models.

This module provides utilities for fine-tuning sentence transformers
on the CoSQA dataset using contrastive learning.
"""

from .trainer import CoSQATrainer

__all__ = [
    "CoSQATrainer",
]
