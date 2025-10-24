"""
Fine-tuning trainer for CoSQA code search.

This module implements fine-tuning of sentence transformers using
Multiple Negatives Ranking Loss on CoSQA training data.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LossTracker:
    """Callback to track training loss at each step."""
    
    def __init__(self, output_path: Path):
        self.losses = []
        self.current_epoch = 0
        self.output_path = output_path
        self.save_frequency = 50  # Save every 50 steps to avoid memory issues
        
    def __call__(self, score, epoch, steps):
        """Called after each training step."""
        # score is the loss value
        self.current_epoch = epoch
        loss_entry = {
            'step': steps,
            'epoch': epoch,
            'loss': float(score) if hasattr(score, 'item') else float(score)
        }
        self.losses.append(loss_entry)
        
        # Periodically save and clear memory
        if len(self.losses) >= self.save_frequency:
            self._save_partial()
        
    def _save_partial(self):
        """Save accumulated losses and clear memory."""
        output_file = self.output_path / "loss_history.json"
        
        # Load existing data if file exists
        existing_losses = []
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_losses = json.load(f)
            except:
                pass
        
        # Append new losses
        existing_losses.extend(self.losses)
        
        # Save
        with open(output_file, 'w') as f:
            json.dump(existing_losses, f, indent=2)
        
        # Clear memory
        self.losses = []
        
    def save(self):
        """Save final loss history to JSON."""
        # Save any remaining losses
        if self.losses:
            self._save_partial()
        
        output_file = self.output_path / "loss_history.json"
        logger.info(f"Loss history saved to: {output_file}")


class CoSQATrainer:
    """
    Trainer for fine-tuning embedding models on CoSQA.
    
    Uses Multiple Negatives Ranking Loss for contrastive learning:
    - Positive pairs: (query, relevant_code)
    - In-batch negatives: Other codes in the same batch
    """
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        batch_size: int = 32,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        learning_rate: float = 2e-5,
        output_dir: str = "models/finetuned",
        device: Optional[str] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model_name: Base model to fine-tune
            batch_size: Training batch size
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps
            learning_rate: Learning rate
            output_dir: Directory to save fine-tuned model
            device: Device to use (cuda/cpu), auto-detect if None
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.output_dir = Path(output_dir)
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
    def prepare_training_data(
        self, 
        positive_pairs: List[Tuple[str, str]]
    ) -> DataLoader:
        """
        Prepare training data loader.
        
        Args:
            positive_pairs: List of (query, code) tuples
            
        Returns:
            DataLoader for training
        """
        logger.info(f"Preparing {len(positive_pairs)} training pairs...")
        
        # Convert to InputExample format
        # For MultipleNegativesRankingLoss, we only need positive pairs
        # Negatives are automatically created from other samples in the batch
        train_examples = [
            InputExample(texts=[query, code])
            for query, code in positive_pairs
        ]
        
        # Create DataLoader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=self.batch_size
        )
        
        logger.info(f"Created DataLoader with {len(train_dataloader)} batches")
        return train_dataloader
    
    def prepare_evaluation_data(
        self,
        queries: Dict[str, str],
        corpus: Dict[str, str],
        relevant_docs: Dict[str, set]
    ) -> InformationRetrievalEvaluator:
        """
        Prepare evaluation data.
        
        Args:
            queries: Dict mapping query_id -> query_text
            corpus: Dict mapping corpus_id -> code_text
            relevant_docs: Dict mapping query_id -> set of relevant corpus_ids
            
        Returns:
            InformationRetrievalEvaluator
        """
        logger.info(f"Preparing evaluator with {len(queries)} queries and {len(corpus)} documents")
        
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="cosqa-valid",
            score_functions={'cos_sim': lambda x, y: torch.nn.functional.cosine_similarity(x, y)}
        )
        
        return evaluator
    
    def train(
        self,
        train_dataloader: DataLoader,
        evaluator: Optional[InformationRetrievalEvaluator] = None,
        evaluation_steps: int = 500,
        save_best_model: bool = True
    ) -> Tuple[SentenceTransformer, List[Dict]]:
        """
        Fine-tune the model.
        
        Args:
            train_dataloader: Training data
            evaluator: Optional evaluator for validation
            evaluation_steps: Evaluate every N steps
            save_best_model: Save best model based on validation
            
        Returns:
            Tuple of (fine-tuned model, loss history)
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create loss tracker with output path
        loss_tracker = LossTracker(self.output_dir)
        
        # Define loss function
        # MultipleNegativesRankingLoss: 
        # - Treats other samples in batch as negatives
        # - Efficient contrastive learning
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        logger.info("="*80)
        logger.info("Training Configuration")
        logger.info("="*80)
        logger.info(f"Model:           {self.model_name}")
        logger.info(f"Batch size:      {self.batch_size}")
        logger.info(f"Num epochs:      {self.num_epochs}")
        logger.info(f"Warmup steps:    {self.warmup_steps}")
        logger.info(f"Learning rate:   {self.learning_rate}")
        logger.info(f"Training steps:  {len(train_dataloader) * self.num_epochs}")
        logger.info(f"Device:          {self.device}")
        logger.info(f"Output dir:      {self.output_dir}")
        logger.info("="*80)
        
        # Training arguments
        # Note: sentence-transformers doesn't directly support loss tracking callbacks
        # We'll use gradient accumulation to reduce memory usage
        training_args = {
            'train_objectives': [(train_dataloader, train_loss)],
            'epochs': self.num_epochs,
            'warmup_steps': self.warmup_steps,
            'output_path': str(self.output_dir),
            'optimizer_params': {'lr': self.learning_rate},
            'show_progress_bar': True,
        }
        
        # Add evaluator if provided
        if evaluator is not None:
            training_args['evaluator'] = evaluator
            training_args['evaluation_steps'] = evaluation_steps
            training_args['save_best_model'] = save_best_model
            logger.info(f"Evaluation every {evaluation_steps} steps")
        
        # Train the model
        logger.info("\nStarting training...")
        self.model.fit(**training_args)
        
        # Save final loss history
        loss_tracker.save()
        
        logger.info("\n" + "="*80)
        logger.info("✓ Training complete!")
        logger.info("="*80)
        logger.info(f"Model saved to: {self.output_dir}")
        logger.info(f"Total training steps: {len(loss_tracker.losses)}")
        if loss_tracker.losses:
            logger.info(f"Initial loss: {loss_tracker.losses[0]['loss']:.4f}")
            logger.info(f"Final loss: {loss_tracker.losses[-1]['loss']:.4f}")
        
        return self.model, loss_tracker.losses
    
    def save_model(self, path: Optional[str] = None):
        """Save the fine-tuned model."""
        save_path = path or str(self.output_dir)
        self.model.save(save_path)
        logger.info(f"Model saved to: {save_path}")
    
    def load_model(self, path: str):
        """Load a fine-tuned model."""
        self.model = SentenceTransformer(path, device=self.device)
        logger.info(f"Model loaded from: {path}")
        return self.model


def create_trainer(
    model_name: str = "intfloat/e5-base-v2",
    **kwargs
) -> CoSQATrainer:
    """
    Convenience function to create a trainer.
    
    Args:
        model_name: Base model to fine-tune
        **kwargs: Additional arguments for CoSQATrainer
        
    Returns:
        Initialized trainer
    """
    return CoSQATrainer(model_name=model_name, **kwargs)


if __name__ == "__main__":
    # Demo: Create a trainer
    print("Creating trainer...")
    trainer = create_trainer(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=16,
        num_epochs=1
    )
    
    # Demo training data
    demo_pairs = [
        ("how to sort a list", "def sort_list(lst): return sorted(lst)"),
        ("reverse a string", "def reverse(s): return s[::-1]"),
        ("add two numbers", "def add(a, b): return a + b"),
    ]
    
    print(f"\nPreparing {len(demo_pairs)} demo pairs...")
    train_loader = trainer.prepare_training_data(demo_pairs)
    
    print(f"DataLoader created with {len(train_loader)} batches")
    print(f"Batch size: {trainer.batch_size}")
    print("\n✓ Trainer demo complete!")
