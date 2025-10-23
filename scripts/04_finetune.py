"""
Script 4: Fine-tune Model on CoSQA

This script fine-tunes an embedding model on CoSQA training data
using Multiple Negatives Ranking Loss.

Run: python scripts/04_finetune.py --model intfloat/e5-base-v2 --epochs 3
"""

import sys
import argparse
from pathlib import Path
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.load_cosqa import CoSQADataLoader
from src.training.trainer import CoSQATrainer
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune embedding model on CoSQA")
    parser.add_argument(
        "--model",
        type=str,
        default="intfloat/e5-base-v2",
        help="Base model to fine-tune (default: intfloat/e5-base-v2)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps (default: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/finetuned",
        help="Output directory for fine-tuned model (default: models/finetuned)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/cache",
        help="Cache directory for data (default: data/cache)"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps (default: 500)"
    )
    parser.add_argument(
        "--use-validation",
        action="store_true",
        help="Use validation set for evaluation during training"
    )
    return parser.parse_args()


def prepare_validation_data(loader):
    """Prepare validation data for evaluation during training."""
    valid_df = loader.load_valid()
    
    # Create queries dict
    queries = {}
    corpus = {}
    relevant_docs = {}
    
    for _, row in valid_df.iterrows():
        query_id = row['query_id']
        corpus_id = row['corpus_id']
        
        queries[query_id] = row['query_text']
        corpus[corpus_id] = row['code_text']
        
        if query_id not in relevant_docs:
            relevant_docs[query_id] = set()
        
        if row['score'] == 1:
            relevant_docs[query_id].add(corpus_id)
    
    return queries, corpus, relevant_docs


def main():
    args = parse_args()
    
    print("="*80)
    print("STEP 4: Fine-tune Model on CoSQA")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Base model:     {args.model}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Warmup steps:   {args.warmup_steps}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Use validation: {args.use_validation}")
    
    # Load training data
    print("\n" + "-"*80)
    print("Loading training data...")
    print("-"*80)
    
    loader = CoSQADataLoader(cache_dir=args.cache_dir)
    
    # Get positive training pairs
    positive_pairs = loader.get_positive_pairs('train')
    
    print(f"\n✓ Loaded {len(positive_pairs):,} positive (query, code) pairs")
    print(f"  These pairs will be used for contrastive learning")
    print(f"  In-batch negatives: {args.batch_size - 1} per sample")
    
    # Initialize trainer
    print("\n" + "-"*80)
    print("Initializing trainer...")
    print("-"*80)
    
    trainer = CoSQATrainer(
        model_name=args.model,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    print(f"\n✓ Trainer initialized")
    
    # Prepare training data
    print("\n" + "-"*80)
    print("Preparing training data...")
    print("-"*80)
    
    train_dataloader = trainer.prepare_training_data(positive_pairs)
    
    total_steps = len(train_dataloader) * args.epochs
    print(f"\n✓ Training data prepared")
    print(f"  Batches per epoch: {len(train_dataloader)}")
    print(f"  Total steps:       {total_steps}")
    print(f"  Estimated time:    ~{total_steps * 0.5 / 60:.1f}-{total_steps * 1.0 / 60:.1f} minutes")
    
    # Prepare validation data if requested
    evaluator = None
    if args.use_validation:
        print("\n" + "-"*80)
        print("Preparing validation data...")
        print("-"*80)
        
        queries, corpus, relevant_docs = prepare_validation_data(loader)
        evaluator = trainer.prepare_evaluation_data(queries, corpus, relevant_docs)
        
        print(f"\n✓ Validation data prepared")
        print(f"  Queries:  {len(queries)}")
        print(f"  Corpus:   {len(corpus)}")
        print(f"  Will evaluate every {args.eval_steps} steps")
    
    # Train the model
    print("\n" + "-"*80)
    print("Starting fine-tuning...")
    print("-"*80)
    
    start_time = time.time()
    
    model, loss_history = trainer.train(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        evaluation_steps=args.eval_steps,
        save_best_model=True
    )
    
    train_time = time.time() - start_time
    
    print(f"\n✓ Training completed in {train_time:.2f}s ({train_time/60:.2f} minutes)")
    print(f"  Steps/sec: {total_steps / train_time:.2f}")
    print(f"  Recorded {len(loss_history)} loss values")
    if loss_history:
        print(f"  Initial loss: {loss_history[0]['loss']:.4f}")
        print(f"  Final loss: {loss_history[-1]['loss']:.4f}")
    
    # Save training info
    print("\n" + "-"*80)
    print("Saving training info...")
    print("-"*80)
    
    output_dir = Path(args.output_dir)
    
    training_info = {
        'base_model': args.model,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'warmup_steps': args.warmup_steps,
        'training_pairs': len(positive_pairs),
        'total_steps': total_steps,
        'training_time_sec': train_time,
        'training_time_min': train_time / 60,
        'used_validation': args.use_validation
    }
    
    info_file = output_dir / 'training_info.json'
    with open(info_file, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"✓ Training info saved to: {info_file}")
    
    # Summary
    print("\n" + "="*80)
    print("✓ Fine-tuning complete!")
    print("="*80)
    
    print(f"\nModel statistics:")
    print(f"  Base model:       {args.model}")
    print(f"  Training pairs:   {len(positive_pairs):,}")
    print(f"  Training epochs:  {args.epochs}")
    print(f"  Total steps:      {total_steps}")
    print(f"  Training time:    {train_time/60:.2f} minutes")
    
    print(f"\nModel saved to: {output_dir}")
    
    print(f"\nNext step: Run scripts/05_evaluate_finetuned.py to evaluate the fine-tuned model")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
