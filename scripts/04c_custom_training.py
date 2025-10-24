"""
Script 4c: Custom Training Loop with Loss Tracking

This script implements a custom training loop to properly track
and save loss values at each step, avoiding OOM issues.

Run: python scripts/04c_custom_training.py
"""

import sys
import json
import time
from pathlib import Path
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.load_cosqa import CoSQADataLoader


def train_with_loss_tracking(
    model,
    train_dataloader,
    loss_fn,
    epochs=3,
    warmup_steps=100,
    lr=2e-5,
    output_dir="models/finetuned",
    batch_size=16
):
    """Custom training loop with loss tracking."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = model.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    total_steps = len(train_dataloader) * epochs
    
    # Learning rate scheduler with warmup
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Track loss
    loss_history = []
    global_step = 0
    
    print(f"\nTraining Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Batches per epoch: {len(train_dataloader)}")
    
    model.train()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_start = time.time()
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*80}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # smart_batching_collate returns (features, labels)
            # For MultipleNegativesRankingLoss, we need the features list
            features, labels = batch
            
            # Move features to device
            for feature_dict in features:
                for key in feature_dict:
                    if isinstance(feature_dict[key], torch.Tensor):
                        feature_dict[key] = feature_dict[key].to(device)
            
            # Forward pass through loss function
            loss_value = loss_fn(features, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss_value.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Track loss
            current_loss = loss_value.item()
            loss_history.append({
                'step': global_step,
                'epoch': epoch,
                'loss': current_loss
            })
            
            epoch_loss += current_loss
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{epoch_loss / (batch_idx + 1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Periodically save loss history
            if global_step % 100 == 0:
                loss_file = output_path / "loss_history.json"
                with open(loss_file, 'w') as f:
                    json.dump(loss_history, f, indent=2)
            
            # Clear CUDA cache periodically
            if global_step % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        
        print(f"\nEpoch {epoch + 1} completed in {epoch_time/60:.2f} minutes")
        print(f"  Average loss: {avg_epoch_loss:.4f}")
        print(f"  Learning rate: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint after each epoch
        checkpoint_path = output_path / f"checkpoint-epoch{epoch+1}"
        model.save(str(checkpoint_path))
        print(f"  Checkpoint saved: {checkpoint_path}")
    
    train_time = time.time() - start_time
    
    # Save final model
    model.save(str(output_path))
    print(f"\n✓ Final model saved to: {output_path}")
    
    # Save final loss history
    loss_file = output_path / "loss_history.json"
    with open(loss_file, 'w') as f:
        json.dump(loss_history, f, indent=2)
    print(f"✓ Loss history saved to: {loss_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}")
    print(f"Total training time: {train_time/60:.2f} minutes ({train_time/3600:.2f} hours)")
    print(f"Total steps: {global_step}")
    print(f"Initial loss: {loss_history[0]['loss']:.4f}")
    print(f"Final loss: {loss_history[-1]['loss']:.4f}")
    print(f"Loss reduction: {loss_history[0]['loss'] - loss_history[-1]['loss']:.4f}")
    
    return model, loss_history, train_time


def main():
    print("="*80)
    print("Custom Training with Loss Tracking")
    print("="*80)
    
    # Configuration
    model_name = "intfloat/e5-base-v2"
    batch_size = 16  # Reduced to avoid OOM
    epochs = 3
    lr = 2e-5
    warmup_steps = 100
    output_dir = "models/finetuned"
    cache_dir = "data/cache"
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.empty_cache()
    else:
        print("\n⚠ No GPU available, using CPU (training will be very slow)")
    
    # Load data
    print("\n" + "-"*80)
    print("Loading training data...")
    print("-"*80)
    
    loader = CoSQADataLoader(cache_dir=cache_dir)
    positive_pairs = loader.get_positive_pairs('train')
    
    print(f"✓ Loaded {len(positive_pairs):,} training pairs")
    
    # Load model FIRST (before creating dataloader)
    print("\n" + "-"*80)
    print("Loading model...")
    print("-"*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    
    print(f"✓ Model loaded: {model_name}")
    print(f"  Device: {device}")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Prepare dataloader (now model is available)
    print("\n" + "-"*80)
    print("Preparing dataloader...")
    print("-"*80)
    
    train_examples = [
        InputExample(texts=[query, code])
        for query, code in positive_pairs
    ]
    
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=model.smart_batching_collate
    )
    
    print(f"✓ Created dataloader with {len(train_dataloader)} batches")
    
    # Create loss function
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    
    # Train
    print("\n" + "-"*80)
    print("Starting training...")
    print("-"*80)
    
    try:
        model, loss_history, train_time = train_with_loss_tracking(
            model=model,
            train_dataloader=train_dataloader,
            loss_fn=loss_fn,
            epochs=epochs,
            warmup_steps=warmup_steps,
            lr=lr,
            output_dir=output_dir,
            batch_size=batch_size
        )
        
        # Save training info
        training_info = {
            'base_model': model_name,
            'batch_size': batch_size,
            'num_epochs': epochs,
            'learning_rate': lr,
            'warmup_steps': warmup_steps,
            'training_pairs': len(positive_pairs),
            'total_steps': len(loss_history),
            'training_time_sec': train_time,
            'training_time_min': train_time / 60,
            'training_time_hours': train_time / 3600,
            'initial_loss': loss_history[0]['loss'],
            'final_loss': loss_history[-1]['loss'],
            'loss_reduction': loss_history[0]['loss'] - loss_history[-1]['loss']
        }
        
        info_file = Path(output_dir) / 'training_info.json'
        with open(info_file, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        print(f"\n✓ Training info saved to: {info_file}")
        
        print("\n" + "="*80)
        print("✓ Training complete!")
        print("="*80)
        print(f"\nNext step: python scripts/05_evaluate_finetuned.py")
        
        return 0
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ OOM Error even with batch_size={batch_size}")
            print("\nSuggestions:")
            print("  1. Try batch_size=8: python scripts/04c_custom_training.py")
            print("     (modify batch_size in the script)")
            print("  2. Close other GPU applications")
            print("  3. Restart Python kernel to clear GPU memory")
            return 1
        else:
            raise


if __name__ == "__main__":
    sys.exit(main())
