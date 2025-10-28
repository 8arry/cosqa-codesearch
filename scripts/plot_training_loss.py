"""
Visualize Training Loss Curve

This script reads loss_history.json and generates visualizations of training loss
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load loss history
loss_file = Path('models/finetuned/loss_history.json')

if not loss_file.exists():
    print(f"❌ File not found: {loss_file}")
    print("Please run the training script first to generate loss_history.json")
    exit(1)

with open(loss_file, 'r') as f:
    loss_history = json.load(f)

print(f"✅ Loaded loss data for {len(loss_history)} training steps")

# Extract data
steps = [entry['step'] for entry in loss_history]
losses = [entry['loss'] for entry in loss_history]
epochs = [entry['epoch'] for entry in loss_history]

# Calculate epoch boundaries
epoch_boundaries = []
for i in range(1, len(epochs)):
    if epochs[i] != epochs[i-1]:
        epoch_boundaries.append(steps[i])

# ========================================
# Chart 1: Full Training Loss Curve
# ========================================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, losses, linewidth=1, alpha=0.7, color='#3498db')
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Curve (Linear Scale)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Mark epoch boundaries
for epoch_step in epoch_boundaries:
    plt.axvline(x=epoch_step, color='red', linestyle='--', alpha=0.5, linewidth=1)

# Add initial and final loss annotations
plt.text(steps[0], losses[0], f'Initial: {losses[0]:.4f}', 
         fontsize=10, ha='left', va='bottom', color='red', fontweight='bold')
plt.text(steps[-1], losses[-1], f'Final: {losses[-1]:.4f}', 
         fontsize=10, ha='right', va='top', color='red', fontweight='bold')

# ========================================
# Chart 2: Log Scale (better view of decline trend)
# ========================================
plt.subplot(1, 2, 2)
plt.plot(steps, losses, linewidth=1, alpha=0.7, color='#e74c3c')
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss (log scale)', fontsize=12)
plt.title('Training Loss Curve (Log Scale)', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# Mark epoch boundaries
for epoch_step in epoch_boundaries:
    plt.axvline(x=epoch_step, color='red', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('results/training_loss_curve.png', dpi=300, bbox_inches='tight')
print(f"✅ Chart saved to: results/training_loss_curve.png")

# ========================================
# Chart 3: Loss Distribution by Epoch
# ========================================
plt.figure(figsize=(12, 6))

# Group by epoch
epoch_losses = {0: [], 1: [], 2: []}
for i, epoch in enumerate(epochs):
    epoch_losses[epoch].append(losses[i])

# Plot each epoch
colors = ['#3498db', '#2ecc71', '#e74c3c']
for epoch in [0, 1, 2]:
    epoch_steps = [i for i, e in enumerate(epochs) if e == epoch]
    epoch_loss_values = epoch_losses[epoch]
    plt.plot(epoch_steps, epoch_loss_values, 
             label=f'Epoch {epoch+1}', color=colors[epoch], linewidth=2, alpha=0.8)

plt.xlabel('Steps within Training', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss by Epoch', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_loss_by_epoch.png', dpi=300, bbox_inches='tight')
print(f"✅ Chart saved to: results/training_loss_by_epoch.png")

# ========================================
# Chart 4: Smoothed Loss Curve (Moving Average)
# ========================================
def moving_average(data, window_size=50):
    """Calculate moving average"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(12, 6))

# Original loss (semi-transparent)
plt.plot(steps, losses, linewidth=0.5, alpha=0.3, color='gray', label='Original')

# Smoothed loss
window = 50
smoothed_losses = moving_average(losses, window)
smoothed_steps = steps[window-1:]
plt.plot(smoothed_steps, smoothed_losses, linewidth=2, color='#9b59b6', label=f'Moving Avg (window={window})')

plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Smoothed Training Loss Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Mark epoch boundaries
for epoch_step in epoch_boundaries:
    plt.axvline(x=epoch_step, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
plt.tight_layout()
plt.savefig('results/training_loss_smoothed.png', dpi=300, bbox_inches='tight')
print(f"✅ Chart saved to: results/training_loss_smoothed.png")

# ========================================
# Statistics
# ========================================
print("\n" + "="*60)
print("Training Loss Statistics")
print("="*60)

print(f"\nInitial Loss: {losses[0]:.4f}")
print(f"Final Loss: {losses[-1]:.4f}")
print(f"Loss Reduction: {losses[0] - losses[-1]:.4f} ({(losses[0] - losses[-1])/losses[0]*100:.1f}%)")

print(f"\nLoss per Epoch:")
for epoch in [0, 1, 2]:
    epoch_loss = epoch_losses[epoch]
    print(f"  Epoch {epoch+1}: Start={epoch_loss[0]:.4f}, End={epoch_loss[-1]:.4f}, "
          f"Mean={np.mean(epoch_loss):.4f}")

print(f"\nTotal Training Steps: {len(steps)}")
print(f"Number of Epochs: 3")

plt.show()
print("\n✅ All charts generated and saved!")
