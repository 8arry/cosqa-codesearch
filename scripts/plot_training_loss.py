"""
可视化训练损失曲线

这个脚本读取 loss_history.json 并生成训练损失的可视化图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 加载损失历史
loss_file = Path('models/finetuned/loss_history.json')

if not loss_file.exists():
    print(f"❌ 找不到文件: {loss_file}")
    print("请先运行训练脚本生成 loss_history.json")
    exit(1)

with open(loss_file, 'r') as f:
    loss_history = json.load(f)

print(f"✅ 加载了 {len(loss_history)} 个训练步骤的损失数据")

# 提取数据
steps = [entry['step'] for entry in loss_history]
losses = [entry['loss'] for entry in loss_history]
epochs = [entry['epoch'] for entry in loss_history]

# 计算每个 epoch 的边界
epoch_boundaries = []
for i in range(1, len(epochs)):
    if epochs[i] != epochs[i-1]:
        epoch_boundaries.append(steps[i])

# ========================================
# 图1: 完整训练损失曲线
# ========================================
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(steps, losses, linewidth=1, alpha=0.7, color='#3498db')
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Curve (Linear Scale)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# 标记 epoch 边界
for epoch_step in epoch_boundaries:
    plt.axvline(x=epoch_step, color='red', linestyle='--', alpha=0.5, linewidth=1)

# 添加初始和最终损失标注
plt.text(steps[0], losses[0], f'Initial: {losses[0]:.4f}', 
         fontsize=10, ha='left', va='bottom', color='red', fontweight='bold')
plt.text(steps[-1], losses[-1], f'Final: {losses[-1]:.4f}', 
         fontsize=10, ha='right', va='top', color='red', fontweight='bold')

# ========================================
# 图2: 对数刻度（更清楚地看下降趋势）
# ========================================
plt.subplot(1, 2, 2)
plt.plot(steps, losses, linewidth=1, alpha=0.7, color='#e74c3c')
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss (log scale)', fontsize=12)
plt.title('Training Loss Curve (Log Scale)', fontsize=14, fontweight='bold')
plt.yscale('log')
plt.grid(True, alpha=0.3)

# 标记 epoch 边界
for epoch_step in epoch_boundaries:
    plt.axvline(x=epoch_step, color='red', linestyle='--', alpha=0.5, linewidth=1)

plt.tight_layout()
plt.savefig('results/training_loss_curve.png', dpi=300, bbox_inches='tight')
print(f"✅ 保存图表到: results/training_loss_curve.png")

# ========================================
# 图3: 每个 Epoch 的损失分布
# ========================================
plt.figure(figsize=(12, 6))

# 按 epoch 分组
epoch_losses = {0: [], 1: [], 2: []}
for i, epoch in enumerate(epochs):
    epoch_losses[epoch].append(losses[i])

# 绘制每个 epoch
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
print(f"✅ 保存图表到: results/training_loss_by_epoch.png")

# ========================================
# 图4: 平滑损失曲线（移动平均）
# ========================================
def moving_average(data, window_size=50):
    """计算移动平均"""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(12, 6))

# 原始损失（半透明）
plt.plot(steps, losses, linewidth=0.5, alpha=0.3, color='gray', label='Original')

# 平滑损失
window = 50
smoothed_losses = moving_average(losses, window)
smoothed_steps = steps[window-1:]
plt.plot(smoothed_steps, smoothed_losses, linewidth=2, color='#9b59b6', label=f'Moving Avg (window={window})')

plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Smoothed Training Loss Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# 标记 epoch 边界
for epoch_step in epoch_boundaries:
    plt.axvline(x=epoch_step, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
plt.tight_layout()
plt.savefig('results/training_loss_smoothed.png', dpi=300, bbox_inches='tight')
print(f"✅ 保存图表到: results/training_loss_smoothed.png")

# ========================================
# 统计信息
# ========================================
print("\n" + "="*60)
print("训练损失统计")
print("="*60)

print(f"\n初始损失: {losses[0]:.4f}")
print(f"最终损失: {losses[-1]:.4f}")
print(f"损失降低: {losses[0] - losses[-1]:.4f} ({(losses[0] - losses[-1])/losses[0]*100:.1f}%)")

print(f"\n每个 Epoch 的损失:")
for epoch in [0, 1, 2]:
    epoch_loss = epoch_losses[epoch]
    print(f"  Epoch {epoch+1}: 起始={epoch_loss[0]:.4f}, 结束={epoch_loss[-1]:.4f}, "
          f"平均={np.mean(epoch_loss):.4f}")

print(f"\n总训练步数: {len(steps)}")
print(f"Epoch 数量: 3")

plt.show()
print("\n✅ 所有图表已生成并保存！")
