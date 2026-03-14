import json

with open('history_val.json', 'r') as f:
    val_history = json.load(f)

with open('history_train.json', 'r') as f:
    train_history = json.load(f)

val_losses = [entry[0] for entry in val_history]
train_losses = [entry[0] for entry in train_history]

print('Validation Loss Over Time:')
print(f'Epoch 50:  {val_losses[49]:.4f}')
print(f'Epoch 100: {val_losses[99]:.4f}')
print(f'Epoch 150: {val_losses[149]:.4f}')
print(f'Epoch 200: {val_losses[199]:.4f}')
print(f'Epoch 250: {val_losses[249]:.4f}')
print(f'Epoch 300: {val_losses[299]:.4f}')
print()

best_loss = min(val_losses)
best_epoch = val_losses.index(best_loss) + 1
print(f'Best validation loss: {best_loss:.4f} at epoch {best_epoch}')
print(f'Final validation loss: {val_losses[-1]:.4f}')
print()

improvement = val_losses[199] - val_losses[299]
print(f'Improvement from epoch 200 to 300: {improvement:.4f}')
if abs(improvement) < 0.01:
    print('→ Minimal difference between 200 and 300 epochs')
elif improvement > 0:
    print('→ Model improved from epoch 200 to 300')
else:
    print('→ Model got worse from epoch 200 to 300')
print()

print('Train vs Val Loss:')
for epoch in [50, 100, 150, 200, 250, 300]:
    idx = epoch - 1
    gap = val_losses[idx] - train_losses[idx]
    print(f'Epoch {epoch:3d}: Train={train_losses[idx]:.4f}, Val={val_losses[idx]:.4f}, Gap={gap:+.4f}')
