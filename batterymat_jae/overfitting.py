import json

with open('history_train.json', 'r') as f:
    train_history = json.load(f)
with open('history_val.json', 'r') as f:
    val_history = json.load(f)

train_losses = [entry[0] for entry in train_history]
val_losses = [entry[0] for entry in val_history]

print('Train vs Val Loss:')
for epoch in [50, 100, 150, 200, 250, 300]:
    idx = epoch - 1
    print(f'Epoch {epoch:3d}: Train={train_losses[idx]:.4f}, Val={val_losses[idx]:.4f}, Gap={val_losses[idx]-train_losses[idx]:+.4f}')
