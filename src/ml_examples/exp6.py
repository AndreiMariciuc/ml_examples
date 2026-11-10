import torch

x = torch.tensor(3.0, requires_grad=True)
f = x ** 2 + 2 * x + 1  # f(x) = x² + 2x + 1
f.backward()  # compute df/dx
print("df/dx at x=3:", x.grad.item(), "  (by hand: 2x + 2 = 8)")

import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)

X, y = load_digits(return_X_y=True)
X = X / 16.0  # pixels 0..16 → 0..1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# NumPy → PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(64, 64),  # 64 pixels in → 64 hidden units
    nn.ReLU(),  # non-linearity
    nn.Linear(64, 10),  # → 10 scores, one per digit
)
print(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
history = []

for epoch in range(200):
    model.train()
    optimizer.zero_grad()  # 1. reset old gradients
    logits = model(X_train)  # 2. forward pass: predictions
    loss = loss_fn(logits, y_train)  # 3. how wrong are we?
    loss.backward()  # 4. backward pass: compute gradients
    optimizer.step()  # 5. update the weights

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
    history.append((loss.item(), test_acc))
    if epoch % 25 == 0:
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}  test acc={test_acc:.3f}")

print(f"\nFinal test accuracy: {history[-1][1]:.3f}")

import matplotlib.pyplot as plt

losses, accs = zip(*history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(losses)
axes[0].set_title("training loss")
axes[0].set_xlabel("epoch")
axes[1].plot(accs)
axes[1].set_title("test accuracy")
axes[1].set_xlabel("epoch")
plt.show()
