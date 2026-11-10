import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=100)
y = 3 * X + 2 + rng.normal(0, 1, size=100)  # true w = 3, b = 2

plt.scatter(X, y, s=10)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Our data")
plt.show()

w, b = 0.0, 0.0  # start with a "dumb" model
lr = 0.01  # learning rate
losses = []

for epoch in range(2000):
    y_pred = w * X + b
    error = y_pred - y
    loss = (error ** 2).mean()  # Mean Squared Error
    losses.append(loss)

    # gradients of the MSE with respect to w and b
    dw = 2 * (error * X).mean()
    db = 2 * error.mean()

    # step downhill
    w -= lr * dw
    b -= lr * db

    if epoch % 400 == 0:
        print(f"epoch {epoch:4d}  loss={loss:8.3f}  w={w:.3f}  b={b:.3f}")

print(f"\nLearned: w={w:.3f}, b={b:.3f}   (true: w=3, b=2)")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(losses)
axes[0].set_yscale("log")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("MSE")
axes[0].set_title("Loss curve")

axes[1].scatter(X, y, s=10, label="data")
xs = np.linspace(0, 10, 100)
axes[1].plot(xs, w * xs + b, color="red", label="learned line")
axes[1].legend()
axes[1].set_title("Fitted model")
plt.show()
