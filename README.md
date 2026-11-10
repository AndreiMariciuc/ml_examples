# Introduction to Machine Learning in Python

---

## Contents

1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [What to know beforehand](#2-what-to-know-beforehand)
3. [The Python ML toolbox](#3-the-python-ml-toolbox)
4. [Environment setup](#4-environment-setup)
5. [Core concepts (read before the experiments)](#5-core-concepts-read-before-the-experiments)
6. [Hands-on experiments](#6-hands-on-experiments)
7. [Further resources](#9-further-resources)

---

## 1. What is Machine Learning?

In classical programming **you write the rules**:

```
data + rules  ──►  answers
```

In machine learning **the computer learns the rules from examples**:

```
data + answers  ──►  rules (a "model")
```

The learned model is then used to make predictions on **new, unseen data**.

| Type | What we have | Goal | Example |
|---|---|---|---|
| **Supervised – classification** | inputs + labels (categories) | predict a category | spam / not spam, digit 0–9 |
| **Supervised – regression** | inputs + numeric targets | predict a number | house price, temperature |
| **Unsupervised** | inputs only | find structure | customer groups, compressing data |

This introduction focuses on **supervised** learning, with a short look at **unsupervised** learning.

---

## 2. What to know beforehand

### 2.1 Python (must have)

- Variables, types: `int`, `float`, `str`, `bool`
- Collections: `list`, `tuple`, `dict`
- Control flow: `if`, `for`, `while`
- Functions (`def`), arguments, return values
- Importing libraries: `import numpy as np`, `from sklearn.datasets import load_iris`
- Basic classes (`class`, `__init__`, `self`): needed for PyTorch models
- Reading error messages (tracebacks)

**Quick self-check:** can you write a function that takes a list of numbers and returns the mean and the maximum, without using any library? If yes, you are ready.

### 2.2 Mathematics (intuition is enough)

| Topic | What you need | Where it shows up |
|---|---|---|
| **Linear algebra** | vectors, matrices, dot product, matrix shape `(rows, cols)` | every dataset is a matrix `X` of shape `(n_samples, n_features)` |
| **Statistics** | mean, variance, standard deviation, distributions | feature scaling, noise, evaluating models |
| **Probability** | probability of an event, conditional probability | classifiers output probabilities |
| **Calculus** | derivative = slope; the minimum of a function is where the slope is 0 | gradient descent (how models learn) |

You do **not** need to derive formulas by hand. You need to understand *what* they do.

### 2.3 The ML vocabulary

| Term | Meaning |
|---|---|
| **Sample / example** | one row of data (one flower, one image, one patient) |
| **Feature** | one input column (petal length, pixel value, age) |
| **Label / target** | what we want to predict (`y`) |
| **Model** | a function with adjustable **parameters** that maps `X → y` |
| **Training / fitting** | adjusting the parameters so predictions match the labels |
| **Loss function** | a number measuring how wrong the model is (lower = better) |
| **Hyperparameter** | a setting *you* choose before training (e.g. number of neighbours, learning rate) |
| **Prediction / inference** | using a trained model on new data |

### 2.4 Conventions used in Python ML code

```python
X  # features, 2D array, shape (n_samples, n_features)  – capital X because it's a matrix
y  # targets,  1D array, shape (n_samples,)              – lowercase because it's a vector
```

---

## 3. The Python ML toolbox

| Library | Role | Typical use |
|---|---|---|
| **NumPy** | fast arrays and math | the base of everything; datasets are NumPy arrays |
| **pandas** | tables (DataFrames) | loading CSV files, cleaning and exploring data |
| **Matplotlib / seaborn** | plotting | visualising data and results |
| **scikit-learn** | classical ML | ready-made models, preprocessing, evaluation |
| **PyTorch** | deep learning | neural networks, automatic differentiation, GPUs |
| **Jupyter** | interactive notebooks | experimenting, mixing code, plots and text |

### The scikit-learn API (learn this, it's used everywhere)

Every scikit-learn model follows the same three steps:

```python
model = SomeModel(hyperparameter=value)  # 1. create
model.fit(X_train, y_train)              # 2. learn from training data
y_pred = model.predict(X_test)           # 3. predict on new data
```

Preprocessing tools follow a similar pattern: `fit` (learn statistics), then `transform` (apply them).

---

## 4. Environment setup

We use [uv](https://docs.astral.sh/uv/) to manage Python and packages.

```bash
# install uv (once): https://docs.astral.sh/uv/getting-started/installation/
uv init ml-intro
cd ml-intro
uv add numpy pandas matplotlib seaborn scikit-learn torch
uv add --dev notebook
uv run jupyter notebook
```

> **CPU-only PyTorch (smaller download):** add this to `pyproject.toml` *before* `uv add torch`:
>
> ```toml
> [[tool.uv.index]]
> name = "pytorch-cpu"
> url = "https://download.pytorch.org/whl/cpu"
> explicit = true
>
> [tool.uv.sources]
> torch = [{ index = "pytorch-cpu" }]
> ```

Check that everything works by running this in the first notebook cell:

```python
import numpy, pandas, matplotlib, sklearn, torch
print("numpy", numpy.__version__, "| sklearn", sklearn.__version__, "| torch", torch.__version__)
```

All datasets used below **come bundled with scikit-learn**, so no downloads or internet access are needed.

---

## 5. Core concepts (read before the experiments)

### 5.1 The ML workflow

```
1. Get data  →  2. Explore  →  3. Split train/test  →  4. Preprocess
       →  5. Train model  →  6. Evaluate on test set  →  7. Improve & repeat
```

### 5.2 Train / test split: the most important idea

A model that is evaluated on the data it was trained on is like a student who has seen the exam questions beforehand. The score says nothing about real understanding.

- **Training set** (e.g. 75–80%): used to fit the model.
- **Test set** (e.g. 20–25%): locked away and used **only once**, at the end, to estimate performance on new data.
- **Validation set / cross-validation**: used to choose hyperparameters *without* touching the test set.

### 5.3 Underfitting vs overfitting

| | Underfitting | Good fit | Overfitting |
|---|---|---|---|
| Model | too simple | just right | too complex |
| Training error | high | low | very low |
| Test error | high | low | **high** |
| Analogy | didn't study | understood the material | memorised the answers |

The goal is **generalisation**: good performance on data the model has never seen.

### 5.4 How models learn: gradient descent

1. Start with random parameters.
2. Compute the **loss** (how wrong the predictions are).
3. Compute the **gradient**, the direction in which the loss increases fastest.
4. Take a small step in the **opposite** direction. The step size is the **learning rate**.
5. Repeat until the loss stops decreasing.

Think of walking downhill in fog: you only feel the slope under your feet, so you keep stepping downhill.

### 5.5 Evaluation metrics

- **Regression:** Mean Squared Error (MSE), Mean Absolute Error (MAE), R².
- **Classification:** accuracy, precision, recall, F1-score, confusion matrix.
- **Always compare against a baseline** (e.g. "always predict the most common class"). A 95% accuracy is not impressive if 95% of the samples belong to one class.

### 5.6 Data leakage

Leakage happens when information from the test set influences training. A typical example is scaling the *whole* dataset before splitting it. Results look great, then fail in reality. **Split first, then fit any preprocessing on the training data only.** scikit-learn `Pipeline`s do this for you.

---

## 6. Hands-on experiments

Each experiment is designed for **one Jupyter notebook** and takes about 15–30 minutes. Run the code cell by cell, and stop to discuss the questions marked 💬.

| # | Experiment | Key idea | Tools |
|---|---|---|---|
| 1 | Gradient descent from scratch | how learning works | NumPy |
| 2 | Your first classifier | fit / predict / evaluate | scikit-learn |
| 3 | Overfitting with polynomials | model complexity | scikit-learn |
| 4 | Why feature scaling matters | preprocessing, pipelines | scikit-learn |
| 5 | Accuracy is not everything | baselines, metrics, cross-validation | scikit-learn |
| 6 | Finding structure without labels | clustering, PCA | scikit-learn |
| 7 | Your first neural network | tensors, autograd, training loop | PyTorch |

---

### Experiment 1: Gradient descent from scratch (NumPy)

**Goal:** see that "learning" is just repeatedly adjusting numbers to reduce an error.

We create data from a known line `y = 3x + 2` plus noise, then check whether gradient descent can recover `w = 3` and `b = 2`.

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=100)
y = 3 * X + 2 + rng.normal(0, 1, size=100)   # true w = 3, b = 2

plt.scatter(X, y, s=10)
plt.xlabel("x"); plt.ylabel("y"); plt.title("Our data")
plt.show()
```

```python
w, b = 0.0, 0.0        # start with a "dumb" model
lr = 0.01              # learning rate
losses = []

for epoch in range(2000):
    y_pred = w * X + b
    error = y_pred - y
    loss = (error ** 2).mean()          # Mean Squared Error
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
```

```python
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(losses)
axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE"); axes[0].set_title("Loss curve")

axes[1].scatter(X, y, s=10, label="data")
xs = np.linspace(0, 10, 100)
axes[1].plot(xs, w * xs + b, color="red", label="learned line")
axes[1].legend(); axes[1].set_title("Fitted model")
plt.show()
```

💬 **Try & discuss:**
- Set `lr = 0.05`. What happens, and why? (Hint: the steps are too big and you "jump over" the valley.)
- Set `lr = 0.0001`. What happens?
- Why don't we get *exactly* `w = 3, b = 2`?

---

### Experiment 2: Your first classifier (scikit-learn)

**Goal:** learn the standard `fit → predict → evaluate` workflow on the classic Iris dataset (150 flowers, 4 measurements, 3 species).

```python
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print(df.head())
print(df["species"].value_counts())

sns.pairplot(df.drop(columns="target"), hue="species")
```

💬 *Looking at the plots: which features separate the species best? Which two species are hardest to tell apart?*

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("train:", X_train.shape, " test:", X_test.shape)

model = KNeighborsClassifier(n_neighbors=5)   # 1. create
model.fit(X_train, y_train)                   # 2. learn
y_pred = model.predict(X_test)                # 3. predict

print("Test accuracy:", accuracy_score(y_test, y_pred))
```

```python
# Predict a brand-new flower
new_flower = pd.DataFrame([[5.0, 3.4, 1.5, 0.2]], columns=X.columns)
print("Prediction:", iris.target_names[model.predict(new_flower)[0]])
print("Probabilities:", model.predict_proba(new_flower).round(2))
```

💬 **Try & discuss:**
- How does k-Nearest Neighbours decide? (It looks at the `k` closest training flowers and takes a vote.)
- Change `random_state` to a few other values. Does the accuracy change? Why?
- Swap in another model: `from sklearn.tree import DecisionTreeClassifier` or `from sklearn.linear_model import LogisticRegression`. Note that **only one line changes**.

---

### Experiment 3: Why feature scaling matters

**Goal:** discover that some models are very sensitive to the *scale* of the features, and learn to use `Pipeline`.

The Wine dataset has 13 chemical measurements. Some are around `0.1` and others around `1000`.

```python
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

wine = load_wine(as_frame=True)
X, y = wine.data, wine.target
print(X.describe().loc[["mean", "std", "min", "max"]].round(2).T)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)
```

💬 *Look at the `proline` and `hue` columns. KNN measures distances between samples, so which feature will dominate the distance?*

```python
raw_model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
scaled_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)).fit(X_train, y_train)

print(f"KNN without scaling: {raw_model.score(X_test, y_test):.3f}")
print(f"KNN with scaling:    {scaled_model.score(X_test, y_test):.3f}")
```

`StandardScaler` transforms each feature to have mean 0 and standard deviation 1. The pipeline makes sure the scaler is **fitted on the training data only** (no leakage).

---

### Experiment 4: Accuracy is not everything

**Goal:** use baselines, the confusion matrix, precision/recall and cross-validation to evaluate models properly.

Breast Cancer dataset: 569 tumours, 30 features, labels `0 = malignant`, `1 = benign`.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

data = load_breast_cancer()
X, y = data.data, data.target
print("Class counts:", dict(zip(data.target_names, np.bincount(y))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)
```

```python
# A "model" that always predicts the most frequent class
baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print(f"Baseline accuracy: {baseline.score(X_test, y_test):.3f}")
```

💬 *This "model" never looks at the data and still gets over 60% accuracy. What does that tell us about reporting accuracy alone?*

```python
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred, target_names=data.target_names))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=data.target_names)
```

- **Precision:** of the tumours predicted *malignant*, how many really were?
- **Recall:** of the truly *malignant* tumours, how many did we catch?

```python
# Cross-validation: a more reliable estimate than a single split
scores = cross_val_score(model, X_train, y_train, cv=5)
print("Fold accuracies:", scores.round(3))
print(f"Mean: {scores.mean():.3f} ± {scores.std():.3f}")
```

💬 **Try & discuss:**
- In a medical setting, which error is worse: a false "benign" or a false "malignant"? Which metric should we care about most?
- Why is the cross-validated mean more trustworthy than a single test score?

---

### Experiment 5: Finding structure without labels (unsupervised)

**Goal:** compress data with PCA and discover groups with k-means, all without using the labels.

The Digits dataset has 1797 handwritten digits, each an 8×8 image (64 pixel features).

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

digits = load_digits()
X, y = digits.data, digits.target

fig, axes = plt.subplots(1, 10, figsize=(12, 1.5))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap="gray_r"); ax.set_title(y[i]); ax.axis("off")
plt.show()
```

```python
# PCA: squeeze 64 dimensions into 2 so we can plot them
X_2d = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(7, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=8)
plt.colorbar(scatter, label="true digit")
plt.title("Digits projected to 2D with PCA"); plt.show()
```

```python
# k-means: find 10 groups WITHOUT looking at the labels
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
clusters = kmeans.fit_predict(X)

print("Agreement between clusters and true digits (ARI, 1 = perfect):",
      round(adjusted_rand_score(y, clusters), 3))

# What does each cluster "centre" look like?
fig, axes = plt.subplots(1, 10, figsize=(12, 1.5))
for i, ax in enumerate(axes):
    ax.imshow(kmeans.cluster_centers_[i].reshape(8, 8), cmap="gray_r"); ax.set_title(f"c{i}"); ax.axis("off")
plt.show()
```

---

### Experiment 6: Your first neural network (PyTorch)

**Goal:** understand tensors, automatic differentiation and the training loop that every deep learning model uses.

**Part A: autograd computes derivatives for you**

```python
import torch

x = torch.tensor(3.0, requires_grad=True)
f = x**2 + 2*x + 1          # f(x) = x² + 2x + 1
f.backward()                # compute df/dx
print("df/dx at x=3:", x.grad.item(), "  (by hand: 2x + 2 = 8)")
```

**Part B: a neural network that classifies handwritten digits**

```python
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)

X, y = load_digits(return_X_y=True)
X = X / 16.0                                    # pixels 0..16 → 0..1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# NumPy → PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

model = nn.Sequential(
    nn.Linear(64, 64),   # 64 pixels in → 64 hidden units
    nn.ReLU(),           # non-linearity
    nn.Linear(64, 10),   # → 10 scores, one per digit
)
print(model)
print("Number of parameters:", sum(p.numel() for p in model.parameters()))
```

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
history = []

for epoch in range(200):
    model.train()
    optimizer.zero_grad()              # 1. reset old gradients
    logits = model(X_train)            # 2. forward pass: predictions
    loss = loss_fn(logits, y_train)    # 3. how wrong are we?
    loss.backward()                    # 4. backward pass: compute gradients
    optimizer.step()                   # 5. update the weights

    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test).argmax(dim=1) == y_test).float().mean().item()
    history.append((loss.item(), test_acc))
    if epoch % 25 == 0:
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}  test acc={test_acc:.3f}")

print(f"\nFinal test accuracy: {history[-1][1]:.3f}")
```

```python
import matplotlib.pyplot as plt

losses, accs = zip(*history)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(losses); axes[0].set_title("training loss"); axes[0].set_xlabel("epoch")
axes[1].plot(accs);   axes[1].set_title("test accuracy"); axes[1].set_xlabel("epoch")
plt.show()
```

💬 **Try & discuss:**
- Compare the five training-loop steps with the gradient descent loop from **Experiment 1**. They are the same idea!
- Remove the `nn.ReLU()` line. Does accuracy drop? Why do we need non-linearities?
- What happens with `lr=1.0`? With `lr=0.0001`?
- How does this compare with `LogisticRegression` from scikit-learn on the same data?

---

## 7. Further resources

- **scikit-learn User Guide & tutorials:** https://scikit-learn.org/stable/user_guide.html
- **PyTorch "Learn the Basics":** https://pytorch.org/tutorials/beginner/basics/intro.html
- **NumPy for absolute beginners:** https://numpy.org/doc/stable/user/absolute_beginners.html
- **Google's Machine Learning Crash Course:** https://developers.google.com/machine-learning/crash-course
- **3Blue1Brown, *Neural Networks* (videos):** https://www.3blue1brown.com/topics/neural-networks
- **Book:** *An Introduction to Statistical Learning (with Python)*, free at https://www.statlearning.com/
