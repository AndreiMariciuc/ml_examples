import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

digits = load_digits()
X, y = digits.data, digits.target

fig, axes = plt.subplots(1, 10, figsize=(12, 1.5))
for i, ax in enumerate(axes):
    ax.imshow(digits.images[i], cmap="gray_r")
    ax.set_title(y[i]);
    ax.axis("off")
plt.show()

# PCA: squeeze 64 dimensions into 2 so we can plot them
X_2d = PCA(n_components=2).fit_transform(X)

plt.figure(figsize=(7, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab10", s=8)
plt.colorbar(scatter, label="true digit")
plt.title("Digits projected to 2D with PCA")
plt.show()

# k-means: find 10 groups WITHOUT looking at the labels
kmeans = KMeans(n_clusters=10, n_init=10, random_state=0)
clusters = kmeans.fit_predict(X)

print("Agreement between clusters and true digits (ARI, 1 = perfect):",
      round(adjusted_rand_score(y, clusters), 3))

# What does each cluster "centre" look like?
fig, axes = plt.subplots(1, 10, figsize=(12, 1.5))
for i, ax in enumerate(axes):
    ax.imshow(kmeans.cluster_centers_[i].reshape(8, 8), cmap="gray_r")
    ax.set_title(f"c{i}")
    ax.axis("off")
plt.show()
