import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

# features: (500, 768)
# labels: (500,) 每个样本一个类别标签
features = torch.randn(500, 768)
labels = torch.randint(0, 5, (500,))

X = features.numpy()
y = labels.numpy()

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(X_2d[y == label, 0], X_2d[y == label, 1], label=f'Class {label}', alpha=0.6)

plt.legend()
plt.title("t-SNE Visualization of High-Dimensional Features")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()
