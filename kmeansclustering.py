import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Load Iris dataset (ignoring the actual labels)
X, _ = load_iris(return_X_y=True)

# Create and train K-Means with 3 clusters
kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(X)

# Get cluster labels and cluster centers
cluster_labels = kmeans_model.labels_
centers = kmeans_model.cluster_centers_

# Visualize clustering results using the first two features (sepal length and sepal width)
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', marker='o', edgecolor='k', label='Data points')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Means Clustering on Iris Data (First Two Features)')
plt.legend()
plt.show()