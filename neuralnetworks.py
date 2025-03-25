import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load iris data and select two features: petal length (index 2) and petal width (index 3)
X, y = load_iris(return_X_y=True)
X_vis = X[:, [2, 3]]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_vis, y, test_size=0.2, random_state=42)

# Create and train the Neural Network
nn_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
nn_model.fit(X_train, y_train)

# Create a mesh grid over the feature space
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Use the neural network to predict over the grid
Z = nn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)

# Plot training and test points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', edgecolor='k', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='^', edgecolor='k', label='Test')
plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('MLP Neural Network Decision Boundary on Iris Data')
plt.legend()
plt.show()