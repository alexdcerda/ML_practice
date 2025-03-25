import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

predictions = tree_model.predict(X_test)

print("Predicted labels:", predictions[:5])
print("Actual labels   :", y_test[:5])


# Select two features for visualization
feature1, feature2 = 0, 1  # Using first two features

# Create mesh grid for decision boundary
x_min, x_max = X[:, feature1].min() - 1, X[:, feature1].max() + 1
y_min, y_max = X[:, feature2].min() - 1, X[:, feature2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Make predictions on mesh grid
Z = tree_model.predict(np.c_[xx.ravel(), yy.ravel(),
                           np.zeros(xx.ravel().shape[0]),
                           np.zeros(xx.ravel().shape[0])])
Z = Z.reshape(xx.shape)

# Create plot
plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.4)

# Plot the training points
scatter = plt.scatter(X[:, feature1], X[:, feature2], c=y, 
                     alpha=0.8, edgecolor='black')

# Add labels and title
plt.xlabel(f'Feature {feature1}')
plt.ylabel(f'Feature {feature2}')
plt.title('Decision Tree Classification Boundaries (Iris Dataset)')
plt.legend(*scatter.legend_elements(), title="Classes")

plt.show()