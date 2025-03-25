from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# load a simple classification dataset (iris dataset)
X, y = load_iris(return_X_y=True)

# split into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create model
model = LogisticRegression(max_iter=200)

# train the model
model.fit(X_train, y_train)

# make prediction
predictions = model.predict(X_test)


# Select two features for visualization
feature1, feature2 = 0, 1  # First two features

# Create mesh grid
x_min, x_max = X[:, feature1].min() - 1, X[:, feature1].max() + 1
y_min, y_max = X[:, feature2].min() - 1, X[:, feature2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Get predictions for mesh grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel(), 
                       np.zeros(xx.ravel().shape[0]), 
                       np.zeros(xx.ravel().shape[0])])
Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4)

# Plot training points (fixed edgecolor parameter)
scatter = plt.scatter(X[:, feature1], X[:, feature2], c=y, 
                     alpha=0.8, edgecolor='black')

# Add labels and title
plt.xlabel(f'Feature {feature1}')
plt.ylabel(f'Feature {feature2}')
plt.title('Logistic Regression Decision Boundary (Iris Dataset)')
plt.legend(*scatter.legend_elements(), title="Classes")

# Show the plot
plt.show()

print("Predicted labels:", predictions[:5])
print("Actual labels   :", y_test[:5])