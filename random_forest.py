import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)


predictions = rf_model.predict(X_test)


# Create figure with subplots
plt.figure(figsize=(15, 5))

# Plot 1: Decision Boundaries
plt.subplot(1, 2, 1)
feature1, feature2 = 0, 1

# Create mesh grid
x_min, x_max = X[:, feature1].min() - 1, X[:, feature1].max() + 1
y_min, y_max = X[:, feature2].min() - 1, X[:, feature2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# Get predictions
Z = rf_model.predict(np.c_[xx.ravel(), yy.ravel(),
                          np.zeros(xx.ravel().shape[0]),
                          np.zeros(xx.ravel().shape[0])])
Z = Z.reshape(xx.shape)

# Plot decision regions
plt.contourf(xx, yy, Z, alpha=0.4)
scatter = plt.scatter(X[:, feature1], X[:, feature2], c=y, 
                     alpha=0.8, edgecolor='black')
plt.xlabel(f'Feature {feature1}')
plt.ylabel(f'Feature {feature2}')
plt.title('Random Forest Decision Boundaries')
plt.legend(*scatter.legend_elements(), title="Classes")

# Plot 2: Feature Importance
plt.subplot(1, 2, 2)
importances = rf_model.feature_importances_
feature_names = [f'Feature {i}' for i in range(X.shape[1])]
plt.bar(feature_names, importances)
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


print("Predicted labels:", predictions[:5])
print("Actual labels   :", y_test[:5])