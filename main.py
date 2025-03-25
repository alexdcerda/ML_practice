# linear regression
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


# load regression dataset 
X, y = load_diabetes(return_X_y=True)

# print(X, y)

# split into training (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# create the linear regression model
model = LinearRegression()


# train the model
model.fit(X_train, y_train)


# make predictions
predictions = model.predict(X_test)



# Select one feature (let's use feature 0) for visualization
feature_index = 7

# Plot the actual data points
plt.scatter(X_test[:, feature_index], y_test, color='blue', label='Actual Data')

# Generate points for the regression line
X_line = np.linspace(X_test[:, feature_index].min(), X_test[:, feature_index].max(), 100).reshape(-1, 1)
X_line_full = np.zeros((100, X_test.shape[1]))  # Create full feature matrix
X_line_full[:, feature_index] = X_line[:, 0]    # Set the chosen feature
y_line = model.predict(X_line_full)

# Plot the regression line
plt.plot(X_line, y_line, color='red', label='Regression Line')

# Add labels and title
plt.xlabel(f'Feature {feature_index}')
plt.ylabel('Target (Diabetes Progression)')
plt.title('Linear Regression: Diabetes Progression vs. Feature')
plt.legend()

# Show the plot
plt.show()


print("Sample Predictions:", predictions[:5])
print("Actual Values :", y_test[:5])