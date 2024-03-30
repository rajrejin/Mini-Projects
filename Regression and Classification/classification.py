import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Separate features (X) and labels (y)
data = pd.read_csv('D:/FAU/4. WS 23/DSS/Exercises\My-Projects/Regression and Classification/classification.csv')
x = data[['x1', 'x2']]
y = data['label']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
classifier = SVC(kernel='linear')

# Train the classifier on the training data
classifier.fit(x_train, y_train)

# Print the coefficients (weights) assigned to the features
print("Coefficients (weights):", classifier.coef_)

# Print the intercept (bias term)
print("Intercept (bias):", classifier.intercept_)

# Make predictions on the test data
predictions = classifier.predict(x_test)

# Plot the classified output and decision boundary
plt.figure(figsize=(8, 6))

# Plot the data points color-coded by their labels
plt.scatter(x_test[y_test==0]['x1'], x_test[y_test==0]['x2'], color='blue', edgecolors='k', marker='o', label='Class 0')
plt.scatter(x_test[y_test==1]['x1'], x_test[y_test==1]['x2'], color='red', edgecolors='k', marker='o', label='Class 1')

# Create a meshgrid to plot the decision boundary
h = .02  # Step size in the mesh
x_min, x_max = x_test['x1'].min() - 1, x_test['x1'].max() + 1
y_min, y_max = x_test['x2'].min() - 1, x_test['x2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot decision boundary
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='black')

plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()