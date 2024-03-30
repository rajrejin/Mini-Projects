import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

# Read data from the CSV file
data = pd.read_csv('D:/FAU/4. WS 23/DSS/Exercises\My-Projects/Regression and Classification/regression.csv')

# Extract x and y data from the DataFrame
x_data = data['x1']
y_data = data['x2']

# Define the function to fit (in this case, a polynomial of degree 3)
def func(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d

# Use curve_fit to find the parameters that minimize the difference between the actual and predicted y values
params, covariance = curve_fit(func, x_data, y_data)

# Extract the fitted parameters
a, b, c, d = params
print('a = ', a, 'b = ', b, 'c = ', c, 'd = ', d)

# Generate points for the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = func(x_fit, a, b, c, d)

# Plot the original data and the fitted curve
plt.scatter(x_data, y_data, label='Original Data')
plt.plot(x_fit, y_fit, label='Fitted Curve', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()