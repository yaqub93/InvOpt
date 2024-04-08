import pandas as pd
import numpy as np

def load_data(train_test_slip, num_sample = None):
    file_path = "dataset/filtered_samso_complete1.csv"
    #file_path = "dataset/filtered_out4.csv"
    df = pd.read_csv(file_path)
    #df = df[df["ship_type"] == "Tanker"]
    #df = df[(df["cog"] > 150)]
    #df = df[(df["theta"] > -55) & (df["theta"] < 55)]
    #df = df[df["scenario"] == "2fe5027c75"]
    #df = df[df["mmsi"] == 236501000]
    #df = df[df["mmsi"] == 244678000]
    #X = df[["cog_dot", "distance", "DDV", "distance_dot", "DDV_dot"]]
    df["distance"] = (df["distance"])**(-1)
    df["distance_dot"] = (df["distance_dot"])**(-1)
    X = df[["distance", "DDV", "distance_dot", "DDV_dot", "width", "length"]]
    y = df[["cog_dot", "sog_dot"]]
    #print(np.min(df["cog_dot"]),np.min(df["sog_dot"]))
    #X = X / np.abs(X).max()
    constraint = np.min(df["cog_dot"]), np.max(df["cog_dot"]), np.min(df["sog_dot"]), np.max(df["sog_dot"])
    #print(feature_values)
    from sklearn.model_selection import train_test_split
    
    if num_sample is None:
        num_sample = len(df)
        
    train_samples = int(num_sample*(1-train_test_slip))
    test_samples = int(num_sample*train_test_slip)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_samples, test_size=test_samples, random_state=42)
    #X_train = X_train.values
    #X_test = X_test.values
    return X_train, X_test, y_train, y_test, constraint


from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test, constraint = load_data(0.2)

# Importing necessary libraries
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

# Importing necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define a function to create a neural network regression model
def create_neural_network_regression_model():
    model = Sequential([
        Dense(128, activation='tanh', input_shape=(6,)),
        Dense(64, activation='tanh'),
        Dense(64, activation='tanh'),
        Dense(32, activation='tanh'),
        Dense(16, activation='tanh'),
        Dense(4, activation='tanh'),
        Dense(2)  # Output layer with 1 neuron for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# List of regression methods
regression_methods = {
    #'Neural Network Regression': create_neural_network_regression_model(),
    #'Linear Regression': LinearRegression(),
    'Kernel Ridge Regression': KernelRidge(),
    #'Ridge Regression': Ridge(),
    #'Lasso Regression': Lasso(),
    #'ElasticNet Regression': ElasticNet(),
    #'Support Vector Regression (SVR)': SVR(),
    #'Decision Tree Regression': DecisionTreeRegressor(),
    #'Random Forest Regression': RandomForestRegressor(),
    #'Gradient Boosting Regression': GradientBoostingRegressor(),
    #'AdaBoost Regression': AdaBoostRegressor(),
    #'K-Nearest Neighbors Regression': KNeighborsRegressor(),
    #'Gaussian Process Regression': GaussianProcessRegressor(),
    #'Naive Bayes Regression': GaussianNB(),
    #'Passive Aggressive Regression': PassiveAggressiveRegressor()
}
"""
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate random input features (X) uniformly distributed between 0 and 100
X = np.random.uniform(0, 100, size=(num_samples, 3))  # 3-dimensional input

# Generate random coefficients for linear equation
true_coefficients = np.random.uniform(-10, 10, size=(3,))  # 3-dimensional coefficients

# Generate random noise
noise = np.random.normal(0, 10, size=(num_samples,))

# Generate output labels (y) using linear equation: y = m*X + b + noise
y = np.dot(X, true_coefficients) + noise  # Dot product to calculate output

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the true coefficients
print("True Coefficients:", true_coefficients)

# Print the shapes of X_train, X_test, y_train, and y_test
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
"""
# Print the list of regression methods
for name, model in regression_methods.items():
    print(name)
    model.fit(X_train, y_train)

    # Making predictions on the test set
    predictions = model.predict(X_test)
    print(predictions.shape)
    # Evaluating the model
    from sklearn.metrics import mean_squared_error

    # Calculating MSE for each feature
    mse_feature1 = mean_squared_error(y_test["cog_dot"], predictions[:, 0])
    
    plt.hist(y_test["cog_dot"], label = "cog_dot", bins=500,alpha=0.5)
    plt.hist(y_test["cog_dot"]-predictions[:, 0], label = "error", bins=50,alpha=0.5)
    plt.legend()
    plt.grid()
    plt.show()
    
    mse_feature2 = mean_squared_error(y_test["sog_dot"], predictions[:, 1])

    plt.hist(y_test["sog_dot"], label = "sog_dot", bins=500,alpha=0.5)
    plt.hist(y_test["sog_dot"]-predictions[:, 1], label = "error", bins=50,alpha=0.5)
    plt.legend()
    plt.grid()
    plt.show()
    
    print("Mean Squared Error for Feature 1:", mse_feature1)
    print("Mean Squared Error for Feature 2:", mse_feature2)

    # Importing necessary libraries
    from sklearn.metrics import r2_score

    # Calculating R-squared
    r_squared1 = r2_score(y_test["cog_dot"], predictions[:, 0])
    r_squared2 = r2_score(y_test["sog_dot"], predictions[:, 1])

    print("R-squared 1:", r_squared1)
    print("R-squared 2:", r_squared2)
