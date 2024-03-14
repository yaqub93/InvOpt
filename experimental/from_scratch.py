import cvxpy as cp
import numpy as np

# Generate some sample data
np.random.seed(0)
n = 100
x_data = np.random.rand(n)
y_data = np.random.rand(n)

import pickle
file_path = "dataset/dataset_all_samso_belt_length_normalized.pkl"
with open(file_path, 'rb') as file:
    dataset_raw = pickle.load(file)
feature_names = list(dataset_raw.keys())
print(feature_names)
print("original size:",len(dataset_raw[feature_names[0]]))
import random 

# Sample indices to select elements from lists
sample_indices = random.sample(range(len(dataset_raw[feature_names[0]])), k=20000)  # Change 'k' to the desired sample size

# Sample elements from both lists using the sampled indices
dataset = {}
for feature_name in feature_names:
    dataset[feature_name] = [dataset_raw[feature_name][i] for i in sample_indices]

data_size = len(dataset[feature_names[0]])
feature_values = []
for key,val in dataset.items():
    feature_values.append(val)
feature_values = np.array(feature_values).T

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(feature_values, test_size=0.1)
#print(X_train.shape)
#S_train = X_train[:,1]
#X_train = X_train[:,0]

print(feature_names)

x_data = X_train[:,0]
y_data = np.sqrt(X_train[:,6]**2+X_train[:,7]**2)
print(np.min(x_data), np.max(x_data), np.mean(x_data**2))
print(np.min(y_data), np.max(y_data), np.mean(y_data**2))
#S_test = X_test[:,1]
#X_test = X_test[:,0]

# Define variables
alpha = cp.Variable()

# Define objective function
objective = cp.Minimize(cp.sum_squares(alpha * x_data**2 + (1-alpha) * y_data**2))

constraints = [alpha >=0.0 , alpha <= 1.0]
# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Display the results
optimal_alpha = alpha.value
#optimal_beta = beta.value
print("Optimal alpha:", optimal_alpha)
#print("Optimal beta:", optimal_beta)

optimal_xs = []
for y_d in y_data:
    x = cp.Variable()
    objective = cp.Minimize(optimal_alpha * x**2 + (1-optimal_alpha) * y_d**2)
    problem = cp.Problem(objective)

    problem.solve()
    optimal_x = x.value
    optimal_xs.append(optimal_x)
print(np.linalg.norm(np.array(optimal_xs)-x_data))

optimal_ys = []
for x_d in x_data:
    y = cp.Variable()
    objective = cp.Minimize(optimal_alpha * x_d**2 + (1-optimal_alpha) * y**2)
    problem = cp.Problem(objective)

    problem.solve()
    optimal_y = y.value
    optimal_ys.append(optimal_y)
print(np.linalg.norm(np.array(optimal_ys)-y_data))
