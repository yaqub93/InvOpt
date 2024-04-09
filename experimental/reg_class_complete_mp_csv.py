"""
InvOpt package example: simultaneous regression and classification.

Dataset: Breast Cancer Wisconsin Prognostic (BCWP)
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(Prognostic)

Author: Pedro Zattoni Scroccaro
"""

from os.path import dirname, abspath
import sys
import time
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import multiprocessing

sys.path.append("src")  # nopep8
sys.path.append(dirname(abspath(__file__))+"/../../src")
sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8

sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
sys.path.append("src")
sys.path.append("examples")

print(dirname(abspath(__file__))+"/../../src")
print(sys.path)
import invopt as iop
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import kernel_ridge
from sklearn.metrics import r2_score

from utils_examples import colors, mean_percentiles, L1, L2

np.random.seed(0)

from multiprocessing import freeze_support
from time import sleep

freeze_support()

def theta_to_Qq(theta, n):
    """Extract Q and q from cost vector theta."""
    Q = theta[:n]
    q = theta[n:]
    return Q, q

def Qq_to_theta(Q, q):
    """Vectorize Q and q to create cost vector theta."""
    theta = np.concatenate((Q.flatten('F'), q))
    return theta

def quadratic_FOP(theta, s):
    """Forward optimization approach: quadratic program."""
    from gurobipy import Model, GRB, quicksum
    
    A, b, w = s
    n = 1
    t = 2
    Qxx, qx = theta_to_Qq(theta, n)

    # if len(theta) != (n**2 + n):
    #     raise Exception('Dimentions do not match!')

    mdl = Model('QP')
    mdl.setParam('OutputFlag', 0)
    x = mdl.addVars(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='x')

    mdl.setObjective(
        quicksum(Qxx[i]*x[i]*x[i] for i in range(n))
        + quicksum(qx[i]*w[i] for i in range(t)),
        GRB.MINIMIZE
    )

    mdl.addConstrs(
        quicksum(A[k, i]*x[i] for i in range(n)) <= b[k] for k in range(t)
    )

    mdl.optimize()

    if mdl.status == 2:
        x_opt = np.array([x[k].X for k in range(n)])
    else:
        raise Exception(
            f'Optimal solution not found. Gurobi status code = {mdl.status}.'
        )

    return x_opt


def phi1(w):
    """Feature mapping."""
    return np.array([])

def phi2(w):
    """Feature mapping."""
    return np.array(w) #np.array([w])

def phi(s, x):
    """Transform phi1 into phi for continuous quadratic case."""
    _, _, w = s
    return np.concatenate((x**2, w))

def load_data(train_test_slip, num_sample = 100):
    import pandas as pd
    file_path = "dataset/filtered_out3.csv"
    df = pd.read_csv(file_path)
    X = df[["cog_dot"]]
    #S = df[["distance", "DDV", "distance_dot", "DDV_dot"]]
    S = df[["distance", "DDV"]]
    X = X / np.abs(X).max()
    S = S / np.abs(S).max()
    #constraint = np.min(df["cog_dot"]), np.max(df["cog_dot"]), np.min(df["sog_dot"]), np.max(df["sog_dot"])
    constraint = np.min(df["cog_dot"]), np.max(df["cog_dot"])
    #print(feature_values)
    from sklearn.model_selection import train_test_split
    train_samples = int(num_sample*(1-train_test_slip))
    test_samples = int(num_sample*train_test_slip)
    X_train, X_test, S_train, S_test = train_test_split(X, S, train_size=train_samples, test_size=test_samples, random_state=42)
    S_train = S_train.values
    X_train = X_train.values
    S_test = S_test.values
    X_test = X_test.values
    return S_train, X_train, S_test, X_test, constraint

def IO_preprocessing(S_train, X_train, S_test, X_test, constraint):
    """Preprocess data for IO."""
    #A = -np.eye(1)
    #B = np.zeros((1, 1))
    A = np.array([-1,1]).reshape((2,1))
    B = np.array([np.abs(constraint[0]),constraint[1]])

    N_train = len(S_train)
    N_test = len(S_test)

    # Create dataset for IO
    dataset_train = []
    for i in range(N_train):
        s_hat = (A, B, S_train[i,:])
        x_hat = (X_train[i,:].T)
        dataset_train.append((s_hat, x_hat))

    dataset_test = []
    for i in range(N_test):
        s_hat = (A, B, S_test[i,:])
        x_hat = (X_test[i,:].T)
        dataset_test.append((s_hat, x_hat))

    return dataset_train, dataset_test

def dist_x(x1, x2):
    """Distance function for continous partof decision vector."""
    dx = L2(x1, x2)
    return dx

# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

train_test_slip = 0.1
runs = 3
max_sample = 500
min_sample = 10
num_samples = []

import copy 
sample = copy.deepcopy(min_sample)
while sample < max_sample:
    num_samples.append(sample)
    sample = sample*2

sample_indices = range(len(num_samples)) 

#resolution = 250
#num_samples = np.arange(resolution, total_sample+resolution, resolution)
add_y = True
#kappa = 10**3
kappa = 10**5
N_PROCESS = int(multiprocessing.cpu_count()/2)

print('')
print(f'train_test_slip = {train_test_slip}')
print(f'runs = {runs}')
print(f'add_y = {add_y}')
print(f'kappa = {kappa}')
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_diff_train_hist = np.empty((len(num_samples),runs))
y_diff_test_hist = np.empty((len(num_samples),runs))

y_diff_train_sk_hist = np.empty((len(num_samples),runs))
y_diff_test_sk_hist = np.empty((len(num_samples),runs))

#S_train, X_train, S_test, X_test, constraint = load_data(train_test_slip, num_sample = 1000)
#dataset_train, dataset_test = IO_preprocessing(
#    S_train, X_train, S_test, X_test, constraint
#)

#print(dataset_train)

#theta_IO = iop.ais_quadratic(dataset_train,
#                                        reg_param=kappa,
#                                        add_dist_func_y=add_y)
#print(theta_IO)

#S_train, X_train, S_test, X_test, constraint = load_data(train_test_slip, num_sample = 100)
#dataset_train, dataset_test = IO_preprocessing2(
#    S_train, X_train, S_test, X_test, constraint
#)

#theta_IO = iop.continuous_quadratic(dataset_train,
#                                    phi1,
#                                    add_dist_func_y=add_y,
#                                    reg_param=kappa)

def process_run(args):
    idx, num_sample, run = args
    S_train, X_train, S_test, X_test, constraint = load_data(train_test_slip, num_sample = num_sample)
    dataset_train, dataset_test = IO_preprocessing(
        S_train, X_train, S_test, X_test, constraint
    )

    theta_IO = iop.continuous_quadratic(dataset_train,
                                           phi1=phi1,
                                           phi2=phi2,
                                           reg_param=kappa,
                                           add_dist_func_y=add_y)
    print("theta_IO", theta_IO)
    y_diff_train, _ = iop.evaluate(theta_IO, dataset_train, quadratic_FOP, dist_x, phi=phi)
    y_diff_test, _ = iop.evaluate(theta_IO, dataset_test, quadratic_FOP, dist_x, phi=phi)

    reg = kernel_ridge.KernelRidge()
    reg.fit(S_train, X_train)
    #y_diff_train_sk = np.mean(np.abs(reg.predict(S_train) - X_train.reshape(N,1)))
    #y_diff_test_sk = np.mean(np.abs(reg.predict(S_test) - X_test.reshape(P,1)))
    y_diff_train_sk = np.mean(np.sqrt((reg.predict(S_train) - X_train)**2))
    y_diff_test_sk = np.mean(np.sqrt((reg.predict(S_test) - X_test)**2))

    return idx, run, y_diff_train, y_diff_test, y_diff_train_sk, y_diff_test_sk, theta_IO, reg.predict(S_train), X_train, reg.predict(S_test), X_test

tic = time.time()
pool = multiprocessing.Pool(processes=N_PROCESS)  # Create a pool of processes

run_args = []
for i,num_sample in enumerate(num_samples):
    for run in range(runs):
        run_args.append([i,num_sample, run])
"""
results = []
for args in run_args:
    idx, num_sample, run = args
    S_train, X_train, S_test, X_test, constraint = load_data(train_test_slip, num_sample = num_sample)
    dataset_train, dataset_test = IO_preprocessing(
        S_train, X_train, S_test, X_test, constraint
    )

    theta_IO = iop.continuous_quadratic(dataset_train,
                                           phi1=phi1,
                                           reg_param=kappa,
                                           add_dist_func_y=add_y)
    print("theta_IO", theta_IO)
    y_diff_train, _ = iop.evaluate(theta_IO, dataset_train, quadratic_FOP, dist_x, phi=phi)
    y_diff_test, _ = iop.evaluate(theta_IO, dataset_test, quadratic_FOP, dist_x, phi=phi)

    reg = kernel_ridge.KernelRidge()
    reg.fit(S_train, X_train)
    #y_diff_train_sk = np.mean(np.abs(reg.predict(S_train) - X_train.reshape(N,1)))
    #y_diff_test_sk = np.mean(np.abs(reg.predict(S_test) - X_test.reshape(P,1)))
    y_diff_train_sk = np.mean(np.sqrt((reg.predict(S_train) - X_train)**2))
    y_diff_test_sk = np.mean(np.sqrt((reg.predict(S_test) - X_test)**2))

    r2_train = r2_score(reg.predict(S_train), X_train)
    r2_test = r2_score(reg.predict(S_test), X_test)
    result = (idx, run, y_diff_train, y_diff_test, y_diff_train_sk, y_diff_test_sk, theta_IO, r2_train, r2_test)
    results.append(result)
"""
results = pool.map(process_run, run_args)  # Map the function to each run using the pool
pool.close()  # Close the pool to prevent any more tasks from being submitted to it
pool.join()
toc = time.time()
print(f"Simulation time = {round(toc-tic,2)} seconds")

theta_IOs = {}
for result in results:
    y_diff_train_hist[result[0], result[1]] = result[2]
    y_diff_test_hist[result[0], result[1]] = result[3]
    y_diff_train_sk_hist[result[0], result[1]] = result[4]
    y_diff_test_sk_hist[result[0], result[1]] = result[5]
    theta_IOs[(result[0], result[1])] = result[6]

for i,num_sample in enumerate(num_samples):
    print("================================")
    print(i)
    print(f'y_diff train error = {mean_percentiles(y_diff_train_hist[i])[0]}')
    print(f'y_diff test error = {mean_percentiles(y_diff_test_hist[i])[0]}')
    print('')
    print(f'SK y_diff train error = {mean_percentiles(y_diff_train_sk_hist[i])[0]}')
    print(f'SK y_diff test error = {mean_percentiles(y_diff_test_sk_hist[i])[0]}')

import pickle 
# Open file in binary write mode
with open("results.pkl", 'wb') as f:
    # Dump data to the file
    pickle.dump((num_samples, results, theta_IOs), f)

IO_means = []
IO_low = []
IO_high = []
sk_means = []
sk_low = []
sk_high = []
for i in sample_indices:
    IO_means.append(mean_percentiles(y_diff_test_hist)[0])
    IO_low.append(mean_percentiles(y_diff_test_hist)[1])
    IO_high.append(mean_percentiles(y_diff_test_hist)[2])
    sk_means.append(mean_percentiles(y_diff_test_sk_hist)[0])
    sk_low.append(mean_percentiles(y_diff_test_sk_hist)[1])
    sk_high.append(mean_percentiles(y_diff_test_sk_hist)[2])

# Path to the pickle file
file_path = 'results_processed.pickle'

# Open file in binary write mode
with open(file_path, 'wb') as f:
    # Dump data to the file
    pickle.dump((num_samples, IO_means, IO_low, IO_high, sk_means, sk_low, sk_high, theta_IOs), f)
    
"""

IO_means = []
IO_percentile_low = []
IO_percentile_high = []
sk_means = []
sk_percentile_low = []
sk_percentile_high = []
for i,num_sample in enumerate(num_samples):
    IO_means.append(mean_percentiles(y_diff_test_hist[i])[0])
    IO_percentile_low.append(mean_percentiles(y_diff_test_hist[i])[1])
    IO_percentile_high.append(mean_percentiles(y_diff_test_hist[i])[2])
    sk_means.append(mean_percentiles(y_diff_test_sk_hist[i])[0])
    sk_percentile_low.append(mean_percentiles(y_diff_test_sk_hist[i])[1])
    sk_percentile_high.append(mean_percentiles(y_diff_test_sk_hist[i])[2])


import matplotlib.pyplot as plt

IO_means = np.array(IO_means)
IO_percentile_low = np.array(IO_percentile_low)
IO_percentile_high = np.array(IO_percentile_high)
sk_means = np.array(sk_means)
sk_percentile_low = np.array(sk_percentile_low)
sk_percentile_high = np.array(sk_percentile_high)
# Plot
plt.figure(figsize=(8, 6))

# Plot the mean
plt.plot(num_samples, IO_means, label='Inverse Optimization (ASL)', color='blue')

# Plot the 25th and 75th percentiles as error bars
plt.errorbar(num_samples, IO_means, yerr=[IO_means - IO_percentile_low, IO_percentile_high - IO_means],
             fmt='o', color='blue', ecolor='lightblue', elinewidth=3, capsize=0)
# Plot the mean
plt.plot(num_samples, sk_means, label='Ridge Regression', color='orange')

# Plot the 25th and 75th percentiles as error bars
plt.errorbar(num_samples, sk_means, yerr=[sk_means - sk_percentile_low, sk_percentile_high - sk_means],
             fmt='o', color='orange', ecolor='lightorange', elinewidth=3, capsize=0)


plt.title('Inverse Optimization vs Ridge Regression')
plt.xlabel('Number of training data')
plt.ylabel('RMSE(x)')
plt.legend()
plt.grid()

# Save the figure as a PNG file
plt.savefig('comparison_io_sk.png')

plt.show()
"""
