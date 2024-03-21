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

from utils_examples import colors, mean_percentiles, L1, L2

np.random.seed(0)


def FOP_QP(theta, s):
    """Forward optimization problem."""
    A, B, w = s
    u = len(w)
    t = 2
    Qyy = theta[0]
    Q = theta[1:]

    #if len(theta) != (1 + cQ + cQ):
    #    raise Exception('Dimentions do not match!')

    mdl = gp.Model('QP')
    mdl.setParam('OutputFlag', 0)
    y = mdl.addVar(vtype=gp.GRB.CONTINUOUS, name='y')
    #z = mdl.addVar(vtype=gp.GRB.BINARY, name='z')

    phi1_w = np.concatenate([w, np.array([1])])

    mdl.setObjective(
        #Qyy*y**2 + y*gp.quicksum(Q[i]*phi1_w[i] for i in range(u+1)), gp.GRB.MINIMIZE
        Qyy*y**2 + gp.quicksum(Q[i]*phi1_w[i] for i in range(u+1)), gp.GRB.MINIMIZE
    )
    for k in range(t):
        mdl.addConstr(
            (A[k,0]*y <= B[k])
        )

    mdl.optimize()

    y_opt = np.array([y.X])

    return y_opt


def load_data(train_test_slip, num_sample = 100):

    import pickle
    file_path = "dataset/dataset_all_samso_belt_length_normalized.pkl"
    with open(file_path, 'rb') as file:
        dataset_raw = pickle.load(file)
    feature_names = list(dataset_raw.keys())
    print(feature_names)
    print("original size:",len(dataset_raw[feature_names[0]]))
    import random 

    # Sample indices to select elements from lists
    sample_indices = random.sample(range(len(dataset_raw[feature_names[0]])), k=num_sample)  # Change 'k' to the desired sample size

    # Sample elements from both lists using the sampled indices
    dataset = {}
    for feature_name in feature_names:
        dataset[feature_name] = [dataset_raw[feature_name][i] for i in sample_indices]

    data_size = len(dataset[feature_names[0]])
    feature_values = []
    for key,val in dataset.items():
        feature_values.append(val)
    feature_values = np.array(feature_values).T
    #print(feature_values)
    from sklearn.model_selection import train_test_split

    X_train, X_test = train_test_split(feature_values, test_size=train_test_slip)
    #print(X_train.shape)
    S_train = X_train[:,1:]
    X_train = X_train[:,0]

    S_test = X_test[:,1:]
    X_test = X_test[:,0]
    return S_train, X_train, S_test, X_test

def IO_preprocessing(S_train, X_train, S_test, X_test):
    """Preprocess data for IO."""
    #A = -np.eye(1)
    #B = np.zeros((1, 1))
    A = np.array([-1,1]).reshape((2,1))
    B = np.array([2,2])

    N_train = len(S_train)
    N_test = len(S_test)

    # Create dataset for IO
    dataset_train = []
    for i in range(N_train):
        s_hat = (A, B, S_train[i,:])
        x_hat = (np.array([X_train[i]]))
        dataset_train.append((s_hat, x_hat))

    dataset_test = []
    for i in range(N_test):
        s_hat = (A, B, S_test[i,:])
        x_hat = (np.array([X_test[i]]))
        dataset_test.append((s_hat, x_hat))

    return dataset_train, dataset_test

def phi1(w):
    """Feature function."""
    #return (w, [1])
    return np.concatenate([w, np.array([1])])

def phi(s, x):
    """Transform phi1 and phi2 into phi."""
    _, _, w = s
    #return np.concatenate((x*phi1(w), phi2(w)))
    #return np.concatenate((np.kron(x, x), np.kron(phi1(w), x)))
    return np.concatenate((np.kron(x, x), phi1(w)**2))

def dist_x(x1, x2):
    """Distance function for continous partof decision vector."""
    dx = L2(x1, x2)
    return dx

# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

train_test_slip = 0.1
runs = 3
max_sample = 5000
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
N_PROCESS = int(multiprocessing.cpu_count())

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

def process_run(args):
    idx, num_sample, run = args
    S_train, X_train, S_test, X_test = load_data(train_test_slip, num_sample = num_sample)
    dataset_train, dataset_test = IO_preprocessing(
        S_train, X_train, S_test, X_test
    )

    theta_IO = iop.continuous_quadratic(dataset_train,
                                           phi1=phi1,
                                           reg_param=kappa,
                                           add_dist_func_y=add_y)
    print("theta_IO", theta_IO)
    y_diff_train, _ = iop.evaluate(theta_IO, dataset_train, FOP_QP, dist_x)
    y_diff_test, _ = iop.evaluate(theta_IO, dataset_test, FOP_QP, dist_x)

    reg = kernel_ridge.KernelRidge()
    N, = X_train.shape
    P, = X_test.shape
    reg.fit(S_train, X_train.reshape(N,1))
    #y_diff_train_sk = np.mean(np.abs(reg.predict(S_train) - X_train.reshape(N,1)))
    #y_diff_test_sk = np.mean(np.abs(reg.predict(S_test) - X_test.reshape(P,1)))
    y_diff_train_sk = np.mean(np.sqrt((reg.predict(S_train) - X_train.reshape(N,1))**2))
    y_diff_test_sk = np.mean(np.sqrt((reg.predict(S_test) - X_test.reshape(P,1))**2))

    return idx, run, y_diff_train, y_diff_test, y_diff_train_sk, y_diff_test_sk, theta_IO

tic = time.time()
pool = multiprocessing.Pool(processes=N_PROCESS)  # Create a pool of processes

run_args = []
for i,num_sample in enumerate(num_samples):
    for run in range(runs):
        run_args.append([i,num_sample, run])
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
