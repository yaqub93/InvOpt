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
    _, _, w = s
    u = len(w)
    Qyy = theta[0]
    Q = theta[1:]

    #if len(theta) != (1 + cQ + cQ):
    #    raise Exception('Dimentions do not match!')

    mdl = gp.Model('MIQP')
    mdl.setParam('OutputFlag', 0)
    y = mdl.addVar(vtype=gp.GRB.CONTINUOUS, name='y')
    #z = mdl.addVar(vtype=gp.GRB.BINARY, name='z')

    phi1_w = np.concatenate([w, np.array([1])])

    mdl.setObjective(
        #Qyy*y**2 + y*gp.quicksum(Q[i]*phi1_w[i] for i in range(u+1)), gp.GRB.MINIMIZE
        Qyy*y**2 + gp.quicksum(Q[i]*phi1_w[i] for i in range(u+1)), gp.GRB.MINIMIZE
    )

    mdl.optimize()

    y_opt = np.array([y.X])

    return y_opt


def load_data(train_test_slip):
    """Load and preprosses BCWP data."""
    # dataset = np.genfromtxt(path_to_invopt + r'\examples\mixed_integer_quadratic\breast-cancer-wisconsin-data\wpbc_data.csv',   # nopep8
    #                         delimiter=',')

    """
    dataset = np.genfromtxt(
        dirname(abspath(__file__))+r'/breast-cancer-wisconsin-data/wpbc_data.csv', delimiter=','
    )

    # Signal-response data
    S = dataset[:, 2:].copy()
    X = dataset[:, :2].copy()
    X[:, [1, 0]] = X[:, [0, 1]]

    N, m = S.shape
    N_train = round(N*(1-train_test_slip))
    # N_test = round(N*train_test_slip)

    train_idx = np.random.choice(N, N_train, replace=False)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True

    # Split data into train/test
    S_train = S[train_mask, :].copy()
    X_train = X[train_mask, :].copy()
    S_test = S[~train_mask, :].copy()
    X_test = X[~train_mask, :].copy()
    
    print(S_train.shape, X_train.shape)
    print(np.min(X_train[:,0]), np.max(X_train[:,0]))
    print(np.min(X_train[:,1]), np.max(X_train[:,1]))

    return S_train, X_train, S_test, X_test
    """

    import pickle
    file_path = "dataset/dataset_all_samso_belt_length_normalized.pkl"
    with open(file_path, 'rb') as file:
        dataset_raw = pickle.load(file)
    feature_names = list(dataset_raw.keys())
    print(feature_names)
    print("original size:",len(dataset_raw[feature_names[0]]))
    import random 

    # Sample indices to select elements from lists
    sample_indices = random.sample(range(len(dataset_raw[feature_names[0]])), k=1000)  # Change 'k' to the desired sample size

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
    A = -np.eye(1)
    B = np.zeros((1, 1))

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
y_diff_train_hist = np.empty(runs)
y_diff_test_hist = np.empty(runs)

y_diff_train_sk_hist = np.empty(runs)
y_diff_test_sk_hist = np.empty(runs)
"""
tic = time.time()
for run in range(runs):
    #np.random.seed(run)  # Make sure the same random slipt is used
    S_train, X_train, S_test, X_test = load_data(train_test_slip)
    dataset_train, dataset_test = IO_preprocessing(
        S_train, X_train, S_test, X_test
    )

    theta_IO = iop.continuous_quadratic(dataset_train,
                                           phi1=phi1,
                                           reg_param=kappa,
                                           add_dist_func_y=add_y)

    y_diff_train, _ = iop.evaluate(theta_IO, dataset_train, FOP_QP, dist_x)
    y_diff_test, _ = iop.evaluate(theta_IO, dataset_test, FOP_QP, dist_x)

    y_diff_train_hist[run] = y_diff_train
    y_diff_test_hist[run] = y_diff_test

    # Scikit-learn regressors
    # reg = svm.SVR()
    # reg = linear_model.LinearRegression()
    reg = kernel_ridge.KernelRidge()
    # reg = neural_network.MLPRegressor(max_iter=3000)
    # reg = neighbors.KNeighborsRegressor()
    # reg = gaussian_process.GaussianProcessRegressor()
    # reg = tree.DecisionTreeRegressor()
    #print(S_train.shape, X_train.shape)
    N, = S_train.shape
    P, = S_test.shape
    reg.fit(S_train.reshape(N,1), X_train.reshape(N,1))
    y_diff_train_sk = np.mean(np.abs(reg.predict(S_train.reshape(N,1)) - X_train.reshape(N,1)))
    y_diff_test_sk = np.mean(np.abs(reg.predict(S_test.reshape(P,1)) - X_test.reshape(P,1)))
    y_diff_train_sk_hist[run] = y_diff_train_sk
    y_diff_test_sk_hist[run] = y_diff_test_sk

    print(f'{round(100*(run+1)/runs)}%')
"""

def process_run(args):
    run = args
    S_train, X_train, S_test, X_test = load_data(train_test_slip)
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
    y_diff_train_sk = np.mean(np.abs(reg.predict(S_train) - X_train.reshape(N,1)))
    y_diff_test_sk = np.mean(np.abs(reg.predict(S_test) - X_test.reshape(P,1)))

    return run, y_diff_train, y_diff_test, y_diff_train_sk, y_diff_test_sk

tic = time.time()
pool = multiprocessing.Pool(processes=N_PROCESS)  # Create a pool of processes

run_args = []
for run in range(runs):
    run_args.append([run])
results = pool.map(process_run, run_args)  # Map the function to each run using the pool
pool.close()  # Close the pool to prevent any more tasks from being submitted to it
pool.join()
toc = time.time()
print(f"Simulation time = {round(toc-tic,2)} seconds")

for result in results:
    y_diff_train_hist[result[0]] = result[1]
    y_diff_test_hist[result[0]] = result[2]
    y_diff_train_sk_hist[result[0]] = result[3]
    y_diff_test_sk_hist[result[0]] = result[4]

print(f'y_diff train error = {mean_percentiles(y_diff_train_hist)[0]}')
print(f'y_diff test error = {mean_percentiles(y_diff_test_hist)[0]}')
print('')
print(f'SK y_diff train error = {mean_percentiles(y_diff_train_sk_hist)[0]}')
print(f'SK y_diff test error = {mean_percentiles(y_diff_test_sk_hist)[0]}')
