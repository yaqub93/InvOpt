import pickle
import numpy as np 

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

from utils_examples import colors, mean_percentiles, L1, L2

# Path to the pickle file
file_path = 'result/results.pkl'

# Open file in binary read mode
with open(file_path, 'rb') as f:
    # Load data from the file
    results = pickle.load(f)

runs = 3
total_sample = 1000
resolution = 100
num_samples = np.arange(resolution, total_sample+resolution, resolution)

y_diff_train_hist = np.empty((len(num_samples),runs))
y_diff_test_hist = np.empty((len(num_samples),runs))

y_diff_train_sk_hist = np.empty((len(num_samples),runs))
y_diff_test_sk_hist = np.empty((len(num_samples),runs))
for result in results:
    y_diff_train_hist[result[0], result[1]] = result[2]
    y_diff_test_hist[result[0], result[1]] = result[3]
    y_diff_train_sk_hist[result[0], result[1]] = result[4]
    y_diff_test_sk_hist[result[0], result[1]] = result[5]


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
#plt.fill_between(num_samples, IO_means, yerr=[IO_means - IO_percentile_low, IO_percentile_high - IO_means],
#             fmt='o', color='blue', ecolor='lightblue', elinewidth=3, capsize=0)
plt.fill_between(num_samples, IO_percentile_low, IO_percentile_high,alpha=0.3,
                facecolor="blue")
# Plot the mean
plt.plot(num_samples, sk_means, label='Ridge Regression', color='green')

light_orange_rgb = (1.0, 0.8, 0.6)
# Plot the 25th and 75th percentiles as error bars
plt.fill_between(num_samples, sk_percentile_low, sk_percentile_high,alpha=0.3,
                facecolor="green")


plt.title('Inverse Optimization vs Ridge Regression')
plt.xlabel('Number of training data')
plt.ylabel('RMSE(x)')
plt.legend()
plt.grid()

# Save the figure as a PNG file
plt.savefig('comparison_io_sk.png')

plt.show()