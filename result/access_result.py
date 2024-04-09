import pickle

# Assuming you have a file named 'data.pickle' containing a pickled object
with open('2024_04_09_12_56/results.pkl', 'rb') as f:
    # Load the pickled object from the file
    loaded_object = pickle.load(f)

# Now 'loaded_object' contains the deserialized object
num_samples, results, theta_IOs = loaded_object 

theta_IOs = {}
x_trains = []
x_train_preds = []
x_tests = []
x_test_preds = []
for result in results:
    theta_IOs[(result[0], result[1])] = result[6]
    x_train_preds.append(result[7])
    x_trains.append(result[8])
    x_test_preds.append(result[9])
    x_tests.append(result[10])
    
import matplotlib.pyplot as plt  

plt.scatter(x_trains[-1], x_train_preds[-1],label="train")
plt.scatter(x_tests[-1], x_test_preds[-1],label="test")
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.legend()
plt.savefig("result.png")
    