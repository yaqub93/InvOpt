import cvxpy as cp
import numpy as np
from sklearn.metrics import r2_score

def load_data(train_test_slip, num_sample = 100):
    import pandas as pd
    
    file_path = "dataset/filtered_samso_complete1.csv"
    #file_path = "dataset/filtered_samso_complete_time_sampled2.csv"
    df = pd.read_csv(file_path)
    #print(df['mmsi'].value_counts())
    #df = df[df["ship_type"] == "Tanker"]
    #df = df[(df["cog"] > 150)]
    #df = df[(df["theta"] > -55) & (df["theta"] < 55)]
    #df = df[df["scenario"] == "2fe5027c75"]
    #df = df[df["mmsi"] == 236501000]
    #df = df[df["mmsi"] == 244678000]
    #df = df[df["mmsi"] == 356234000]
    #print(df["scenario"])
    
    #X = df[["cog_dot", "distance", "DDV", "distance_dot", "DDV_dot"]]
    df["distance"] = (df["distance"])**(-1)
    df["distance_dot"] = (df["distance_dot"])**(-1)
    X = df[["cog_dot", "sog_dot", "distance", "DDV", "distance_dot", "DDV_dot", "width", "length"]]
    print(np.min(df["cog_dot"]),np.min(df["sog_dot"]))
    #X = X / np.abs(X).max()
    constraint = np.min(df["cog_dot"]), np.max(df["cog_dot"]), np.min(df["sog_dot"]), np.max(df["sog_dot"])
    #print(feature_values)
    from sklearn.model_selection import train_test_split
    
    if num_sample is None:
        num_sample = len(df)
        
    train_samples = int(num_sample*(1-train_test_slip))
    test_samples = int(num_sample*train_test_slip)
    X_train, X_test = train_test_split(X, train_size=train_samples, test_size=test_samples, random_state=42)
    #X_train = X_train.values
    #X_test = X_test.values
    return X_train, X_test, constraint

train_test_slip = 0.2
X_train, X_test, constraint = load_data(train_test_slip, num_sample = None) # 28687
#print(np.min(X_train[:][0]),np.min(X_train[:][1]))
#print(np.min(X_test[:][0]),np.min(X_test[:][1]))

#x_data = X_train[:,0]
#y_data = np.sqrt(X_train[:,6]**2+X_train[:,7]**2)
#print(np.min(x_data), np.max(x_data), np.mean(x_data**2))
#print(np.min(y_data), np.max(y_data), np.mean(y_data**2))
#S_test = X_test[:,1]
#X_test = X_test[:,0]

n = X_train.shape[1]
N = X_train.shape[0]

# Define variables
alpha = cp.Variable((n,n))

EXP = 3

theta_v_1 = cp.Variable()
theta_v_2 = cp.Variable()
theta_yaw_1 = cp.Variable()
theta_yaw_2 = cp.Variable()
theta_v_3 = cp.Variable()
theta_yaw_3 = cp.Variable()
theta_v_4 = cp.Variable()
theta_v_5 = cp.Variable()
theta_yaw_4 = cp.Variable()
theta_yaw_5 = cp.Variable()

# Define objective function
quad = 0
constraints = []
for i in range(N):
    vdot = X_train.iloc[i]["sog_dot"]
    yawdot = X_train.iloc[i]["cog_dot"]
    d_1 = X_train.iloc[i]["distance"]
    ddv = X_train.iloc[i]["DDV"]
    ddv_dot = X_train.iloc[i]["DDV_dot"]
    width = X_train.iloc[i]["width"]
    length = X_train.iloc[i]["length"]
    if EXP == 0:
        #vdot_ = theta_v_1*ddv+theta_v_2
        yawdot_ = theta_yaw_1*ddv_dot+theta_yaw_2*length+theta_yaw_3*width+theta_yaw_4
        #constraints += [theta_v_1*ddv+theta_v_2 >= -4]
        #constraints += [theta_v_1*ddv+theta_v_2 <= 4]
        #constraints += [theta_yaw_1*ddv+theta_yaw_2 >= -1]
        #constraints += [theta_yaw_1*ddv+theta_yaw_2 <= 1]
    elif EXP == 1:
        vdot_ = theta_v_1*d_1+theta_v_2
        yawdot_ = theta_yaw_1*d_1+theta_yaw_2
    elif EXP == 2:
        vdot_ = theta_v_1*d_1+theta_v_2*ddv+theta_v_3
        yawdot_ = theta_yaw_1*d_1+theta_yaw_2*ddv+theta_yaw_3
    elif EXP == 3:
        ddv_dot = X_train.iloc[i]["DDV_dot"]
        distance_dot = X_train.iloc[i]["distance_dot"]
        #vdot_ = theta_v_1*d_1+theta_v_2*ddv+theta_v_3*ddv_dot+theta_v_4*distance_dot+theta_v_5
        yawdot_ = theta_yaw_1*d_1+theta_yaw_2*ddv+theta_yaw_3*ddv_dot+theta_yaw_4*distance_dot+theta_yaw_5
    #verr = (vdot-vdot_)**2
    yawerr = (yawdot-yawdot_)**2
    #if EXP == 0:
    #    constraints += [verr >= 0]
    #    constraints += [verr <= 4]
    #    constraints += [yawerr >= 0]
    #    constraints += [yawerr <= 1]
    quad += yawerr

objective = cp.Minimize(quad)
problem = cp.Problem(objective, constraints)

# Solve the problem
msk_param_dict = {}
msk_param_dict['MSK_IPAR_PRESOLVE_USE'] = 0
msk_param_dict['MSK_IPAR_NUM_THREADS'] = 0
problem.solve(verbose=False, solver=cp.MOSEK, mosek_params=msk_param_dict)

# Display the results
if EXP == 0 or EXP == 1:
    print("Optimal theta:", theta_yaw_1.value, theta_yaw_2.value, theta_yaw_3.value, theta_yaw_4.value)
elif EXP == 2:
    print("Optimal theta:", theta_v_1.value, theta_v_2.value, theta_v_3.value, theta_yaw_1.value, theta_yaw_2.value, theta_yaw_3.value)
elif EXP == 3:
    print("Optimal theta:", theta_v_1.value, theta_v_2.value, theta_v_3.value, theta_v_4.value, theta_v_5.value, theta_yaw_1.value, theta_yaw_2.value, theta_yaw_3.value, theta_yaw_4.value, theta_yaw_5.value)
N = X_test.shape[0]
verrs = []
yawerrs = []
yawpreds = []
for i in range(N):
    vdot = X_test.iloc[i]["sog_dot"]
    yawdot = X_test.iloc[i]["cog_dot"]
    d_1 = X_test.iloc[i]["distance"]
    ddv = X_test.iloc[i]["DDV"]
    ddv_dot = X_test.iloc[i]["DDV_dot"]
    if EXP == 0:
        #vdot_ = theta_v_1.value*ddv+theta_v_2.value
        yawdot_ = theta_yaw_1.value*ddv_dot+theta_yaw_2.value*length+theta_yaw_3.value*width+theta_yaw_4.value
    elif EXP == 1:
        vdot_ = theta_v_1.value*d_1+theta_v_2.value
        yawdot_ = theta_yaw_1.value*d_1+theta_yaw_2.value
    elif EXP == 2:
        vdot_ = theta_v_1.value*d_1+theta_v_2.value*ddv+theta_v_3.value
        yawdot_ = theta_yaw_1.value*d_1+theta_yaw_2.value*ddv+theta_yaw_3.value
    elif EXP == 3:
        ddv_dot = X_test.iloc[i]["DDV_dot"]
        distance_dot = X_test.iloc[i]["distance_dot"]
        #vdot_ = theta_v_1.value*d_1+theta_v_2.value*ddv+theta_v_3.value*ddv_dot+theta_v_4.value*distance_dot+theta_v_5.value
        yawdot_ = theta_yaw_1.value*d_1+theta_yaw_2.value*ddv+theta_yaw_3.value*ddv_dot+theta_yaw_4.value*distance_dot+theta_yaw_5.value
    #verr = (vdot-vdot_)
    #verrs.append(verr)
    yawerr = (yawdot-yawdot_)
    yawerrs.append(yawerr)
    yawpreds.append(yawdot_)

#print("RMSE v",np.min(verrs),np.mean(verrs),np.max(verrs))
print("RMSE yaw",np.min(yawerrs),np.mean(np.abs(yawerrs)),np.max(yawerrs))
#print(min(X_test["sog_dot"]), max(X_test["sog_dot"]), np.mean(X_test["sog_dot"]), np.std(X_test["sog_dot"]))
print(min(X_test["cog_dot"]), max(X_test["cog_dot"]), np.mean(X_test["cog_dot"]), np.std(X_test["cog_dot"]))

r_squared = r2_score(X_test["cog_dot"], yawpreds)
print("R-squared:", r_squared)

import matplotlib.pyplot as plt

#plt.hist(X_test["sog_dot"], bins = 100, label = "testing_data",alpha=0.5)
#plt.hist(verrs, bins = 100, label = "error",alpha=0.5)
#plt.legend() 
#plt.grid()
#plt.show()

plt.hist(X_test["cog_dot"], bins = 100, label = "testing_data",alpha=0.5)
plt.hist(yawerrs, bins = 100, label = "error",alpha=0.5)
plt.legend() 
plt.grid()
plt.show()



plt.scatter(X_test["cog_dot"], yawpreds)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.show()

N = X_train.shape[0]
verrs = []
yawerrs = []
yawpreds = []
for i in range(N):
    vdot = X_train.iloc[i]["sog_dot"]
    yawdot = X_train.iloc[i]["cog_dot"]
    d_1 = X_train.iloc[i]["distance"]
    ddv = X_train.iloc[i]["DDV"]
    ddv_dot = X_train.iloc[i]["DDV_dot"]
    if EXP == 0:
        #vdot_ = theta_v_1.value*ddv+theta_v_2.value
        yawdot_ = theta_yaw_1.value*ddv_dot+theta_yaw_2.value*length+theta_yaw_3.value*width+theta_yaw_4.value
    elif EXP == 1:
        vdot_ = theta_v_1.value*d_1+theta_v_2.value
        yawdot_ = theta_yaw_1.value*d_1+theta_yaw_2.value
    elif EXP == 2:
        vdot_ = theta_v_1.value*d_1+theta_v_2.value*ddv+theta_v_3.value
        yawdot_ = theta_yaw_1.value*d_1+theta_yaw_2.value*ddv+theta_yaw_3.value
    elif EXP == 3:
        ddv_dot = X_train.iloc[i]["DDV_dot"]
        distance_dot = X_train.iloc[i]["distance_dot"]
        #vdot_ = theta_v_1.value*d_1+theta_v_2.value*ddv+theta_v_3.value*ddv_dot+theta_v_4.value*distance_dot+theta_v_5.value
        yawdot_ = theta_yaw_1.value*d_1+theta_yaw_2.value*ddv+theta_yaw_3.value*ddv_dot+theta_yaw_4.value*distance_dot+theta_yaw_5.value
    #verr = (vdot-vdot_)
    #verrs.append(verr)
    yawerr = (yawdot-yawdot_)
    yawerrs.append(yawerr)
    yawpreds.append(yawdot_)

#print("RMSE v",np.min(verrs),np.mean(verrs),np.max(verrs))
print("RMSE yaw",np.min(yawerrs),np.mean(np.abs(yawerrs)),np.max(yawerrs))
#print(min(X_train["sog_dot"]), max(X_train["sog_dot"]), np.mean(X_train["sog_dot"]), np.std(X_test["sog_dot"]))
print(min(X_train["cog_dot"]), max(X_train["cog_dot"]), np.mean(X_train["cog_dot"]), np.std(X_test["cog_dot"]))

r_squared = r2_score(X_train["cog_dot"], yawpreds)
print("R-squared:", r_squared)
import matplotlib.pyplot as plt

#plt.hist(X_train["sog_dot"], bins = 100, label = "testing_data",alpha=0.5)
#plt.hist(verrs, bins = 100, label = "error",alpha=0.5)
#plt.legend() 
#plt.grid()
#plt.show()

plt.hist(X_train["cog_dot"], bins = 100, label = "training_data",alpha=0.5)
plt.hist(yawerrs, bins = 100, label = "error",alpha=0.5)
plt.legend() 
plt.grid()
plt.show()


plt.scatter(X_train["cog_dot"], yawpreds)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid()
plt.show()