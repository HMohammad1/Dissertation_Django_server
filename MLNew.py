from sktime.transformations.panel.rocket import Rocket
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import LeaveOneGroupOut

import pandas as pd
import numpy as np

# display all the data from pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# load in the csv file
filename = "AllData.csv"
data = pd.read_csv(filename, header=None)

# sort the data intro train and test
y = data[1].values
participant_no = data[0].values
X = data.drop(columns=[0,1]).values.reshape(len(y),1,-1)

# participant_no = data[0].values

# get the data to use for testing
logo = LeaveOneGroupOut()

logo.get_n_splits(X, y, participant_no)

# rocket pipeline to make the feature map and setup classifier
rocket_pipeline_ridge = make_pipeline(
    Rocket(random_state=0),
    StandardScaler(),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
)

# global classifier
Rocket_score_glob = []
for i, (train_index, test_index) in enumerate(logo.split(X, y, participant_no)):
    print(i)
    rocket_pipeline_ridge.fit(X[train_index], y[train_index])

    Rocket_score = rocket_pipeline_ridge.score(X[test_index], y[test_index])
    Rocket_score_glob = np.append(Rocket_score_glob, Rocket_score)


print("Global Model Results")
print(f"mean accuracy: {np.mean(Rocket_score_glob)}")
print(f"standard deviation: {np.std(Rocket_score_glob)}")
print(f"minimum accuracy: {np.min(Rocket_score_glob)}")
print(f"maximum accuracy: {np.max(Rocket_score_glob)}")

# Uncomment below for the personalised classifier and comment the global one (lines 40 - 53)

# Rocket_score_pers = []
# for i, (train_index, test_index) in enumerate(logo.split(X, y, participant_no)):
#     print(i)
#     # print(f"Participant: {participant_no[test_index][0]}")
#     label = y[test_index]
#     X_S = X[test_index]
#
#     # Identify the indices for each class
#     class_0_indices = np.where(label == 'BUS')[0]
#     class_1_indices = np.where(label == 'CAR')[0]
#
#     # Split each class into train and test using indexing
#     class_0_split_index = int(0.66 * len(class_0_indices))
#     class_1_split_index = int(0.66 * len(class_1_indices))
#
#     X_train = np.concatenate((X_S[class_0_indices[:class_0_split_index]], X_S[class_1_indices[:class_1_split_index]]),
#                              axis=0)
#     y_train = np.concatenate(
#         (label[class_0_indices[:class_0_split_index]], label[class_1_indices[:class_1_split_index]]), axis=0)
#
#     X_test = np.concatenate((X_S[class_0_indices[class_0_split_index:]], X_S[class_1_indices[class_1_split_index:]]),
#                             axis=0)
#     y_test = np.concatenate(
#         (label[class_0_indices[class_0_split_index:]], label[class_1_indices[class_1_split_index:]]), axis=0)
#
#     rocket_pipeline_ridge.fit(X_train, y_train)
#
#     Rocket_score_pers = np.append(Rocket_score_pers, rocket_pipeline_ridge.score(X_test, y_test))
#
#
# print("Personalised Model Results")
# print(f"mean accuracy: {np.mean(Rocket_score_pers)}")
# print(f"standard deviation: {np.std(Rocket_score_pers)}")
# print(f"minimum accuracy: {np.min(Rocket_score_pers)}")
# print(f"maximum accuracy: {np.max(Rocket_score_pers)}")


# Test the classifier with new data (8 for each bus and car)
new_data = pd.read_csv("1-bus.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("BUS Accuracy:", nf_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("1-car.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("CAR Accuracy:", f_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("2-bus.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("BUS Accuracy:", nf_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("2-car.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("CAR Accuracy:", f_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("3-bus.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("BUS Accuracy:", nf_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("3-car.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("CAR Accuracy:", f_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("4-bus.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("BUS Accuracy:", nf_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("4-car.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("CAR Accuracy:", f_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("5-bus.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("BUS Accuracy:", nf_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("5-car.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("CAR Accuracy:", f_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("6-bus.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("BUS Accuracy:", nf_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("6-car.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("CAR Accuracy:", f_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("7-bus.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("BUS Accuracy:", nf_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("7-car.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("CAR Accuracy:", f_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("8-bus.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("BUS Accuracy:", nf_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")

new_data = pd.read_csv("8-car.csv", header=None)
X_new = new_data.drop(columns=[0, 1]).values.reshape(len(new_data), 1, -1)

predictions = rocket_pipeline_ridge.predict(X_new)

f_count = predictions.tolist().count('CAR')
nf_count = predictions.tolist().count('BUS')

print("Number of 'CAR':", f_count)
print("Number of 'BUS':", nf_count)
print("CAR Accuracy:", f_count/(f_count+nf_count))
print("---------------------------------------------------------------------------")
