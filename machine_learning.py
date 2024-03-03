import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# all_data = pd.read_csv('all_data_1.csv').drop(columns=['id', 'activityType', 'transitionType', 'lat', 'long', 'time', 'lat2', 'long2', 'prediction'])
#
# all_data = all_data[~all_data['transport'].isin(['walking', 'still', 'walking ', 'testing', 'test'])]
#
# randomized_data = all_data.sample(frac=1, random_state=42)
#
# X = randomized_data.drop(columns=['transport'])
#
# y = randomized_data['transport']
#
# model = DecisionTreeClassifier()
# model.fit(X, y)
# predict_data = [
#     # [19.78060531616211, 1.1967654954382334, 5.257773399353027, 2.2861340045928955, 5.379897594451904, 1.0082207918167114, 1014.9027709960938], # bus
#     # [1.2106719017028809, 1.3663483032132298, 3.978463888168335, 3.066884994506836, 6.49955940246582, 1.0053261518478394, 1012.7695922851562], # bus
#     [0.6273741126060486, 0.6007902810731807, 2.9550669193267822, 2.9293830394744873, 8.038862228393555, 1.0052070617675781, 1017.5653686523438], # bus
#     [0.49905794858932495, 0.32080606901700726, 2.8237195014953613, 2.9558942317962646, 8.091769218444824, 1.0071349143981934, 1017.552001953125] # bus
# ]
#
# predictions = model.predict(predict_data)
#
# print(predictions)

# make pands show all the rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# print(predictions)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)
#
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")

# new_data = [...]  # Your new data points
# predictions = clf.predict([[0.037871,0.002964,0.007380,4.242659,8.313830,1.005327,996.467102]])

# print(predictions)

######################################################################### using CSV

all_data = pd.read_csv('all_data_1.csv').drop(columns=['id', 'activityType', 'transitionType', 'lat', 'long', 'time', 'lat2', 'long2', 'prediction'])

all_data = all_data[~all_data['transport'].isin(['walking', 'still', 'walking ', 'testing', 'test'])]

randomized_data = all_data.sample(frac=1, random_state=42)

X = randomized_data.drop(columns=['transport'])

y = randomized_data['transport']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

feature_names = X.columns.tolist()

predict_data = [
    # [19.78060531616211, 1.1967654954382334, 5.257773399353027, 2.2861340045928955, 5.379897594451904, 1.0082207918167114, 1014.9027709960938], # bus
    # [1.2106719017028809, 1.3663483032132298, 3.978463888168335, 3.066884994506836, 6.49955940246582, 1.0053261518478394, 1012.7695922851562], # bus
    [0.6273741126060486, 0.6007902810731807, 2.9550669193267822, 2.9293830394744873, 8.038862228393555, 1.0052070617675781, 1017.5653686523438], # bus
    [0.49905794858932495, 0.32080606901700726, 2.8237195014953613, 2.9558942317962646, 8.091769218444824, 1.0071349143981934, 1017.552001953125], # bus
    [13.722701072692871, 6.0520406342195034, 0.3391350507736206, 4.312935829162598, 7.745648384094238, 1.0188889503479004, 1013.7005615234375] # car

]


predict_data_df = pd.DataFrame(predict_data, columns=feature_names)

predictions = clf.predict(predict_data_df)

unique, counts = np.unique(predictions, return_counts=True)
most_common_transport_amount = np.argmax(counts)
most_common_prediction = unique[most_common_transport_amount]
count_of_most_common = counts[most_common_transport_amount]

print(f"Most common prediction: {most_common_prediction}, Count: {count_of_most_common}")
# print(f"Bus: {counts[0]} Car: {counts[1]}")
print(predictions)