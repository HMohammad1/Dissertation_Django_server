import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('13_02_23.txt', header=None)

filtered_data = data[(data.iloc[:, 8] != 'walking') & (data.iloc[:, 8] != 'walking ') & (data.iloc[:, 8] != 'still') & (data.iloc[:, 8] != 'testing')]

X = filtered_data.iloc[:, 1:-6]

y = filtered_data.iloc[:, 8:-5]

model = DecisionTreeClassifier()
model.fit(X, y)
# predictions = model.predict([[0.037871,0.002964,0.007380,4.242659,8.313830,1.005327,996.467102]])



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

all_data = pd.read_csv('all_data.csv').drop(columns=['id', 'activityType', 'transitionType', 'lat', 'long', 'time', 'lat2', 'long2'])

all_data = all_data[~all_data['transport'].isin(['walking', 'still', 'walking ', 'testing', 'test'])]

X = all_data.drop(columns=['transport'])

y = all_data['transport']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

feature_names = X.columns.tolist()

predict_data = [
    [3.0962629318237305, 0.4297594445606424, -0.10849695652723312, 4.308098316192627, 8.819670677185059, 1.0252342224121094, 995.9992065429688], # car
    [3.0962629318237305, 0.4297594445606424, -0.10849695652723312, 4.308098316192627, 8.819670677185059, 1.0252342224121094, 995.9992065429688], # car
    [12.567118644714355, 0.08026108655626572, 1.3032509088516235, 1.8289859294891357, 6.180316925048828, 1.012398600578308, 996.7298583984375], # bus
    [7.208097457885742, 3.6235756378989024, 1.2640049457550049, 1.8894696235656738, 6.253283500671387, 1.0083556175231934, 996.8511962890625], # bus
    [20.73564338684082, 1.519104592849551, -0.13035139441490173, 3.7056143283843994, 8.833455085754395, 1.012909173965454, 997.2237548828125] # car
]

predict_data_df = pd.DataFrame(predict_data, columns=feature_names)

predictions = clf.predict(predict_data_df)

unique, counts = np.unique(predictions, return_counts=True)
most_common_transport_amount = np.argmax(counts)
most_common_prediction = unique[most_common_transport_amount]
count_of_most_common = counts[most_common_transport_amount]

print(f"Most common prediction: {most_common_prediction}, Count: {count_of_most_common}")
print(predictions)