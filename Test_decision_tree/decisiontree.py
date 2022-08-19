import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

data_test = pd.read_csv("C:/Users/julie/PycharmProjects/medbiot1/df.csv")
filtered_data_set = data_test.dropna(axis=0)
list_columns_filtered = filtered_data_set.columns
y = filtered_data_set.MI_dir_5_weight
features = list_columns_filtered[0:]
# features = list_columns_filtered[0:10]  # reducing the number of features
X = filtered_data_set[features]

# Standardize data
y_scaled = (y - y.mean(axis=0)) / y.std(axis=0)
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

# Splitting the values
train_X, val_X, train_y, val_y = train_test_split(X_scaled, y_scaled, random_state=0, test_size=0.3)


# Function to count the number of common values
def count1(list1, list2):
    s = 0
    for i in range(len(list1)):
        if list1[i] == list2[i]:
            s += 1
    return s


# Function to calculate the accuracy of the model
def accuracy(positive_instances, test_instances):
    return positive_instances / test_instances


# Creating a first model
tree_model = DecisionTreeRegressor(random_state=1)
tree_model.fit(train_X, train_y)
pred_y = tree_model.predict(val_X)

# Transform into lists to count
val_y_list = list(val_y)
pred_y_list = list(pred_y)

# Round the values in the lists
val_y_list_round = list(round(val_y))
pred_y_list_round = [round(val) for val in pred_y_list]

print("Mean absolute error (first model): ", mean_absolute_error(val_y, pred_y))
print("Accuracy not round values (first model):", accuracy(count1(val_y_list, pred_y_list), len(val_y)))
print("Accuracy round values (first model):", accuracy(count1(val_y_list_round, pred_y_list_round), len(val_y)))
