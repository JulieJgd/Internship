import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

data_test = pd.read_csv("C:/Users/julie/PycharmProjects/medbiot1/df.csv")
filtered_data_set = data_test.dropna(axis=0)
list_columns_filtered = filtered_data_set.columns
y = filtered_data_set.MI_dir_5_weight
features = list_columns_filtered[0:]
X = filtered_data_set[features]

# Splitting the values
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.3)

#
val_y_list = list(val_y)
val_y_list_round = list(round(val_y))


# Function to count the number of common values
def count_real(list):
    s = 0
    for i in range(len(val_y_list)):
        if list[i] == val_y_list[i]:
            s += 1
    return s


# Function to count the number of common values when we round the values
def count_round(list):
    s = 0
    for i in range(len(val_y_list_round)):
        if list[i] == val_y_list_round[i]:
            s += 1
    return s


# Function to calculate the accuracy of the model
def accuracy(positive_instances):
    return positive_instances / len(val_y)


# Function to calculate the mean absolute error
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    pred_y = model.predict(val_X)
    mae = mean_absolute_error(val_y, pred_y)
    return mae, pred_y


for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae, my_pred_y = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    preds_list = list(my_pred_y)
    preds_list_round = [round(val) for val in preds_list]
    accurate = accuracy(count_real(preds_list))
    accurate_round = accuracy(count_round(preds_list_round))
    print("For max_leaf_nodes :", max_leaf_nodes, "\nMean Absolute Error:", my_mae, "\nAccuracy for real values:", accurate, "\nAccuracy for round "
          "values:", accurate_round, "\n")
