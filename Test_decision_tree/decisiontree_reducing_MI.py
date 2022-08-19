import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_regression

data_test = pd.read_csv("C:/Users/julie/PycharmProjects/medbiot1/df_bc.csv")
data_set = data_test.dropna(axis=0)
list_columns_filtered = data_set.columns
y = data_set.MI_dir_5_weight
features = list_columns_filtered[0:]
X = data_set[features]

# Standardize data
y_scaled = (y - y.mean(axis=0)) / y.std(axis=0)
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)


# Checking the MI of the features
def make_mi_scores(X, y):
    X = X.astype(int)
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()
    discrete_features = X.dtypes == int
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


# Get the MI scores to determine a limit
mi_scores = make_mi_scores(X, y)


# print(mi_scores)


# Create a new list of features where MI>limit (default limit: 0.5)
def MI_features(mi_scores, limit=0.5):
    mi_features = list(mi_scores[mi_scores >= limit].index)
    return (mi_features)


X_mi = X[MI_features(mi_scores, limit=0.5)]
X_mi_scaled = X_scaled[MI_features(mi_scores, limit=0.5)]

# Splitting the values
train_X, val_X, train_y, val_y = train_test_split(X_mi_scaled, y, random_state=0, test_size=0.3)


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

print("Number of features: ", len(MI_features(mi_scores, limit=0.5)))
print("Mean absolute error (first model): ", mean_absolute_error(val_y, pred_y))
print("Accuracy not round values (first model):", accuracy(count1(val_y_list, pred_y_list), len(val_y)))
print("Accuracy round values (first model):", accuracy(count1(val_y_list_round, pred_y_list_round), len(val_y)))
