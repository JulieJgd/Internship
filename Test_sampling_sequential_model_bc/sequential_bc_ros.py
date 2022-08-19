import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

df = pd.read_csv("D:/Documents/TalTech/BaIoT/df.csv")
columns = df.columns
y = df[['Malware']]
X = df[columns.drop('Malware')]

ros = RandomOverSampler(random_state=0)
ros.fit(X, y)
X_ros, y_ros = ros.fit_resample(X, y)

# Split data
train_X_ros, val_X_ros, train_y_ros, val_y_ros = train_test_split(X_ros, y_ros, random_state=0, test_size=0.3)


model = keras.Sequential([keras.layers.Dense(4, activation='relu', input_shape=[train_X_ros.shape[1]]),
                                keras.layers.BatchNormalization(),
                                keras.layers.Dropout(rate=0.3),
                                keras.layers.Dense(4, activation='relu'),
                                keras.layers.BatchNormalization(),
                                keras.layers.Dropout(rate=0.3),
                                keras.layers.Dense(4, activation='relu'),
                                keras.layers.BatchNormalization(),
                                keras.layers.Dropout(rate=0.3),
                                keras.layers.Dense(4, activation='relu'),
                                keras.layers.BatchNormalization(),
                                keras.layers.Dropout(rate=0.3),
                                keras.layers.Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy',
                                                                     tf.keras.metrics.Precision(),
                                                                     tf.keras.metrics.Recall()])

history_ros = model.fit(train_X_ros, train_y_ros, validation_data=[val_X_ros, val_y_ros], batch_size=512,
                                epochs=100, verbose=0)
history_ros_df = pd.DataFrame(history_ros.history)
history_ros_df['f1_score'] = 2 * (history_ros_df['precision'] * history_ros_df['recall']) / (
            history_ros_df['precision'] + history_ros_df['recall'])
history_ros_df['val_f1_score'] = 2 * (history_ros_df['val_precision'] * history_ros_df['val_recall']) / (
            history_ros_df['val_precision'] + history_ros_df['val_recall'])

# Print Graphs
print("Model ros")
history_ros_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
history_ros_df.loc[:, ['precision', 'val_precision']].plot()
history_ros_df.loc[:, ['recall', 'val_recall']].plot()
history_ros_df.loc[:, ['f1_score', 'val_f1_score']].plot()

# Use the model and get prediction
def get_f1_score(y_test, prediction):
    a = 2 * precision_score(y_test, prediction.round()) * recall_score(y_test, prediction.round())
    b = precision_score(y_test, prediction.round()) + recall_score(y_test, prediction.round())
    return a / b


prediction = model.predict(val_X_ros)
print("Accuracy:", accuracy_score(val_y_ros, prediction.round()))
print("Precision:", precision_score(val_y_ros, prediction.round()))
print("F1-Score:", get_f1_score(val_y_ros, prediction))
print("Recall:", recall_score(val_y_ros, prediction.round()))
print("Classification report: \n", classification_report(val_y_ros, prediction.round()))
# Show all
plt.show()
