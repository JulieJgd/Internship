import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score


df = pd.read_csv("D:/Documents/TalTech/BaIoT/df.csv")
columns = df.columns
y = df[['Malware']]
X = df[columns.drop('Malware')]

# Splitting the values
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.3)
# print("OK4")


model = keras.Sequential([keras.layers.Dense(4, activation='relu', input_shape=[train_X.shape[1]]),
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

history_1 = model.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=512, epochs=100, verbose=0)
history_df = pd.DataFrame(history_1.history)
history_df['f1_score'] = 2 * (history_df['precision'] * history_df['recall']) / (
            history_df['precision'] + history_df['recall'])
history_df['val_f1_score'] = 2 * (history_df['val_precision'] * history_df['val_recall']) / (
            history_df['val_precision'] + history_df['val_recall'])

# Print Graphs
print("Model raw values")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
history_df.loc[:, ['precision', 'val_precision']].plot()
history_df.loc[:, ['recall', 'val_recall']].plot()
history_df.loc[:, ['f1_score', 'val_f1_score']].plot()

# Use the model and get prediction
def get_f1_score(y_test, prediction):
    a = 2 * precision_score(y_test, prediction.round()) * recall_score(y_test, prediction.round())
    b = precision_score(y_test, prediction.round()) + recall_score(y_test, prediction.round())
    return a / b


prediction = model.predict(val_X)
print("Accuracy:", accuracy_score(val_y, prediction.round()))
print("Precision:", precision_score(val_y, prediction.round()))
print("F1-Score:", get_f1_score(val_y, prediction))
print("Recall:", recall_score(val_y, prediction.round()))
print("Classification report: \n", classification_report(val_y, prediction.round()))
# Show all
plt.show()
