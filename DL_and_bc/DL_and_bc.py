import tensorflow as tf
from tensorflow import keras
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

early_stopping = EarlyStopping(min_delta=0.001,
                               patience=20,
                               restore_best_weights=True)

path = "C:/Users/julie/PycharmProjects/medbiot1/df_bc.csv"

df = pd.read_csv(path)
columns = df.columns
y = df[["Malware"]]  # target
X = df[columns.drop("Malware")]  # features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

model = keras.Sequential([
    layers.Dense(units=4, activation='relu', input_shape=[X_train.shape[1]]),  # hidden layer
    layers.Dropout(0.3),  # apply 30% of dropout on the next layer
    layers.BatchNormalization(),
    layers.Dense(units=4, activation='relu'),  # hidden layer
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(units=1, activation='sigmoid')  # output layer
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy',
                                                                     tf.keras.metrics.Precision(),
                                                                     tf.keras.metrics.Recall()])
history_model = model.fit(X_train, y_train,
                          validation_data=[X_test, y_test],
                          batch_size=128,
                          epochs=100,
                          callbacks=[early_stopping])

# convert the training history to a dataframe
history_df = pd.DataFrame(history_model.history)
history_df['f1_score'] = 2 * (history_df['precision'] * history_df['recall']) / (
            history_df['precision'] + history_df['recall'])
history_df['val_f1_score'] = 2 * (history_df['val_precision'] * history_df['val_recall']) / (
            history_df['val_precision'] + history_df['val_recall'])
# Show in a graph the evolution of the loss
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
history_df.loc[:, ['precision', 'val_precision']].plot()
history_df.loc[:, ['recall', 'val_recall']].plot()
history_df.loc[:, ['f1_score', 'val_f1_score']].plot()


# Use the model and get prediction
def get_f1_score(y_test, prediction):
    a = 2 * precision_score(y_test, prediction.round()) * recall_score(y_test, prediction.round())
    b = precision_score(y_test, prediction.round()) + recall_score(y_test, prediction.round())
    return a / b


prediction = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, prediction.round()))
print("Precision:", precision_score(y_test, prediction.round()))
print("F1-Score:", get_f1_score(y_test, prediction))
print("Recall:", recall_score(y_test, prediction.round()))
print("Classification report: \n", classification_report(y_test, prediction.round()))

plt.show()
