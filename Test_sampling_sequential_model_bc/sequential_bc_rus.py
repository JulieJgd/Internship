import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

# Getting the dataset, identifying the target and the features
df = pd.read_csv("D:/Documents/TalTech/BaIoT/df.csv")
columns = df.columns
y = df[['Malware']]
X = df[columns.drop('Malware')]

# Applying sampling method
rus = RandomUnderSampler(random_state=0)
rus.fit(X, y)
X_rus, y_rus = rus.fit_resample(X, y)

# Split data
train_X_rus, val_X_rus, train_y_rus, val_y_rus = train_test_split(X_rus, y_rus, random_state=0, test_size=0.3)


model = keras.Sequential([keras.layers.Dense(4, activation='relu', input_shape=[train_X_rus.shape[1]]),
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

history_rus = model.fit(train_X_rus, train_y_rus, validation_data=[val_X_rus, val_y_rus], batch_size=512,
                                epochs=100, verbose=0)
history_rus_df = pd.DataFrame(history_rus.history)
history_rus_df['f1_score'] = 2 * (history_rus_df['precision'] * history_rus_df['recall']) / (
            history_rus_df['precision'] + history_rus_df['recall'])
history_rus_df['val_f1_score'] = 2 * (history_rus_df['val_precision'] * history_rus_df['val_recall']) / (
            history_rus_df['val_precision'] + history_rus_df['val_recall'])

# Print Graphs
print("Model rus")
history_rus_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
history_rus_df.loc[:, ['precision', 'val_precision']].plot()
history_rus_df.loc[:, ['recall', 'val_recall']].plot()
history_rus_df.loc[:, ['f1_score', 'val_f1_score']].plot()

# Use the model and get prediction
def get_f1_score(y_test, prediction):
    a = 2 * precision_score(y_test, prediction.round()) * recall_score(y_test, prediction.round())
    b = precision_score(y_test, prediction.round()) + recall_score(y_test, prediction.round())
    return a / b


prediction = model.predict(val_X_rus)
print("Accuracy:", accuracy_score(val_y_rus, prediction.round()))
print("Precision:", precision_score(val_y_rus, prediction.round()))
print("F1-Score:", get_f1_score(val_y_rus, prediction))
print("Recall:", recall_score(val_y_rus, prediction.round()))
print("Classification report: \n", classification_report(val_y_rus, prediction.round()))
# Show all
plt.show()

