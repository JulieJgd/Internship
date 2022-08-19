import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score

df = pd.read_csv("D:/Documents/TalTech/BaIoT/df.csv")
columns = df.columns
y = df[['Malware']]
X = df[columns.drop('Malware')]

smote = SMOTE(random_state=0)
smote.fit(X, y)
X_smote, y_smote = smote.fit_resample(X, y)

# Split data
train_X_smote, val_X_smote, train_y_smote, val_y_smote = train_test_split(X_smote, y_smote, random_state=0, test_size=0.3)

model = keras.Sequential([keras.layers.Dense(4, activation='relu', input_shape=[train_X_smote.shape[1]]),
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

history_smote = model.fit(train_X_smote, train_y_smote, validation_data=[val_X_smote, val_y_smote], batch_size=128, epochs=100, verbose=0)
prediction_smote = model.predict(val_X_smote)
history_smote_df = pd.DataFrame(history_smote.history)
history_smote_df['f1_score'] = 2 * (history_smote_df['precision'] * history_smote_df['recall']) / (
            history_smote_df['precision'] + history_smote_df['recall'])
history_smote_df['val_f1_score'] = 2 * (history_smote_df['val_precision'] * history_smote_df['val_recall']) / (
            history_smote_df['val_precision'] + history_smote_df['val_recall'])

print("Model smote")
history_smote_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
history_smote_df.loc[:, ['precision', 'val_precision']].plot()
history_smote_df.loc[:, ['recall', 'val_recall']].plot()
history_smote_df.loc[:, ['f1_score', 'val_f1_score']].plot()

# Use the model and get prediction
def get_f1_score(y_test, prediction):
    a = 2 * precision_score(y_test, prediction.round()) * recall_score(y_test, prediction.round())
    b = precision_score(y_test, prediction.round()) + recall_score(y_test, prediction.round())
    return a / b


prediction = model.predict(val_X_smote)
print("Accuracy:", accuracy_score(val_y_smote, prediction.round()))
print("Precision:", precision_score(val_y_smote, prediction.round()))
print("F1-Score:", get_f1_score(val_y_smote, prediction))
print("Recall:", recall_score(val_y_smote, prediction.round()))
print("Classification report: \n", classification_report(val_y_smote, prediction.round()))


# Show all
plt.show()
