import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

# early_stopping = keras.callbacks.EarlyStopping(min_delta=0.1, patience=20, restore_best_weights=True,)
# in the part model.fit add callbacks=[early_stopping]

data_test = pd.read_csv("C:/Users/julie/PycharmProjects/medbiot1/df_bc.csv")
df = data_test.dropna(axis=0)
columns = df.columns
y = df[["Malware"]]  # target
X = df[columns.drop("Malware")]  # features
# print("OK2")

# Standardize data
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
# print("OK3")

# Splitting the values
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.3)
# print("OK4")

# Normal values
# Binary classification
model = keras.Sequential([keras.layers.Dense(4, activation='relu', input_shape=[train_X.shape[1]]),
                          keras.layers.Dense(4, activation='relu'),
                          keras.layers.Dense(1, activation='sigmoid')])
# print("OK5")

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
# print("OK6")

history = model.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=512, epochs=1000, verbose=0)
# print("OK7")
history_df = pd.DataFrame(history.history)
# print("OK8")

# print(history_df)
print("Raw values")
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
print(("Best Validation Loss: {:0.4f}" +
       "\nBest validation Accuracy: {:0.4f}")
      .format(history_df['val_loss'].min(),
              history_df['val_binary_accuracy'].max()))

# Standardized values
# Splitting the values
train_X_s, val_X_s, train_y_s, val_y_s = train_test_split(X_scaled, y, random_state=0, test_size=0.3)

# Binary classification
model_s = keras.Sequential([keras.layers.Dense(4, activation='relu', input_shape=[train_X.shape[1]]),
                            keras.layers.Dense(4, activation='relu'),
                            keras.layers.Dense(1, activation='sigmoid')])

model_s.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

history_s = model_s.fit(train_X_s, train_y_s, validation_data=(val_X_s, val_y_s), batch_size=512, epochs=1000,
                        verbose=0)
history_df_s = pd.DataFrame(history_s.history)

# print(history_df_s)
print("Standardized values")
history_df_s.loc[:, ['loss', 'val_loss']].plot()
history_df_s.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
print(("Best Validation Loss: {:0.4f}" +
       "\nBest validation Accuracy: {:0.4f}")
      .format(history_df_s['val_loss'].min(),
              history_df_s['val_binary_accuracy'].max()))

# More layers
# Binary classification
model_l = keras.Sequential([keras.layers.Dense(4, activation='relu', input_shape=[train_X.shape[1]]),
                            keras.layers.Dense(4, activation='relu'),
                            keras.layers.Dense(4, activation='relu'),
                            keras.layers.Dense(4, activation='relu'),
                            keras.layers.Dense(1, activation='sigmoid')])

model_l.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

history_l = model_l.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=512, epochs=1000,
                        verbose=0)
history_df_l = pd.DataFrame(history_l.history)

# print(history_df_l)
print("More layers")
history_df_l.loc[:, ['loss', 'val_loss']].plot()
history_df_l.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
print(("Best Validation Loss: {:0.4f}" +
       "\nBest validation Accuracy: {:0.4f}")
      .format(history_df_l['val_loss'].min(),
              history_df_l['val_binary_accuracy'].max()))

# Batch Normalization
# Binary classification
model_bn = keras.Sequential([keras.layers.Dense(4, activation='relu', input_shape=[train_X.shape[1]]),
                             keras.layers.BatchNormalization(),
                             keras.layers.Dense(4, activation='relu'),
                             keras.layers.BatchNormalization(),
                             keras.layers.Dense(1, activation='sigmoid')])

model_bn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

history_bn = model_bn.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=512, epochs=1000,
                          verbose=0)
history_df_bn = pd.DataFrame(history_bn.history)

# print(history_df_bn)
print("Batch normalization")
history_df_bn.loc[:, ['loss', 'val_loss']].plot()
history_df_bn.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
print(("Best Validation Loss: {:0.4f}" +
       "\nBest validation Accuracy: {:0.4f}")
      .format(history_df_bn['val_loss'].min(),
              history_df_bn['val_binary_accuracy'].max()))

# Dropout
# Binary classification
model_d = keras.Sequential([keras.layers.Dense(4, activation='relu', input_shape=[train_X.shape[1]]),
                            keras.layers.Dropout(rate=0.3),
                            keras.layers.Dense(4, activation='relu'),
                            keras.layers.Dropout(rate=0.3),
                            keras.layers.Dense(1, activation='sigmoid')])

model_d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

history_d = model_d.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=512, epochs=1000,
                        verbose=0)
history_df_d = pd.DataFrame(history_d.history)

# print(history_df_d)
print("Dropout")
history_df_d.loc[:, ['loss', 'val_loss']].plot()
history_df_d.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
print(("Best Validation Loss: {:0.4f}" +
       "\nBest validation Accuracy: {:0.4f}")
      .format(history_df_d['val_loss'].min(),
              history_df_d['val_binary_accuracy'].max()))

# All
model_all = keras.Sequential([keras.layers.Dense(4, activation='relu', input_shape=[train_X.shape[1]]),
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

model_all.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

history_all = model_all.fit(train_X, train_y, validation_data=(val_X, val_y), batch_size=512, epochs=100, verbose=0)
history_df_all = pd.DataFrame(history_all.history)

# print(history_df_all)
print("All")
history_df_all.loc[:, ['loss', 'val_loss']].plot()
history_df_all.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
print(("Best Validation Loss: {:0.4f}" +
       "\nBest validation Accuracy: {:0.4f}")
      .format(history_df_all['val_loss'].min(),
              history_df_all['val_binary_accuracy'].max()))

# Show everything
plt.show()
