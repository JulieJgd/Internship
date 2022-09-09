import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, precision_recall_fscore_support
# from tensorflow import metrics
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import randn
from imblearn.over_sampling import SMOTE, RandomOverSampler

path = "D:/Documents/TalTech/BaIoT_new_1/"
list_col = ["Danmini_Doorbell", "Ecobee_Thermostat", "Ennio_Doorbell", "Philips_B120N10_Baby_Monitor",
            "Provision_PT_737E_Security_Camera", "Provision_PT_838_Security_Camera", "Samsung_SNH_1011_N_Webcam",
            "SimpleHome_XCS7_1002_WHT_Security_Camera", "SimpleHome_XCS7_1003_WHT_Security_Camera"]

data = pd.DataFrame()
for i in list_col:
    path_1 = os.path.join(path, i)
    data_bt = pd.read_csv("D:/Documents/TalTech/BaIoT_new_1/" + i + "/benign_traffic.csv")
    data_a = pd.read_csv("D:/Documents/TalTech/BaIoT_new_1/" + i + "/attacks.csv")
    columns = data_bt.columns
    data = data.append(data_bt)
    data = data.append(data_a).sample(frac=1)

y = data[["Malware"]]
X = data[columns.drop("Malware")]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
# print("X_train: ", X_train.shape, "\ny_train: ", y_train.value_counts())
model_test = LinearSVC().fit(X_train, y_train)
prediction = model_test.predict(X_test)
# Determine the minor class for training data ONLY
classes = y_train.value_counts().to_dict()
# Use the dictionnary to get the label after and to get the number of values for each class
minor_class_count = min(classes.values())
major_class_count = max(classes.values())
# Get the exact number of synthetic data which are needed
a = major_class_count - minor_class_count

# The key will be 0 or 1
key = [k for k, v in classes.items() if v == minor_class_count][0][0]
# print(key)

# Only keep minor_class to create synthetic data for training data ONLY
minor_class = data[data["Malware"] == key]
y_minor = minor_class[["Malware"]]
X_minor = minor_class[columns.drop("Malware")]


# GAN for minor_class
# create new data
def generate_latent_points(latent_dim, n_samples):
    x_input = randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input


# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = generator.predict(x_input)
    y = np.zeros((n_samples, 1))
    return X, y


# generate n real samples with class labels; We randomly select n samples from the real data
def generate_real_samples(n):
    X = minor_class.sample(n)  # data.sample(n)
    y = np.ones((n, 1))
    return X, y


def define_generator(latent_dim, n_outputs=9):
    model = Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(n_outputs, activation='linear'))
    return model


def define_discriminator(n_inputs=9):
    model = Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


# create a line plot of loss for the gan and save to file
def plot_history(d_hist, g_hist):
    # plot loss
    plt.subplot(1, 1, 1)
    plt.plot(d_hist, label='d')
    plt.plot(g_hist, label='gen')
    plt.show()
    plt.close()


# train the generator and discriminator
def train(g_model, d_model, gan_model, latent_dim, n_epochs=100, n_batch=128, n_eval=200):  # n_epochs=1000
    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)
    d_history = []
    g_history = []
    # manually enumerate epochs
    for epoch in range(n_epochs):
        # print("ok")
        # prepare real samples
        x_real, y_real = generate_real_samples(half_batch)
        # prepare fake examples
        x_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        # update discriminator
        d_loss_real, d_real_acc = d_model.train_on_batch(x_real, y_real)
        d_loss_fake, d_fake_acc = d_model.train_on_batch(x_fake, y_fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        # prepare points in latent space as input for the generator
        x_gan = generate_latent_points(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = np.ones((n_batch, 1))
        # update the generator via the discriminator's error
        g_loss_fake = gan_model.train_on_batch(x_gan, y_gan)
        # print('>%d, d1=%.3f, d2=%.3f d=%.3f g=%.3f' % (epoch + 1, d_loss_real, d_loss_fake, d_loss, g_loss_fake))
        # d_history.append(d_loss)
        # g_history.append(g_loss_fake)
        # plot_history(d_history, g_history)
        g_model.save('trained_generated_model.h5')


# size of the latent space
latent_dim = X_minor.shape[1]
# create the discriminator
discriminator = define_discriminator(n_inputs=116)
# create the generator
generator = define_generator(latent_dim, n_outputs=116)
# create the gan
gan_model = define_gan(generator, discriminator)
# train model
train(generator, discriminator, gan_model, latent_dim)
#
from keras.models import load_model

model = load_model('trained_generated_model.h5')

latent_points = generate_latent_points(115, a)
X_1 = model.predict(latent_points)
data_fake = pd.DataFrame(data=X_1, columns=columns)

if key == 1:
    data_fake["Malware"] = 1  # key
else:
    data_fake["Malware"] = 0  # key
features = columns.drop("Malware")
label = ['Malware']
X_fake_created = data_fake[features]
y_fake_created = data_fake[label]

X_GAN_train = X_train.append(X_fake_created, sort=False)
y_GAN_train = y_train.append(y_fake_created)
# print("X_GAN_train: ", X_GAN_train.shape, "\ny_GAN_train: ", y_GAN_train.value_counts())
clf_GAN = LinearSVC().fit(X_GAN_train, y_GAN_train)
y_GAN_pred = clf_GAN.predict(X_test)

# SMOTE
smote = SMOTE(random_state=0, sampling_strategy=1, k_neighbors=10)
smote.fit(X_train, y_train)
X_SMOTE_train, y_SMOTE_train = smote.fit_resample(X_train, y_train)
clf_SMOTE = LinearSVC().fit(X_SMOTE_train, y_SMOTE_train)
y_SMOTE_pred = clf_SMOTE.predict(X_test)

# Oversampling
ros = RandomOverSampler(random_state=0)
ros.fit(X_train, y_train)
X_ros_train, y_ros_train = ros.fit_resample(X_train, y_train)
clf_ros = LinearSVC().fit(X_ros_train, y_ros_train)
y_ros_pred = clf_ros.predict(X_test)

# Undersampling
rus = RandomUnderSampler(random_state=0)
rus.fit(X_train, y_train)
X_rus_train, y_rus_train = rus.fit_resample(X_train, y_train)
clf_rus = LinearSVC().fit(X_rus_train, y_rus_train)
y_rus_pred = clf_rus.predict(X_test)

# Results
# print(i)
# print("Accuracy of real data model:", accuracy_score(y_test, prediction))
# print("Precision:", precision_score(y_test, prediction))
# print("Classification report of real data model: \n", classification_report(y_test, prediction))
#
# print("GAN", i)
# print("Accuracy of new data model:", accuracy_score(y_test, y_GAN_pred))
# print("Precision:", precision_score(y_test, y_GAN_pred))
# print("Classification report of new data model: \n", classification_report(y_test, y_GAN_pred))
#
# print("SMOTE", i)
# print("Accuracy of real data model:", accuracy_score(y_test, y_SMOTE_pred))
# print("Precision:", precision_score(y_test, y_SMOTE_pred))
# print("Classification report of real data model: \n", classification_report(y_test, y_SMOTE_pred))
#

def pandas_classification_report(y_true, y_pred, test="Normal"):
    metrics_summary = precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred)

    avg = list(precision_recall_fscore_support(
        y_true=y_true,
        y_pred=y_pred,
        average='weighted'))

    metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
    class_report_df = pd.DataFrame(
        list(metrics_summary),
        index=metrics_sum_index)

    support = class_report_df.loc['support']
    total = support.sum()
    avg[-1] = total

    class_report_df['avg / total'] = avg
    class_report_df['accuracy'] = accuracy_score(y_true, y_pred)
    class_report_df['Test'] = test

    return class_report_df.T

df_report = pandas_classification_report(y_test, prediction)
df_report_SMOTE = pandas_classification_report(y_test, y_SMOTE_pred, test="SMOTE")
df_report_GAN = pandas_classification_report(y_test, y_GAN_pred, test="GAN")
df_report_ros = pandas_classification_report(y_test, y_ros_pred, test="Oversampling")
df_report_rus = pandas_classification_report(y_test, y_rus_pred, test="Undersampling")
df_report_full = df_report.append(df_report_SMOTE)
df_report_full = df_report_full.append(df_report_GAN)
df_report_full = df_report_full.append(df_report_ros)
df_report_full = df_report_full.append(df_report_rus)


df_report_full.to_csv('D:/Documents/TalTech/BaIoT_new_1/Results/Test_6/results_test_6.csv', sep=';')
