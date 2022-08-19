import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense
from numpy.random import randn


df = pd.read_csv("D:/Documents/TalTech/BaIoT/df.csv")
columns = df.columns
y = df[["Malware"]]
X = df[columns.drop("Malware")]


def get_f1_score(y_test, prediction):
    a = 2 * precision_score(y_test, prediction.round()) * recall_score(y_test, prediction.round())
    b = precision_score(y_test, prediction.round()) + recall_score(y_test, prediction.round())
    return a / b


# Split data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0, test_size=0.3)

test = LinearSVC().fit(train_X, train_y)
prediction = test.predict(val_X)

print("Normal")
print("Accuracy:", accuracy_score(val_y, prediction))
print("Precision:", precision_score(val_y, prediction))
print("F1-Score:", get_f1_score(val_y, prediction))
print("Recall:", recall_score(val_y, prediction))
print("Classification report: \n", classification_report(val_y, prediction))

print("Oversampling")
# Oversampling
ros = RandomOverSampler(random_state=0)
ros.fit(X, y)
X_ros, y_ros = ros.fit_resample(X, y)

# Split data
train_X_ros, val_X_ros, train_y_ros, val_y_ros = train_test_split(X_ros, y_ros, random_state=0, test_size=0.3)

test_ros = LinearSVC().fit(train_X_ros, train_y_ros)
prediction_ros = test_ros.predict(val_X_ros)

# print(test_ros.score(train_X_ros, train_y_ros), "\n", test_ros.score(val_X_ros, val_y_ros))
print("Accuracy:", accuracy_score(val_y_ros, prediction_ros))
print("Precision:", precision_score(val_y_ros, prediction_ros))
print("F1-Score:", get_f1_score(val_y_ros, prediction_ros))
print("Recall:", recall_score(val_y_ros, prediction_ros))
print("Classification report: \n", classification_report(val_y_ros, prediction_ros))

print("Undersampling")
# Undersampling
rus = RandomUnderSampler(random_state=0)
rus.fit(X, y)
X_rus, y_rus = rus.fit_resample(X, y)

# Split data
train_X_rus, val_X_rus, train_y_rus, val_y_rus = train_test_split(X_rus, y_rus, random_state=0, test_size=0.3)

test_rus = LinearSVC().fit(train_X_rus, train_y_rus)
prediction_rus = test_rus.predict(val_X_rus)

# print(test_rus.score(train_X_rus, train_y_rus), "\n", test_rus.score(val_X_rus, val_y_rus))
print("Accuracy:", accuracy_score(val_y_rus, prediction_rus))
print("Precision:", precision_score(val_y_rus, prediction_rus))
print("F1-Score:", get_f1_score(val_y_rus, prediction_rus))
print("Recall:", recall_score(val_y_rus, prediction_rus))
print("Classification report: \n", classification_report(val_y_rus, prediction_rus))

print("Smote")
# Smote
smote = SMOTE(random_state=0)
smote.fit(X, y)
X_smote, y_smote = smote.fit_resample(X, y)

# Split data
train_X_smote, val_X_smote, train_y_smote, val_y_smote = train_test_split(X_smote, y_smote, random_state=0, test_size=0.3)

test_smote = LinearSVC().fit(train_X_smote, train_y_smote)
prediction_smote = test_smote.predict(val_X_smote)

# print(test_smote.score(train_X_smote, train_y_smote), "\n", test_smote.score(val_X_smote, val_y_smote))
print("Accuracy:", accuracy_score(val_y_smote, prediction_smote))
print("Precision:", precision_score(val_y_smote, prediction_smote))
print("F1-Score:", get_f1_score(val_y_smote, prediction_smote))
print("Recall:", recall_score(val_y_smote, prediction_smote))
print("Classification report: \n", classification_report(val_y_smote, prediction_smote))


print("Gan")
# Only keep minor_class to create synthetic data
minor_class = df[df["Malware"] == 0]
y_minor = minor_class[["Malware"]]
X_minor = minor_class[columns.drop("Malware")]
X_minor_train, X_minor_test, y_minor_train, y_minor_test = train_test_split(X_minor, y_minor, test_size=0.30,
                                                                            random_state=42)


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

# Determine the minor class
classes = df.Malware.value_counts().to_dict()
# Use the dictionnary to get the label after and to get the number of values for each class
minor_class_count = min(classes.values())
major_class_count = max(classes.values())
# Get the exact number of synthetic data which are needed
a = major_class_count - minor_class_count
# size of the latent space
latent_dim = X_minor_train.shape[1]
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

data_fake["Malware"] = 0
features = columns.drop("Malware")
label = ['Malware']
X_fake_created = data_fake[features]
y_fake_created = data_fake[label]

X_GAN = X.append(X_fake_created, sort=False)
y_GAN = y.append(y_fake_created)

X_GAN_train, X_GAN_test, y_GAN_train, y_GAN_test = train_test_split(X_GAN, y_GAN, test_size=0.30, random_state=42)
clf_GAN = LinearSVC().fit(X_GAN_train, y_GAN_train)
prediction_GAN = clf_GAN.predict(X_GAN_test)

print("Accuracy:", accuracy_score(y_GAN_test, prediction_GAN))
print("Precision:", precision_score(y_GAN_test, prediction_GAN))
print("F1-Score:", get_f1_score(y_GAN_test, prediction_GAN))
print("Recall:", recall_score(y_GAN_test, prediction_GAN))
print("Classification report: \n", classification_report(y_GAN_test, prediction_GAN))