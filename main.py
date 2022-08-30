# Load and combine Syn, Ldap and NetBios data
import pandas as pd
import numpy as np

chunksize = 10 ** 5

synData = pd.DataFrame()
ldapData = pd.DataFrame()
netbiosData = pd.DataFrame()
data = pd.DataFrame()

for chunk in pd.read_csv("data/03-11/Syn.csv", chunksize=chunksize, nrows=1000000):
    synData = synData.append(chunk)

data = data.append(synData)
del synData

for chunk in pd.read_csv("data/03-11/LDAP.csv", chunksize=chunksize, nrows=1000000):
    ldapData = ldapData.append(chunk)

data = data.append(ldapData)
del ldapData

for chunk in pd.read_csv("data/03-11/NetBIOS.csv", chunksize=chunksize, nrows=1000000):
    netbiosData = netbiosData.append(chunk)

data = data.append(netbiosData)
del netbiosData

# - - - - - - - - - -
# Drop NaN and Inf values

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data = data.dropna()

# # - - - - - - - - - -
# Converting data to the right floats, removing unecessary fields
# Convert int64 and str to float 64

import ipaddress

data.replace({'Syn': 1, 'NetBIOS': 1, 'LDAP': 1, 'BENIGN': 0}, inplace=True) # Replace strings
data[' Label'] = data[' Label'].astype(np.float64) # Cast from int64 to float 64

data['SimillarHTTP'] = data['SimillarHTTP'].astype(bool).astype(np.float64) # Replace non-zero with 1

data.drop(['Unnamed: 0'], axis=1, inplace=True) # drop Unnamed: 0 because is just an ID
data.drop(['Flow ID'], axis=1, inplace=True) # drop Flow ID because info is in other fields
data.drop([' Timestamp'], axis=1, inplace=True) # drop timestamp as we have them in order, not necessary

for column in data.columns:
    if data[column].dtypes == np.int64:
        data[column] = data[column].astype(np.float64)
    elif data[column].dtypes == np.float64:
        break
    else:
        for count, item in enumerate(data[column].values):
            data[column].values[count] = np.float64(int(ipaddress.IPv4Address(item)))
        data[column] = data[column].astype(np.float64)

# - - - - - - - - - -
# Scale the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

columns = data.columns[:-1]

data[columns] = scaler.fit_transform(data[columns])

# - - - - - - - - - -
# Split the data into 80% training, 20% testing
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(data, test_size=0.2, random_state=1)

# - - - - - - - - - -
# Here we create x_train, x_test, y_train, y_test as well as oversampling/undersampling data
# due to the large difference in benign and other data

print(df_train[' Label'].value_counts())
count_class_1, count_class_0 = df_train[' Label'].value_counts()

# divide df_train
df_class_0 = df_train[df_train[' Label'] == 0]
df_class_1 = df_train[df_train[' Label'] == 1]

print(df_class_0[' Label'].value_counts())
print(df_class_1[' Label'].value_counts())

# Oversampling
df_class_0_oversample = df_class_0.sample(round(count_class_1 / 10), replace=True)

# Undersampling
size_to_reduce_1_to = round(count_class_1 / 10)
df_class_1_undersample = df_class_1.sample(size_to_reduce_1_to)
count_class_1 = size_to_reduce_1_to

df_train_over_under = pd.concat([df_class_1_undersample, df_class_0_oversample], axis=0)
df_train = df_train_over_under

labels = df_train.columns[:-1]
x_train = df_train[labels]
y_train = df_train[' Label']

x_test = df_test[labels]
y_test = df_test[' Label']

print('Random combined-sampling:')
print(df_train_over_under[' Label'].value_counts())

# Reshape the data to be suitable for CNN, (12 by 7 'image' shape)
print(x_train.shape)
x_train_reshaped = x_train.values.reshape(449128, 12, 7)
print(x_train_reshaped.shape)

print(x_test.shape)
x_test_reshaped = x_test.values.reshape(568265, 12, 7)
print(x_test_reshaped.shape)

num_train, height, width = x_train_reshaped.shape
print(num_train)
print(height)
print(width)

num_classes = 1

# Parameters
batch_size 		= 20
num_epochs 		= 1

kernel_size 	= 3
pool_size 		= 2
conv_depth_1 	= 20
conv_depth_2 	= 32

drop_prob_1 	= 0.05
drop_prob_2 	= 0.1

hidden_size 	= 10
hidden_size2 	= 5

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, AveragePooling2D, Dense, Dropout, Flatten
import tensorflow as tf

# Callback to stop when there is no loss or accuracy improvement in 3 epochs
callback_loss = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
callback_accuracy = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

inp = Input(shape=(height, width, 1))

# CNN
conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)


# Flatten and Fully-connected layer
flat = Flatten()(drop_1)

hidden = Dense(hidden_size, activation='relu')(flat)

drop_3 = Dropout(drop_prob_2)(hidden)

hidden2 = Dense(hidden_size2, activation='relu')(drop_3)

out = Dense(num_classes, activation='sigmoid')(hidden2)

model = Model(inputs=inp, outputs=out)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model using the training set
# Validation_split indicates using fraction 0.1 (10%) for validation
history =  model.fit(x_train_reshaped, y_train, batch_size=batch_size, epochs=num_epochs, verbose=1, validation_split=0.1, shuffle=True, callbacks=[callback_loss, callback_accuracy])

print("- - - Evaluation - - -")
model.evaluate(x_test_reshaped, y_test, verbose=1)  # Evaluate the trained model on the test set!

print(model.summary())

y_pred = model.predict(x_test_reshaped)

from sklearn.metrics import r2_score

r2_value = r2_score(y_test, y_pred.round())
print(r2_value)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred.round()))

import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()
line1 = ax1.plot(history.history["loss"])
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")

fig2, ax2 = plt.subplots()
line2 = ax2.plot(history.history["accuracy"])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")

# Loading and formatting the unknown (UDPLag) data in the same was as the training data

udplagData = pd.DataFrame()

for chunk in pd.read_csv("data/03-11/UDPLag.csv", chunksize=chunksize, nrows=1000000):
    udplagData = udplagData.append(chunk)

udplagData.replace([np.inf, -np.inf], np.nan, inplace=True)
udplagData = udplagData.dropna()

import ipaddress

print(udplagData[' Label'].value_counts())

udplagData.replace({'UDP': 1, 'UDPLag': 1, 'Syn': 1, 'BENIGN': 0}, inplace=True)
udplagData[' Label'] = udplagData[' Label'].astype(np.float64)

udplagData['SimillarHTTP'] = udplagData['SimillarHTTP'].astype(bool).astype(np.float64)

udplagData.drop(['Unnamed: 0'], axis=1, inplace=True)
udplagData.drop(['Flow ID'], axis=1, inplace=True)
udplagData.drop([' Timestamp'], axis=1, inplace=True)

for column in udplagData.columns:
    if udplagData[column].dtypes == np.int64:
        udplagData[column] = udplagData[column].astype(np.float64)
    elif udplagData[column].dtypes == np.float64:
        break
    else:
        for count, item in enumerate(udplagData[column].values):
            udplagData[column].values[count] = np.float64(int(ipaddress.IPv4Address(item)))
        udplagData[column] = udplagData[column].astype(np.float64)


scaler = StandardScaler()
columns = udplagData.columns[:-1]
udplagData[columns] = scaler.fit_transform(udplagData[columns])

x_test_udplag = udplagData[labels]
y_test_udplag = udplagData[' Label']

print(x_test_udplag.shape)
x_test_udplag_reshaped = x_test_udplag.values.reshape(674463, 12, 7)
print(x_test_udplag_reshaped.shape)

y_pred_udplag = model.predict(x_test_udplag_reshaped)

r2_value = r2_score(y_test_udplag, y_pred_udplag.round())
print(r2_value)

print(confusion_matrix(y_test_udplag, y_pred_udplag.round()))

import joblib
joblib.dump(model, 'model.pkl')

