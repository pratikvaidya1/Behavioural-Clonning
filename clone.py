import cv2
import numpy as np
import csv
import random
from matplotlib import pyplot as plt
from sklearn.utils import shuffle


def flip(image, angle):
  new_image = cv2.flip(image,1)
  new_angle = angle*(-1)
  return new_image, new_angle



def data_generator(image_paths, angles, batch_size=32):
    x = np.zeros((batch_size, 160, 320, 3), dtype=np.uint8)
    y = np.zeros(batch_size)
    while True:
        data, angle = shuffle(image_paths, angles)
        for i in range(batch_size):
            choice = int(np.random.choice(len(data), 1))
            x[i] = cv2.imread(data[choice])
            y[i] = angle[choice]
            # Flip random images#
            flip_coin = random.randint(0, 1)
            if flip_coin == 1:
                x[i], y[i] = flip(x[i], y[i])
        yield x, y


# THis code is referenced from the udacity lectures and slightly modified as required.

images = []
images_paths = []
measurements = []
files = []

# there are several laps recorded 
files.append('data')
files.append('recorded data/track 1/corners/')
files.append('recorded data/track 1/3')
files.append('recorded data/track 1/4')


# getting throttle and angle data associated with the images.
for f in files:
    lines = []
    with open(f+'/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    for line in lines[1:]:
        center_path = line[0]
        left_path = line[1]
        right_path = line[2]

        center_filename = (center_path.split('/')[-1]).split('\\')[-1]
        left_filename = (left_path.split('/')[-1]).split('\\')[-1]
        right_filename = (right_path.split('/')[-1]).split('\\')[-1]

        center_current_path = f + '/IMG/' + center_filename
        left_current_path = f + '/IMG/' + left_filename
        right_current_path = f + '/IMG/' + right_filename

        center_image = cv2.imread(center_current_path)[...,::-1]
        left_image = cv2.imread(left_current_path)[...,::-1]
        right_image = cv2.imread(right_current_path)[...,::-1]

        measurement = float(line[3])

        flip_coin = random.randint(0, 1)
        if flip_coin == 1:
            flipped_center_image, flipped_measurement = flip(center_image, measurement)
            flipped_center_image = cv2.resize(flipped_center_image[40:140, :], (64, 64))
            images.append(flipped_center_image)
            images_paths.append(center_current_path)
            measurements.append(flipped_measurement)
        
        center_image = cv2.resize(center_image[40:140, :], (64, 64))
        images.append(center_image)
        images_paths.append(center_current_path)
        measurements.append(measurement)

        left_image = cv2.resize(left_image[40:140, :], (64, 64))
        images.append(left_image)
        images_paths.append(left_current_path)
        measurements.append(measurement + 0.22)

        right_image = cv2.resize(right_image[40:140, :], (64, 64))
        images.append(right_image)
        images_paths.append(right_current_path)
        measurements.append(measurement - 0.22)

X_train = np.array(images)
y_train = np.array(measurements)


# creating model in Keras.
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout, Cropping2D, Activation
from keras.optimizers import Adam
from keras.layers.advanced_activations import ELU
from sklearn.model_selection import train_test_split

# start with the LeNet model discussed in the class.
def LeNet():
    LeNet = Sequential()
    # LeNet.add(Cropping2D(cropping=((40,20),(0,0)), input_shape=(160,320,3)))
    LeNet.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(64,64,3)))
    LeNet.add(Convolution2D(6, 5, 5, subsample=(1, 1), border_mode='valid', activation='elu'))
    LeNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid'))
    LeNet.add(Convolution2D(16, 5, 5, subsample=(1, 1), border_mode="valid", activation='elu'))
    LeNet.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='valid'))
    LeNet.add(Flatten())
    LeNet.add(Dropout(0.5))
    LeNet.add(Dense(120, activation='elu'))
    LeNet.add(Dropout(0.5))
    LeNet.add(Dense(84, activation='elu'))
    LeNet.add(Dense(10, activation='elu'))
    LeNet.add(Dense(1))
    return LeNet


# NVIDIA model is referenced from nvidia site.. Please find writeup for this.
def Nvidia(input_shape=(160, 320, 3)):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, name="image_normalization", input_shape=input_shape))
    model.add(Convolution2D(24, 5, 5, name="convolution_1", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, name="convolution_2", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, name="convolution_3", subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3, name="convolution_4", border_mode="valid", init='he_normal'))
    model.add(ELU())
    # model.add(Convolution2D(64, 3, 3, name="convolution_5", border_mode="valid", init='he_normal'))
    # model.add(ELU())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(100, name="hidden1", init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(50, name="hidden2", init='he_normal'))
    model.add(ELU())
    model.add(Dense(10, name="hidden3", init='he_normal'))
    model.add(ELU())
    model.add(Dense(1, name="steering_angle", activation="linear"))

    return model



# using NVIDIA model for training.
model = Nvidia((64,64,3))
adam = Adam(lr=0.001)
model.compile(optimizer=adam, loss='mse')
model.summary()
epochs = 20
batch_size = 512
model.fit(x=X_train, y=y_train, nb_epoch=epochs, batch_size=batch_size,  validation_split=0.2, shuffle=True)

#saving the model according to epochs and batch size to visualize and fine tune the model.
model.save('Nvidia_'+str(epochs)+'e_'+str(batch_size)+'_022angleOffset.h5')
# images_paths, images_paths_valid, measurements, measurements_valid = train_test_split(images_paths, measurements, test_size = 0.10, random_state = 100)

# data_generator = data_generator(images_paths, measurements, 256)
# valid_generator = data_generator(images_paths_valid, measurements_valid, 25)
# model.fit_generator(data_generator, samples_per_epoch=len(images_paths), nb_epoch=10)#), validation_data=(images_paths_valid, measurements_valid), nb_val_samples=len(images_paths_valid))




