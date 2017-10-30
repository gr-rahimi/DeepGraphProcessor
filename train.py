from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import LSTM


batch_size = 16

train_datagen = ImageDataGenerator(
        rescale=1./255)
test_datagen = ImageDataGenerator(
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size)

test_generator = test_datagen.flow_from_directory(
        'dataset',
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=batch_size)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.summary()
#model.add(Flatten())
model.add(Reshape((17*17*64,1)))

model.add(LSTM(200,))
model.add(Dense(35, activation='linear'))
model.summary()

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=50,
        validation_data=test_generator,
        validation_steps=800 // batch_size)
