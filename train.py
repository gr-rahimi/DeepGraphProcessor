from keras.preprocessing.image import ImageDataGenerator
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Reshape
#from keras.layers import Activation, Dropout, Flatten, Dense
#from keras.layers import LSTM
import model

batch_size = 32
train_images_count = 10000
test_images_count = 2000


train_datagen = ImageDataGenerator(
        rescale=1./255)
test_datagen = ImageDataGenerator(
    rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode = 'sparse')
#print train_generator.next()
test_generator = test_datagen.flow_from_directory(
        'dataset/test',
        target_size=(224, 224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode = 'sparse')

my_model = model.ModelBuilder.build_vgg16()

my_model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])

my_model.fit_generator(
        train_generator,
        steps_per_epoch= train_images_count // batch_size,
        epochs=200,
        validation_data=test_generator,
        validation_steps=test_images_count // batch_size)


model.save("my_model.h5")
