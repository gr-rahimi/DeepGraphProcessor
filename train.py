from keras.preprocessing.image import ImageDataGenerator
import model
import load_data
import keras
from keras.optimizers import Adam
from generate_dataset import W,H,dataset_size_train,dataset_size_test
from keras.callbacks import LearningRateScheduler
import math

batch_size = 32

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 5
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

x_train, y_train, x_test, y_test = load_data.load_data(dnn_down_scale_factor = 8) # we have 3 max pooling layer 2^3 = 8

#my_model = model.ModelBuilder.build_vgg16()
my_model = model.ModelBuilder.build_custom(input_shape=(256,256,3))

adam_optimizer = Adam(lr = 0.0)

my_model.compile(loss='mean_squared_error',
              optimizer= adam_optimizer,
              metrics=['mean_squared_error'])



my_model.fit(x_train,y_train, epochs= 30, batch_size = batch_size,
        validation_data = (x_test, y_test), callbacks =[LearningRateScheduler(step_decay)] )


my_model.save("my_model.h5")
