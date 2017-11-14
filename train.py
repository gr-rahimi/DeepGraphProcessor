from keras.preprocessing.image import ImageDataGenerator
import model
import load_data
import keras
from generate_dataset import W,H,dataset_size_train,dataset_size_test



batch_size = 32

x_train, y_train, x_test, y_test = load_data.load_data(dnn_down_scale_factor = 8) # we have 3 max pooling layer 2^3 = 8

#my_model = model.ModelBuilder.build_vgg16()
my_model = model.ModelBuilder.build_custom(input_shape=(256,256,3))
my_model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['mean_squared_error'])

my_model.fit(x_train,y_train, epochs= 50, batch_size = batch_size)


model.save("my_model.h5")
