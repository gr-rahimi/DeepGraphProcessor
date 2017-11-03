from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import LSTM
from keras.applications.vgg16 import VGG16

class ModelBuilder:
    def __init__(self):
        pass

    @staticmethod
    def build_custom():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(1, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Reshape((26*26*1,1)))

        model.add(LSTM(200,))
        
        model.add(Dense(1, activation='linear'))
        
        model.summary()
        return model


    @staticmethod
    def build_vgg16(input_shape = (224,224,3) ):
        vgg = VGG16(include_top = False, weights = None, input_shape= input_shape,\
                pooling = 'avg')
        #vgg.summary()
        lstm_input = Reshape((512,1))(vgg.output)
        lstm_output = LSTM(200)(lstm_input)
        
        model_output = Dense(1, activation = "linear")(lstm_output)

        modified_vgg_model = Model(input = vgg.input, output = model_output) 
        vgg.summary()

        return modified_vgg_model





