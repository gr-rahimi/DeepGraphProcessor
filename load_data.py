import numpy as np
from generate_dataset import W,H,dataset_size_train,dataset_size_test
from scipy.misc import imread
import os
import fnmatch
import pickle

train_path = "./dataset/train"
test_path = "./dataset/test"

def load_data(dnn_down_scale_factor = 4):
    x_train = np.empty((dataset_size_train, H, W, 3), dtype='uint8')
    y_train = np.zeros((dataset_size_train,W * H/dnn_down_scale_factor**2), dtype='uint8')

    x_test = np.empty((dataset_size_test, H, W, 3), dtype='uint8')
    y_test = np.zeros((dataset_size_test, W * H / dnn_down_scale_factor ** 2), dtype='uint8')

    train_count = __read_images(train_path, x_train, y_train, dnn_down_scale_factor)
    print train_count , " train files read successfully"

    test_count = __read_images(test_path, x_test, y_test, dnn_down_scale_factor)
    print test_count, " test files read successfully"

    y_train = np.expand_dims(y_train, 2)
    y_test = np.expand_dims(y_test, 2)

    return x_train/255.0, y_train, x_test/255.0, y_test




def __read_images(path, x, y, dnn_down_scale_factor):

    file_count = 0
    for path, subdirs, files in os.walk(path):
        for file_name in fnmatch.filter(files, '*.png'):
            x[file_count, : , : , :] = imread(os.path.join(path, file_name),mode = "RGB")
            pos = pickle.load( open( os.path.join(path, file_name[:-3]+ 'pickle'), "rb" ))
            pos = np.array(pos)
            scale = np.array([[H, W]])
            pos = pos * scale / dnn_down_scale_factor
            pos = pos.round()
            pos = pos.dot(np.array([[float(W)/dnn_down_scale_factor],[1]])).round()
            pos = np.sort(pos, axis=0)

            sum =0;
            for p in pos:
                sum += 1
                y[file_count, int (p) :] = sum
            #print y[file_count, 700:1200]
            file_count += 1
    return file_count


if __name__ == "__main__":
    load_data()