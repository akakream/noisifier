import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.datasets import mnist
from noisifier import Noisifier
 
def load_dataset(dataset):
    
    NUM_OF_CLASSES = None
    
    if dataset == 'cifar10':
        '''
        32x32
        50k train
        10k test
        '''
        NUM_OF_CLASSES = 10
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    elif dataset == 'cifar100_fine':
        '''
        32x32
        50k train
        10k test
        '''
        NUM_OF_CLASSES = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    elif dataset == 'cifar100_coarse':
        '''
        32x32
        50k train
        10k test
        '''
        NUM_OF_CLASSES = 20
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='coarse')

    elif dataset == 'mnist':
        '''
        28x28
        60k train
        10k test
        '''
        NUM_OF_CLASSES = 10
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, NUM_OF_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_OF_CLASSES)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.0
    x_test /= 255.0
    
    return x_train, y_train, x_test, y_test


def test_dataset(dataset):
    
    x_train, y_train, x_test, y_test = load_dataset(dataset)

    noisifier = Noisifier()
    
    noisifier.noisify(y_train, 'pair', 0.45, 10, 0)
    noisifier.noisify(y_train, 'symmetry', 0.2, 10, 0)
    noisifier.noisify(y_train, 'symmetry', 0.5, 10, 0)


def main():

    #TEST CIFAR10
    test_dataset('cifar10')


if __name__ == '__main__':
    main()
