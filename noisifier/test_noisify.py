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

    print(f"x_train.shape: {x_train.shape}")
    print(f"x_train[0].shape: {x_train[0].shape}")

    noisifier = Noisifier()
    
    y_noisy_1 = noisifier.noisify(y_train, 'pair', 0.45, 10)
    y_noisy_2 = noisifier.noisify(y_train, 'symmetry', 0.2, 10)
    y_noisy_3 = noisifier.noisify(y_train, 'symmetry', 0.5, 10)

def multiLabelNoiseTest():
    y = tf.constant([[1,0,1,0,1,1,0,0,1],[1,0,1,0,1,1,0,0,1],[1,0,1,0,1,1,0,0,1]]) 
    p = tf.constant([[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95],[0.91,0.11,0.92,0.13,0.92,0.83,0.12,0.05,0.95],[0.91,0.11,0.87,0.13,0.92,0.83,0.12,0.05,0.95]], dtype=np.float32)

    noisifier = Noisifier()

    noisy_y = noisifier.random_multi_label_noise(y,2/3,5/9)

    return noisy_y

def add_missing_extra_noise_test():
    y = tf.constant([[0,1,0,0,1],[1,1,1,0,0],[0,0,0,0,1]])
    print(f"y: {y}")

    noisifier = Noisifier()

    missing_y = noisifier.add_missing_noise(y, 0.5)
    print(f"missing_y: {missing_y}")
    
    extra_y = noisifier.add_extra_noise(y,0.5)
    print(f"extra_y: {extra_y}")

    both_y = noisifier.add_mix_noise(y,0.5)
    print(f"both_y: {both_y}")

def classwise_noise_test():
    y = tf.constant([[0,1,0,0,1],[1,1,1,0,0],[0,0,0,0,1],[0,0,0,0,1],[0,0,0,0,1]])
    print(f"y: {y}")

    noisifier = Noisifier()
    classwise_y = noisifier.add_classwise_noise(y, 0.5, seed=True)
    print(f"classwise_y: {classwise_y}")

def main():

    # TEST CIFAR10
    # test_dataset('cifar10')
    # noisy_y = multiLabelNoiseTest()
    classwise_noise_test()

if __name__ == '__main__':
    main()
