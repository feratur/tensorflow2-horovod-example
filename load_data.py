#!/usr/bin/env python3
import sys
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

if __name__ == '__main__':
    num_workers = int(sys.argv[1])
    print('Downloading CIFAR10 dataset')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    np.savez('data_test.npz', x_test=x_test, y_test=y_test)
    shuffle(x_train, y_train)
    num_samples = x_train.shape[0]
    samples_in_batch = num_samples // num_workers
    print(f'Splitting data for {num_workers} workers')
    for i in range(num_workers):
        x_batch = x_train[i*samples_in_batch:(i+1)*samples_in_batch]
        y_batch = y_train[i*samples_in_batch:(i+1)*samples_in_batch]
        np.savez(f'data_train_{i}.npz', x_train=x_batch, y_train=y_batch)
    print(f'Dataset has been split into {num_workers} parts ({samples_in_batch} samples in each)')
