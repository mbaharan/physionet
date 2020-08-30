#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physionet ECG classification
@author: Andreas Werdich
Batch generator class
Modified from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
"""
# %% Imports

import numpy as np
import torch
from torch.utils.data import Dataset
from physionet_processing import (
    zero_filter, extend_ts, random_resample, spectrogram, norm_float)

# %% Batch generator class


class DataGenerator(Dataset):
    'Generates data for pytorch dataloader'

    def __init__(self, h5file, list_IDs, labels, batch_size=32, dim=(178, 33),
                 nperseg=64, noverlap=32, data_mean=-9.01, data_std=9.00,
                 n_channels=1, sequence_length=5736,
                 n_classes=4, shuffle=True, augment=False):
        'Initialization'
        self.h5file = h5file
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.data_mean = data_mean
        self.data_std = data_std
        self.n_channels = n_channels
        self.sequence_length = sequence_length
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return torch.from_numpy(X), torch.from_numpy(y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim,
                      self.n_channels), dtype=float)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            data = extend_ts(
                self.h5file[ID]['ecgdata'][:, 0], self.sequence_length)
            data = np.reshape(data, (1, len(data)))

            if self.augment:

                # dropout bursts
                data = zero_filter(data, threshold=2, depth=10)

                # random resampling
                data = random_resample(data)

            # Generate spectrogram
            data_spectrogram = spectrogram(
                data, nperseg=self.nperseg, noverlap=self.noverlap)[2]

            # Normalize
            data_transformed = norm_float(
                data_spectrogram, self.data_mean, self.data_std)

            X[i, ] = np.expand_dims(data_transformed, axis=3)

            # Assuming that the dataset names are unique (only 1 per label)
            y[i] = self.labels[ID]
        
        X = np.reshape(X, (self.batch_size, self.n_channels, 1, self.dim[0]*self.dim[1]))
        return X, y


# Run as Script
if __name__ == '__main__':
    import os
    import h5py
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from physionet_processing import (fetch_h5data, spectrogram,
                                      special_parameters, transformed_stats)

    data_root = '/mnt/4tb/ECG'
    hd_file = os.path.join(data_root, 'physio.h5')
    label_file = os.path.join('./physionet/', 'REFERENCE-v3.csv')

    # Open hdf5 file
    h5file = h5py.File(hd_file, 'r')

    # Get a list of dataset names
    dataset_list = list(h5file.keys())

    # Load the labels
    label_df = pd.read_csv(label_file, header=None, names=['name', 'label'])
    # Filter the labels that are in the small demo set
    label_df = label_df[label_df['name'].isin(dataset_list)]

    # Encode labels to integer numbers
    label_set = list(sorted(label_df.label.unique()))
    encoder = LabelEncoder().fit(label_set)
    label_set_codings = encoder.transform(label_set)
    label_df = label_df.assign(encoded=encoder.transform(label_df.label))

    print('Unique labels:', encoder.inverse_transform(label_set_codings))
    print('Unique codings:', label_set_codings)
    print('Dataset labels:\n', label_df.iloc[100:110, ])

    # Split the IDs in training and validation set
    test_split = 0.33
    idx = np.arange(label_df.shape[0])
    id_train, id_val, _, _ = train_test_split(idx, idx,
                                              test_size=test_split,
                                              shuffle=True,
                                              random_state=123)

    # Store the ids and labels in dictionaries
    partition = {'train': list(label_df.iloc[id_train, ].name),
                 'validation': list(label_df.iloc[id_val, ].name)}

    labels = dict(zip(label_df.name, label_df.encoded))

    # Maximum sequence length
    max_length = 18286

    # Output dimensions
    sequence_length = max_length
    spectrogram_nperseg = 64  # Spectrogram window
    spectrogram_noverlap = 32  # Spectrogram overlap
    n_classes = len(label_df.label.unique())
    batch_size = 32

    # calculate image dimensions
    data = fetch_h5data(h5file, [0], sequence_length)
    _, _, Sxx = spectrogram(
        data, nperseg=spectrogram_nperseg, noverlap=spectrogram_noverlap)
    dim = Sxx[0].shape
    params = {'batch_size': batch_size,
              'dim': dim,
              'nperseg': spectrogram_nperseg,
              'noverlap': spectrogram_noverlap,
              'n_channels': 1,
              'sequence_length': sequence_length,
              'n_classes': n_classes,
              'shuffle': True}

    train_generator = DataGenerator(
        h5file, partition['train'], labels, augment=True, **params)
    val_generator = DataGenerator(
        h5file, partition['validation'], labels, augment=False, **params)

    for i in range(len(train_generator)):
        X, y = train_generator[i]
        print(X.shape)
