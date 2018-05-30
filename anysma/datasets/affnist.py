"""Affnist dataset for classification.

URL:
    https://www.cs.toronto.edu/~tijmen/affNIST/

LICENCE / TERMS / COPYRIGHT:
    The affNIST dataset is made freely available, without restrictions, to whoever wishes to use it, in the hope that
    it may help advance machine learning research, but without any warranty.

Description:
    The affNIST dataset for machine learning is based on the well-known MNIST dataset. MNIST, however, has become quite
    a small set, given the power of today's computers, with their multiple CPU's and sometimes GPU's. affNIST is made
    by taking images from MNIST and applying various reasonable affine transformations to them. In the process, the
    images become 40x40 pixels large, with significant translations involved, so much of the challenge for the models
    is to learn that a digit means the same thing in the upper right corner as it does in the lower left corner.

Workaround copied from Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils.data_utils import get_file
import numpy as np


def load_data(path='affnist.npz'):
    """Loads the affnist dataset.

    x_train: centered MNIST digits on a 40x40 black background
    x_test: official affNIST test dataset (MNIST digits with random affine transformation)

    # Arguments
        path: path where to cache the dataset locally
            (relative to ~/.keras/datasets).

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    path = get_file(path,
                    origin='http://tiny.cc/anysma_datasets_affnist',
                    file_hash='3bc701960dea4cb33d4b4cdfdcfc5cd3')
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)
